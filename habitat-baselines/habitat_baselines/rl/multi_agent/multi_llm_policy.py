from __future__ import annotations
import os
import datetime
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from hydra.core.hydra_config import HydraConfig
from openai.types.chat.chat_completion import ChatCompletionMessage

import gym.spaces as spaces
import numpy as np
import torch
from habitat_mas.agents.actions.discussion_actions import *
from habitat_mas.utils import AgentArguments
from habitat_mas.utils.models import OpenAIModel

from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.multi_agent.pop_play_wrappers import (
    MultiAgentPolicyActionData,
    MultiPolicy,
    MultiStorage,
    MultiUpdater,
    _merge_list_dict,
)
from habitat_baselines.rl.multi_agent.utils import (
    add_agent_names,
    add_agent_prefix,
    update_dict_with_agent_prefix,
)


ROBOT_RESUME_TEMPLATE = (
    '"{robot_key}" is a  "{robot_type}" agent.'
    " It has the following capabilities:\n\n"
    '"""\n{capabilities}\n"""\n\n'
)

def create_leader_prompt(robot_resume):

    LEADER_SYSTEM_PROMPT_TEMPLATE = (
        "You are a group discussion leader."
        " Your have to discuss with real robots (agents) to break an overall"
        " task to smaller subtasks and assign a subtask to each agent."
        "The agents are described below:\n\n"
        '"""\n{robot_resume}\n"""\n\n'
        )
    FORMAT_INSTRUCTION = (
        "You should assign subtasks to each agent based on their capabilities, following this format:\n\n"
        r"{robot_id||subtask_description}\n\n"
        'For example, if you want to assign a subtask "Navigate to box_1" to agent_0, you should type:\n'
        r"{agent_0||Navigate to box_1}\n\n"
        "Remember you must include the brackets and you MUST include all the robots in your response.\n"
        'If you think a robot should not have a subtask, you can assign it "Nothing to do".\n'
    )
    
    return LEADER_SYSTEM_PROMPT_TEMPLATE.format(
        robot_resume=robot_resume) + FORMAT_INSTRUCTION


def create_leader_start_message(task_description, scene_description):

    LEADER_MESSAGE_TEMPLATE = (
        " The task is described below:\n\n"
        '"""\n{task_description}\n"""\n\n'
        "All agents are in a scene, the scene description is as follows:\n\n"
        '"""\n{scene_description}\n"""\n\n'
        "Now you should assign subtasks to each agent based on their capabilities, following the format in the system prompt."
    )
    
    return LEADER_MESSAGE_TEMPLATE.format(
        task_description=task_description, 
        scene_description=scene_description) # + FORMAT_INSTRUCTION

def create_robot_prompt(robot_type, robot_key, capabilities):

    ROBOT_GROUP_DISCUSS_SYSTEM_PROMPT_TEMPLATE = (
        # 'You are a "{robot_type}" robot with id "{robot_id}".'
        'You are a "{robot_type}" agent called "{robot_key}".'
        "You have the following capabilities:\n\n"
        '"""\n{capabilities}\n"""\n\n'
        # "The physics capabilities include its mobility, perception, and manipulation capabilities."
        ' You will receive a subtask from the leader. If you don\'t have any task to do, you will receive "Nothing to do". '
        " Your task is to check if you are able to complete the assigned task by common sense reasoning and if targets is within the range of your sensors and workspace."
        ' You can generate python code to check if task is feasible numerically.'
        ' Please pay attention to the shape type of the robot workspace when your generate code to check if a location is within the 3D space bounded by the shape.'
        ' I will execute the code and give your the result to help you make decisions.'
    )
    FORMAT_INSTRUCTION = (
        r' Please put all your code in a single python code block wrapped within ```python and ```.'
        r' You MUST print the varible with "<name>: <value>" format you want to know in the code.'
        r" Finally, if you think the task is incorrect, you can explain the reason and ask the leader to modify it,"
        r' following this format: "{{no||<the reason>}}".'
        r' If you think the task is correct, you should confirm it by typing "{{yes}}".'
        r" Example responses: {{yes}}, {{no||I have no moving ability}}, {{no||The object is out of my arm workspace}}."
    )
    return ROBOT_GROUP_DISCUSS_SYSTEM_PROMPT_TEMPLATE.format(
        robot_type=robot_type,
        robot_key=robot_key,
        capabilities=capabilities) + FORMAT_INSTRUCTION

def create_robot_start_message(task_description, scene_description, compute_path: bool = False):

    ROBOT_GROUP_DISCUSS_MESSAGE_TEMPLATE = (
        " Your task is to work with other agents to complete the assigned subtask described below:\n\n"
        '"""\n{task_description}\n"""\n\n'
        "The scene description is as follows:\n\n"
        '"""\n{scene_description}\n"""\n\n'
    )
    COMPUTE_PATH = (
        "Please infer your navigation path according to regions description. "
        "You should determine whether you can succeed based on your navigation path and capabilities."
    )
    if compute_path:
        ROBOT_GROUP_DISCUSS_MESSAGE_TEMPLATE += COMPUTE_PATH

    return ROBOT_GROUP_DISCUSS_MESSAGE_TEMPLATE.format(
        task_description=task_description, 
        scene_description=scene_description)


NO_MANIPULATION = "There are no explicit manipulation components"
NO_MOBILITY = "The provided URDF does not include specific joints"
NO_PERCEPTION = "UNKNOWN"


def get_text_capabilities(robot: dict):
    capabilities = ""
    if "mobility" in robot:
        mobility = json.dumps(robot["mobility"]["summary"])
        capabilities += f"Mobility capbility: {mobility}\n"
    if "perception" in robot:
        perception = json.dumps(robot["perception"]["summary"])
        capabilities += f"Perception capbility: {perception}\n"
    if "manipulation" in robot:
        manipulation = json.dumps(robot["manipulation"]["summary"])
        capabilities += f"Manipulation capbility: {manipulation}\n"
    return capabilities


def get_full_capabilities(robot: dict):
    capabilities = "The followinfg is a list of python dicts describing the robot's capabilities:\n"
    if "mobility" in robot:
        mobility = json.dumps(robot["mobility"])
        capabilities += f" - Mobility capability: {mobility}\n"
    if "perception" in robot:
        perception = json.dumps(robot["perception"])
        capabilities += f" - Perception capability: {perception}\n"
    if "manipulation" in robot:
        manipulation = json.dumps(robot["manipulation"])
        capabilities += f" - Manipulation capability: {manipulation}\n"
    return capabilities

def parse_leader_response(text):
    # Define the regular expression pattern
    pattern = r"\{(.*?)\|\|(.*?)\}"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Create a dictionary from the matches
    robot_tasks = {
        robot_id: subtask_description for robot_id, subtask_description in matches
    }

    return robot_tasks


def parse_agent_response(text):
    # Define the regular expression pattern
    pattern = r"\{(yes|no)(?:\|\|(.*?))?\}"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    if len(matches) == 0:
        print("No match found in agent response: ", text)
        return "no", text
    response, reason = matches[0]
    if response == "yes":
        reason = None  # Use None to indicate no reason
    elif response == "no":
        reason = (
            reason.strip() if reason else ""
        )  # Store the reason, strip spaces, or empty if none

    return response, reason


# DISCUSSION_TOOLS = [eval_python_code, add, subtract, multiply, divide]
DISCUSSION_TOOLS = []


def group_discussion(
    robot_resume: dict, scene_description: str, task_description: str, 
    save_chat_history=True, save_chat_history_dir="./chat_history_output", episode_id=-1
) -> dict[str, AgentArguments]:
    compute_path = "regions_description" in scene_description
    robot_resume = json.loads(robot_resume)
    robot_resume_prompt = ""
    for robot_key in robot_resume:
        robot_resume_prompt += ROBOT_RESUME_TEMPLATE.format(
            robot_key=robot_key,
            robot_type=robot_resume[robot_key]["robot_type"],
            capabilities=get_full_capabilities(robot_resume[robot_key]),
        )
    leader_prompt = create_leader_prompt(robot_resume_prompt)
    leader = OpenAIModel(leader_prompt, DISCUSSION_TOOLS, discussion_stage=True, code_execution=False)
    agents: Dict[str, OpenAIModel] = {}
    for robot_key in robot_resume:
        robot_prompt = create_robot_prompt(
            robot_resume[robot_key]["robot_type"],
            robot_key,
            get_full_capabilities(robot_resume[robot_key]),
        )
        agents[robot_key] = OpenAIModel(
            robot_prompt, DISCUSSION_TOOLS, discussion_stage=True, code_execution=True,
        )

    # robot_id_to_model_map = {
    #     robot_resume[robot]["robot_id"]: agents[robot] for robot in robot_resume
    # }
    leader_start_message = create_leader_start_message(
        task_description=task_description, 
        scene_description=scene_description
    )
    response = leader.chat(leader_start_message)
    robot_tasks = parse_leader_response(response)
    print("===============Task Description==============")
    print(task_description)
    print("===============Leader Response==============")
    print(response)
    print("===========================================")

    agent_response = {}
    for robot_id in robot_tasks:
        robot_model = agents[robot_id]
        robot_start_message = create_robot_start_message(
            task_description=robot_tasks[robot_id],
            scene_description=scene_description,
            compute_path=compute_path,
        )
        response = robot_model.chat(robot_start_message)
        agent_response[robot_id] = parse_agent_response(response)
        print("===============Robot Response==============")
        print(f"Robot {robot_id} response: {response}")
        print("===========================================")

    all_yes = True
    prompt = ""
    for robot_id, (response, reason) in agent_response.items():
        if response == "no":
            prompt += f"Robot {robot_id} response: {response}, reason: {reason}\n"
            all_yes = False
            break
    if not all_yes:
        prompt += "Please modify the task and reassign the subtasks."
        response = leader.chat(prompt)
        print("===============Leader Response==============")
        print(response)
        print("===========================================")
        robot_tasks = parse_leader_response(response)

    results = {}
    for agent in agents:
        results[agent] = AgentArguments(
            robot_id=agent,
            robot_type=robot_resume[agent]["robot_type"],
            task_description=task_description,
            subtask_description=robot_tasks[agent],
            chat_history=agents[agent].chat_history,
        )

    print("===============Group Discussion Result==============")
    print(robot_tasks)
    print("====================================================")

    # debug info: save all agent chat history
    if save_chat_history:
        # save dir format: chat_history_output/<date>/<config>/<episode_id>
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        hydra_cfg = HydraConfig.get()
        config_str = hydra_cfg.job.config_name.split("/")[-1].replace(".yaml", "")
        episode_save_dir = os.path.join(save_chat_history_dir, date_str, config_str, str(episode_id))
        
        if not os.path.exists(episode_save_dir):
            os.makedirs(episode_save_dir)

        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, ChatCompletionMessage):
                    # Pydantic models are not serializable by json.dump by default
                    return dict(obj)
                return super().default(obj)

        # save leader chat history
        with open(os.path.join(episode_save_dir, "leader_chat_history.json"), "w") as f:
            full_history = [leader.system_message] + leader.chat_history
            json.dump(full_history, f, indent=2, cls=CustomJSONEncoder)
            
        for agent in agents:
            with open(os.path.join(episode_save_dir, f"{agent}_chat_history.json"), "w") as f:
                agent_chat_history = agents[agent].chat_history
                full_history = [agents[agent].system_message] + agent_chat_history
                json.dump(full_history, f, indent=2, cls=CustomJSONEncoder)

    return results


class MultiLLMPolicy(MultiPolicy):
    """
    Wraps a set of LLM policies. Add group discussion stage before individual policy actions.
    """

    def __init__(self, update_obs_with_agent_prefix_fn):
        self._active_policies = []
        if update_obs_with_agent_prefix_fn is None:
            update_obs_with_agent_prefix_fn = update_dict_with_agent_prefix
        self._update_obs_with_agent_prefix_fn = update_obs_with_agent_prefix_fn

    def set_active(self, active_policies):
        self._active_policies = active_policies

    def on_envs_pause(self, envs_to_pause):
        for policy in self._active_policies:
            policy.on_envs_pause(envs_to_pause)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        envs_text_context=[{}],
        **kwargs,
    ):
        n_agents = len(self._active_policies)
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = rnn_hidden_states.split(
            split_index_dict["index_len_recurrent_hidden_states"], -1
        )
        agent_prev_actions = prev_actions.split(
            split_index_dict["index_len_prev_actions"], -1
        )
        agent_masks = masks.split([1] * n_agents, -1)
        n_envs = prev_actions.shape[0]

        text_goal = observations["pddl_text_goal"][0].tolist()
        text_goal = "".join([chr(code) for code in text_goal]).strip()
        # text_goal = "Fill in a tank with water. The tank is 80cm high, 50cm wide, and 50cm deep. The tank is empty. agent_0 has a 1L backet and agent_1 has a 2L bucket. Use function call to compute how many time agent_1 and agent_2 needs to fill the tank to full."
        # Stage 1: If all prev_actions are zero, which means it is the first step of the episode, then we need to do group discussion
        # Given: Robot resume + Scene description + task instruction
        # Output: (Subtask decomposition) + task assignment
        envs_agent_arguments = []
        for i in range(n_envs):
            env_prev_actions = prev_actions[i]
            env_text_context = envs_text_context[i]
            # if no previous actions, then it is the first step of the episode
            if not env_prev_actions.any():
                if "robot_resume" in env_text_context:
                    robot_resume = env_text_context["robot_resume"]
                if "scene_description" in env_text_context:
                    scene_description = env_text_context["scene_description"]
                if "episode_id" in env_text_context:
                    episode_id = env_text_context["episode_id"]
                # print("===============Group Discussion===============")
                # print(robot_resume)
                # print("=============================================")
                # print(scene_description)
                # print("=============================================")
                # print(text_goal)
                # print("=============================================")
                agent_arguments = group_discussion(
                    robot_resume, scene_description, text_goal, episode_id=episode_id
                )
                envs_agent_arguments.append(agent_arguments)

        # Stage 2: Individual policy actions
        agent_actions = []
        for agent_i, policy in enumerate(self._active_policies):
            # collect assigned tasks for agent_i across all envs
            agent_i_handle = f"agent_{agent_i}"
            select_agent_arguments = [
                agent_arguments[agent_i_handle] if agent_i_handle in arguments else None
                for arguments in envs_agent_arguments
            ]

            agent_obs = self._update_obs_with_agent_prefix_fn(observations, agent_i)

            agent_actions.append(
                policy.act(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                    deterministic,
                    envs_text_context=envs_text_context,
                    agent_arguments=select_agent_arguments,  # pass the task planning result to the policy
                )
            )

        should_terminate = True
        for agent_action in agent_actions:
            if agent_action.skill_id != 2.0:
                should_terminate = False
                break
        if should_terminate:
            for agent_i, policy in enumerate(self._active_policies):
                agent_actions[agent_i].actions[i, policy._stop_action_idx] = 1.0

        policy_info = _merge_list_dict(
            [ac.policy_info for ac in agent_actions]
        )
        batch_size = masks.shape[0]
        device = masks.device

        action_dims = split_index_dict["index_len_prev_actions"]

        # We need to split the `take_actions` if they are being assigned from
        # `actions`. This will be the case if `take_actions` hasn't been
        # assigned, like in a monolithic policy where there is no policy
        # hierarchicy.
        if any(ac.take_actions is None for ac in agent_actions):
            length_take_actions = action_dims
        else:
            length_take_actions = None

        def _maybe_cat(get_dat, feature_dims, dtype):
            all_dat = [get_dat(ac) for ac in agent_actions]
            # Replace any None with dummy data.
            all_dat = [
                torch.zeros((batch_size, feature_dims[ind]), device=device, dtype=dtype)
                if dat is None
                else dat
                for ind, dat in enumerate(all_dat)
            ]
            return torch.cat(all_dat, -1)

        rnn_hidden_lengths = [ac.rnn_hidden_states.shape[-1] for ac in agent_actions]
        return MultiAgentPolicyActionData(
            rnn_hidden_states=torch.cat(
                [ac.rnn_hidden_states for ac in agent_actions], -1
            ),
            actions=_maybe_cat(lambda ac: ac.actions, action_dims, prev_actions.dtype),
            values=_maybe_cat(
                lambda ac: ac.values, [1] * len(agent_actions), torch.float32
            ),
            action_log_probs=_maybe_cat(
                lambda ac: ac.action_log_probs,
                [1] * len(agent_actions),
                torch.float32,
            ),
            take_actions=torch.cat(
                [
                    ac.take_actions if ac.take_actions is not None else ac.actions
                    for ac in agent_actions
                ],
                -1,
            ),
            policy_info=policy_info,
            should_inserts=np.concatenate(
                [
                    ac.should_inserts
                    if ac.should_inserts is not None
                    else np.ones(
                        (batch_size, 1), dtype=bool
                    )  # None for monolithic policy, the buffer should be updated
                    for ac in agent_actions
                ],
                -1,
            ),
            length_rnn_hidden_states=rnn_hidden_lengths,
            length_actions=action_dims,
            length_take_actions=length_take_actions,
            num_agents=n_agents,
        )

    def _build_index_split(self, rnn_hidden_states, prev_actions, kwargs):
        """
        Return a dictionary with rnn_hidden_states lengths and action lengths that
        will be used to split these tensors into different agents. If the lengths
        are already in kwargs, we return them as is, if not, we assume agents
        have the same action/hidden dimension, so the tensors will be split equally.
        Therefore, the lists become [dimension_tensor // num_agents] * num_agents
        """
        n_agents = len(self._active_policies)
        index_names = [
            "index_len_recurrent_hidden_states",
            "index_len_prev_actions",
        ]
        split_index_dict = {}
        for name_index in index_names:
            if name_index not in kwargs:
                if name_index == "index_len_recurrent_hidden_states":
                    all_dim = rnn_hidden_states.shape[-1]
                else:
                    all_dim = prev_actions.shape[-1]
                split_indices = int(all_dim / n_agents)
                split_indices = [split_indices] * n_agents
            else:
                split_indices = kwargs[name_index]
            split_index_dict[name_index] = split_indices
        return split_index_dict

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, **kwargs):
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = torch.split(
            rnn_hidden_states,
            split_index_dict["index_len_recurrent_hidden_states"],
            dim=-1,
        )
        agent_prev_actions = torch.split(
            prev_actions, split_index_dict["index_len_prev_actions"], dim=-1
        )
        agent_masks = torch.split(masks, [1, 1], dim=-1)
        all_value = []
        for agent_i, policy in enumerate(self._active_policies):
            agent_obs = self._update_obs_with_agent_prefix_fn(observations, agent_i)
            all_value.append(
                policy.get_value(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                )
            )
        return torch.stack(all_value, -1)

    def get_extra(
        self, action_data: MultiAgentPolicyActionData, infos, dones
    ) -> List[Dict[str, float]]:
        all_extra = []
        for policy in self._active_policies:
            all_extra.append(policy.get_extra(action_data, infos, dones))
        # The action_data is shared across all policies, so no need to reutrn multiple times
        inputs = all_extra[0]
        ret: List[Dict] = []
        for env_d in inputs:
            ret.append(env_d)

        return ret

    @property
    def policy_action_space(self):
        # TODO: Hack for discrete HL action spaces.
        all_discrete = np.all(
            [
                isinstance(policy.policy_action_space, spaces.MultiDiscrete)
                for policy in self._active_policies
            ]
        )
        if all_discrete:
            return spaces.MultiDiscrete(
                tuple(
                    [policy.policy_action_space.n for policy in self._active_policies]
                )
            )
        else:
            return spaces.Dict(
                {
                    policy_i: policy.policy_action_space
                    for policy_i, policy in enumerate(self._active_policies)
                }
            )

    @property
    def policy_action_space_shape_lens(self):
        lens = []
        for policy in self._active_policies:
            if isinstance(policy.policy_action_space, spaces.Discrete):
                lens.append(1)
            elif isinstance(policy.policy_action_space, spaces.Box):
                lens.append(policy.policy_action_space.shape[0])
            else:
                raise ValueError(
                    f"Action distribution {policy.policy_action_space}" "not supported."
                )
        return lens

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        return cls(update_obs_with_agent_prefix_fn)


class MultiLLMStorage(MultiStorage):
    def __init__(self, update_obs_with_agent_prefix_fn, **kwargs):
        super().__init__(update_obs_with_agent_prefix_fn, **kwargs)

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        return cls(update_obs_with_agent_prefix_fn, **kwargs)


class MultiLLMUpdater(MultiUpdater):
    def __init__(self):
        self._active_updaters = []

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiLLMUpdater()
