from typing import Any, Dict, List, Tuple

import torch

from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData
# TODO: replace dummy_agent with llm_agent
from habitat_mas.agents.dummy_agent import DummyAgent

class LLMHighLevelPolicy(HighLevelPolicy):
    """
    High-level policy that uses an LLM agent to select skills.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_actions = self._setup_actions()
        self._n_actions = len(self._all_actions)
        
        # Initialize the LLM agent
        self._llm_agent = self._init_llm_agent()

    def _init_llm_agent(self):
        # Initialize the LLM agent here based on the config
        # This could load a pre-trained model, set up prompts, etc.
        # Return the initialized agent
        return DummyAgent()

    def _parse_function_call_args(self, llm_args: Dict) -> str:
        """
        Parse the arguments of a function call from the LLM agent to the policy input argument format.
        """
        return llm_args
        
    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor, PolicyActionData]:
        """
        Get the next skill to execute from the LLM agent.
        """
        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue

            # Query the LLM agent with the current observations
            # to get the next action and arguments
            llm_output = self._llm_agent.chat(observations[batch_idx])
            action_name = llm_output['name'] 
            action_args = self._parse_function_call_args(llm_output['arguments'])

            if action_name in self._skill_name_to_idx:
                next_skill[batch_idx] = self._skill_name_to_idx[action_name]
                skill_args_data[batch_idx] = action_args
            else:
                # If the action is not valid, do nothing
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                skill_args_data[batch_idx] = ["1"]

        return (
            next_skill, 
            skill_args_data,
            immediate_end,
            PolicyActionData(),
        )