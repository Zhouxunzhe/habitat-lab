from typing import List, Tuple, Dict, Union, Callable, Optional
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import random


# from habitat_mas.agents.llm_agent_base import LLMAgentBase


class DummyAgent:
    def __init__(self, agent_name, action_list):
        # TODO check if agent_index is needed in llm agent
        self.agent_index = agent_name

    def chat(self, message: str) -> Optional[dict]:
        """
        Mimic a chatbot always sending the oraclenav action.
        """

        index = random.choice(["0", "1"])
        # return random.choice([
        #     # {
        #     #     "name": "nav_to_obj",
        #     #     "arguments": {
        #     #         "target_obj": "any_targets|0",
        #     #         "robot_"+index: "robot_"+index,
        #     #     }
        #     # },
        #     {
        #         "name": "nav_to_goal",
        #         "arguments": {
        #             "target_obj": "any_targets|"+index,
        #             "robot_"+index: "robot_"+index,
        #         }
        #     },
        #     {
        #         "name": "pick",
        #         "arguments": {
        #             "any_targets|"+index: "any_targets|"+index,
        #             "robot_"+index: "robot_"+index,
        #         }
        #     },
        #     # {
        #     #     "name": "place",
        #     #     "arguments": {
        #     #         "any_targets|0": "any_targets|0",
        #     #         "TARGET_any_targets|0": "TARGET_any_targets|0",
        #     #         robot: robot,
        #     #     }
        #     # },
        #     ]
        # )
        return random.choice([
            # {
            #     "name": "nav_to_obj",
            #     "arguments": {
            #         "target_obj": "any_targets|0",
            #         "robot_"+index: "robot_"+index,
            #     }
            # },
            {
                "name": "nav_to_obj",
                "arguments": {
                    "target_obj": "any_targets|0",
                    "robot_1": "robot_1",
                }
            },
            # {
            #     "name": "nav_to_goal",
            #     "arguments": {
            #         "target_obj": "any_targets|1",
            #         "robot_1": "robot_1",
            #     }
            # },
            # {
            #     "name": "pick",
            #     "arguments": {
            #         "any_targets|1": "any_targets|1",
            #         "robot_1": "robot_1",
            #     }
            # },
            # {
            #     "name": "place",
            #     "arguments": {
            #         "any_targets|0": "any_targets|0",
            #         "TARGET_any_targets|0": "TARGET_any_targets|0",
            #         robot: robot,
            #     }
            # },
            ]
        )
