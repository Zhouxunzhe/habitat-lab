from typing import List, Tuple, Dict, Union, Callable
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# from habitat_mas.agents.llm_agent_base import LLMAgentBase


class DummyAgent:
    def __init__(self, **kwargs):
        # TODO check if agent_index is needed in llm agent
        self.agent_index = kwargs.get("agent_index", 0)
        
        
    def chat(self, message: str) -> str:
        """
        Mimic a chatbot always sending the oraclenav action.
        """
        
        return {
            "name": "nav_to_obj",
            "arguments": {
                # always navigate to target object with pddl idx 1
                "target_obj": "any_targets|1"
            }
        }
         

