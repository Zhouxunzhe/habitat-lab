from typing import List

from .crab_core import Action
from .models import OpenAIModel


class CrabAgent:
    def __init__(
        self,
        name: str,
        description: str,
        actions: List[Action],
    ):
        self.name = name
        self.description = description
        self.llm_model = OpenAIModel(description, action_space=actions)

    def chat(self, observation: str) -> dict:
        action_name, parameters = self.llm_model.chat(str(observation))
        return {"name": action_name, "arguments": parameters}
