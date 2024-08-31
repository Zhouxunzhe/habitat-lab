from typing import List, Optional

from .crab_core import Action
from ..utils.models import OpenAIModel


REQUEST_TEMPLATE = '"{source_agent}" agent sent you requests: "{request}".'


class CrabAgent:
    message_pipe: dict[str, list[str]] = {}

    def __init__(
        self,
        name: str,
        description: str,
        actions: List[Action],
    ):
        self.name = name
        self.description = description
        self.llm_model = OpenAIModel(description, action_space=actions)

    def chat(self, observation: str) -> Optional[dict]:
        if self.name in CrabAgent.message_pipe and CrabAgent.message_pipe[self.name]:
            prompt = " ".join(CrabAgent.message_pipe[self.name])
            observation = str(observation) + " " + prompt
            CrabAgent.message_pipe[self.name] = []
            
        action_name, parameters = self.llm_model.chat(str(observation))
        if action_name == "send_request":
            target_agent = parameters["target_agent"]
            if target_agent == self.name: # send request to itself
                return None
            request = parameters["request"]
            if target_agent not in CrabAgent.message_pipe:
                CrabAgent.message_pipe[target_agent] = []
            prompt = REQUEST_TEMPLATE.format(source_agent=self.name, request=request)
            CrabAgent.message_pipe[target_agent].append(prompt)
            return None
        else:
            return {"name": action_name, "arguments": parameters}
