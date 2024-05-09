import os
from typing import List, Dict, Tuple, Union
import numpy as np
from habitat import Env
from agents.actions.action_server import ActionServer
from habitat.tasks.rearrange.actions.oracle_nav_action import (
    OracleNavCoordinateAction,
    OracleNavAction
)


class BaseActions:
    """
    This class is used to collect all interface action APIs between LLM agents and the habitat simulator. 
    """
    
    def __init__(self, agent_name: str, action_server: ActionServer, env:Env) -> None:
        self.agent_name = agent_name
        self.action_server = action_server
        self.env = env
        
    def get_agents(self) -> List[str]:
        """
        Get the list of agents in the environment.
        """
        raise NotImplementedError
    
    def send_request(self, request: str, target_agent: str) -> str:
        """
        Send a text request to the fellow agents.
        """
        # TODO: This should not be part of BaseActions, but a part of the LLM agent.
        raise NotImplementedError