import os
from typing import List, Dict, Tuple, Union
import numpy as np
from habitat import Env

from habitat.tasks.rearrange.actions.oracle_nav_action import (
    OracleNavCoordinateAction,
    OracleNavAction
)

from ..crab_core import action

# TODO: to connect with the habitat-lab/ habitat-mas agents. 

@action
def get_agents() -> List[str]:
    """
    Get the list of agents in the environment.
    """
    pass

@action
def send_request(request: str, target_agent: str) -> str:
    """
    Send a text request to the fellow agents.
    
    Args:
        request: The text request to send.
        target_agent: The agent to send the request to.
    
    """
    pass

# TODO(YCC): add description for this action
@action
def nav_to_obj(target_obj: str):
    """
    Navigate to a TARGET_any_target object, after which the agent will be at the TARGET_any_target position.
    
    Args:
        target_obj: The TARGET_any_targets object to navigate to.
    """
    pass

@action
def nav_to_goal(goal: str):
    """
    Navigate to an any_targets object, after which the agent will be at the any_targets position, can only be called if the agent is not holding the any_targets object or hasn't pick the any_targets object.
    
    Args:
        goal: The any_targets object to navigate to.
    """
    pass
