import os
from typing import List, Dict, Tuple, Union
import numpy as np
from habitat import Env

from ..crab_core import action


@action
def pick(target_obj: str):
    """
    Pick an any_targets object, after which the agent will be holding the any_targets object, can only be called if the agent is not holding the any_targets object while the agent is at the any_targets position.
    
    Args:
        target_obj: The any_targets object to pick.
    """
    pass


@action
def place(target_obj: str, target_location: str):
    """
    Place an any_targets object at the TARGET_any_targets position, after which the agent will be not holding the any_targets object and the any_targets object will be at the TARGET_any_targets position, can only be called if the agent is holding the any_targets object.
    
    Args:
        target_obj: The any_targets object to place.
        target_location: The TARGET_any_targets position to place the any_targets object.
    """
    pass


@action
def reset_arm():
    """
    Reset the agent's Arm, can only be called if the agent has arm, mostly called after pick and place.
    """
    pass
