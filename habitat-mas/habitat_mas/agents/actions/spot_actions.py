import os
from typing import List, Dict, Tuple, Union
from habitat import Env
from habitat.core.simulator import Observations
from multiprocessing.connection import Client
import time

from agents.actions.action_server import ActionServer
from agents.actions.base_actions import BaseActions

class SpotActions(BaseActions):
    """
    This class is used to collect all action APIs for the Spot robot.
    """
    
    def __init__(self, agent_name: str, action_server: ActionServer, env:Env) -> None:
        super().__init__(agent_name, action_server, env)
        
    def navigate_to_position(self, position: Tuple[float, float, float]) -> Observations:
        """
        Navigate the Spot robot to a specific position in the environment.
        """
        action = OracleNavCoordinateAction(position)
        return self.action_server.update(action)
    
    def navigate_to_object(self, object_id: str) -> Observations:
        """
        Navigate the Spot robot to a specific object in the environment.
        """
        action = OracleNavAction(object_id)
        return self.action_server.update(action)
    
    def navigate_to_region(self, region_id: str) -> Observations:
        """
        Navigate the Spot robot to a specific region in the environment.
        """
        action = OracleNavAction(region_id)
        return self.action_server.update(action)
    
    def move_arm_to_position(self, position: Tuple[float, float, float]) -> Observations:
        """
        Move the Spot robot's arm to a specific position.
        """
        action = MoveArmAction(position)
        return self.action_server.update(action)
    
    def move_arm_to_object(self, object_id: str) -> Observations:
        """
        Move the Spot robot's arm to a specific object.
        """
        action = MoveArmAction(object_id)
        return self.action_server.update(action)