import os
from typing import List, Tuple, Dict
import numpy as np
from urchin import URDF, Link, Joint

from habitat_mas.agents.capabilities.parse_urdf import parse_urdf
from habitat_mas.agents.robots.defaults import (
    robot_base_offset_map,
    fetch_camera_params,
    spot_camera_params,
)

def get_camera_tf_to_base(urdf: URDF, camera_link: Link) -> np.ndarray:
    """
    Get the height of the camera by forward kinematics.
    
    Args:
        urdf: the URDF object.
        camera_link: the link that the camera is attached to.
    """
    link_tf = urdf.link_fk(
        link=camera_link
    )
    return link_tf

def get_cameras_height_and_type(urdf_path: str, camera_links:List[str], robot_name:str, skip_link=["third"]) -> dict:
    """
    Get the height and type (fixed, articulated) of the cameras in the URDF file.
    First decide the type of the camera by checking if all parent links are fixed.
    Then get the height of the camera or workspace range of the camera by forward kinematics.
    
    Args:
        urdf_path: the path to the URDF file.
        camera_links: a list of links that have cameras attached to them.
        robot_name: the name of the robot.
        skip_link: a list of links to skip when calculating the height of the camera. Default is ["third"].
    """

    urdf:URDF = parse_urdf(urdf_path)
    cameras_info = {}
    
    for camera_name in camera_links:
        if camera_name in skip_link:
            continue
        camera_link = urdf.link_map[camera_name]
        path = urdf._paths_to_base[camera_link]
        camera_type = "fixed"
        for i in range(len(path) - 1):
            child = path[i]
            parent = path[i + 1]
            joint:Joint = urdf._G.get_edge_data(child, parent)["joint"]
            if joint.joint_type == "revolute":
                camera_type = "articulated"
            
        # get the height of the camera
        camera_tf = get_camera_tf_to_base(urdf, camera_link)
        height_to_base_link = camera_tf[:3, 3][2]
        # robot_base_offset is translation from base_link to feet, mostly negative
        height_to_feet = height_to_base_link - robot_base_offset_map[robot_name][1]
        
        
        # store the height and type of the camera
        camera_info = {"height": height_to_feet, "type": camera_type}
        cameras_info[camera_link.name] = camera_info    
    
    return cameras_info

if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    urdf_path = os.path.join(cur_dir, "../../data/robot_urdf/hab_spot_arm_with_cameras.urdf")
    # camera_params = spot_camera_params["head_only"]
    camera_params = spot_camera_params["default"]
    camera_links = list(camera_params.keys())
    robot_name = "SpotRobot"
    cameras_info = get_cameras_height_and_type(urdf_path, camera_links, robot_name)
    print(cameras_info)
    # output: {'head_stereo_right': {'height': 0.48, 'type': 'fixed'}, 'head_stereo_left': {'height': 0.48, 'type': 'fixed'}}