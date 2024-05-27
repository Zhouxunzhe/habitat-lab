import os
import numpy as np
import magnum as mn
import networkx as nx
from networkx.readwrite.text import generate_network_text
import habitat_sim
from omegaconf import DictConfig
from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulatorParams,
)
import networkx as nx
from networkx.readwrite.text import generate_network_text
import habitat_sim

robot_link_id2name_map = {
    "FetchRobot": {
        0: "bellows_link",
        1: "estop_link",
        2: "l_wheel_link",
        3: "laser_link",
        4: "r_wheel_link",
        5: "torso_fixed_link",
        6: "torso_lift_link",
        7: "bellows_link2",
        8: "head_pan_link",
        9: "head_tilt_link",
        10: "head_camera_link",
        11: "head_camera_depth_frame",
        12: "head_camera_depth_optical_frame",
        13: "head_camera_rgb_frame",
        14: "head_camera_rgb_optical_frame",
        15: "shoulder_pan_link",
        16: "shoulder_lift_link",
        17: "upperarm_roll_link",
        18: "elbow_flex_link",
        19: "forearm_roll_link",
        20: "wrist_flex_link",
        21: "wrist_roll_link",
        22: "gripper_link",
        23: "l_gripper_finger_link",
        24: "r_gripper_finger_link",
    },
    "SpotRobot": {
        0: "arm0.link_sh0",
        1: "arm0.link_sh1",
        2: "arm0.link_hr0",
        3: "arm0.link_el0",
        4: "arm0.link_el1",
        5: "arm0.link_wr0",
        6: "arm0.link_wr1",
        7: "arm0.link_fngr",
        8: "fl.hip",
        9: "fl.uleg",
        10: "fl.lleg",
        11: "fr.hip",
        12: "fr.uleg",
        13: "fr.lleg",
        14: "hl.hip",
        15: "hl.uleg",
        16: "hl.lleg",
        17: "hr.hip",
        18: "hr.uleg",
        19: "hr.lleg",
    },
    "StretchRobot": {
        0: "caster_link",
        1: "link_aruco_left_base",
        2: "link_aruco_right_base",
        3: "laser",
        4: "link_left_wheel",
        5: "link_mast",
        6: "link_head",
        7: "link_head_pan",
        8: "link_head_tilt",
        9: "camera_bottom_screw_frame",
        10: "camera_link",
        11: "camera_accel_frame",
        12: "camera_accel_optical_frame",
        13: "camera_color_frame",
        14: "camera_color_optical_frame",
        15: "camera_depth_frame",
        16: "camera_depth_optical_frame",
        17: "camera_gyro_frame",
        18: "camera_gyro_optical_frame",
        19: "camera_infra1_frame",
        20: "camera_infra1_optical_frame",
        21: "camera_infra2_frame",
        22: "camera_infra2_optical_frame",
        23: "link_lift",
        24: "link_arm_l4",
        25: "link_arm_l3",
        26: "link_arm_l2",
        27: "link_arm_l1",
        28: "link_arm_l0",
        29: "link_aruco_inner_wrist",
        30: "link_aruco_top_wrist",
        31: "link_wrist_yaw",
        32: "link_wrist_yaw_bottom",
        33: "link_wrist_pitch",
        34: "link_wrist_roll",
        35: "link_straight_gripper",
        36: "link_gripper_finger_left",
        37: "link_gripper_fingertip_left",
        38: "link_gripper_finger_right",
        39: "link_gripper_fingertip_right",
        40: "link_aruco_shoulder",
        41: "respeaker_base",
        42: "link_right_wheel",
    },
}

# The offset of the robot base link to the feet.
robot_base_offset_map = {
    "KinematicHumanoid": np.array([0, -0.9, 0]),
    "FetchRobot": np.array([0, 0, 0]),
    "SpotRobot": np.array([0, -0.48, 0]),
    "StretchRobot": np.array([0, -0.5, 0]),
    "DJIDrone": np.array([0, -1.5, 0]),
}

fetch_camera_params = {
    "default": {
        "articulated_agent_arm": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
            cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
            attached_link_id=22,
            relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
            @ mn.Matrix4.rotation_z(mn.Deg(90)),
        ),
        "head": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.25, 1.2, 0.0),
            cam_look_at_pos=mn.Vector3(0.75, 1.0, 0.0),
            attached_link_id=-1,
        ),
        # NOTE: should exclude the third-person view camera since it's for debug only
        "third": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
            cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
            attached_link_id=-1,
        ),
    },
    "head_only": {
        "head": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.25, 1.2, 0.0),
            cam_look_at_pos=mn.Vector3(0.75, 1.0, 0.0),
            attached_link_id=-1,
        ),
    },
    "arm_only": {
        "articulated_agent_arm": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
            cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
            attached_link_id=22,
            relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
            @ mn.Matrix4.rotation_z(mn.Deg(90)),
        ),
    },
}

spot_camera_params = {
    "default": {
        "articulated_agent_arm_depth": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, 0.018),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "articulated_agent_arm_rgb": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.023, 0.03),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "articulated_agent_arm_panoptic": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, 0.018),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "head_stereo_right": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(
                0.4164822634134684, 0.0, 0.03614789234067159
            ),
            cam_orientation=mn.Vector3(0.0290787, -0.940569, -0.38998877),
            attached_link_id=-1,
        ),
        "head_stereo_left": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(
                0.4164822634134684, 0.0, -0.03740343144695029
            ),
            cam_orientation=mn.Vector3(-3.1125141, -0.940569, 2.751605),
            attached_link_id=-1,
        ),
        "third": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
            cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
            attached_link_id=-1,
        ),
        "articulated_agent_jaw_depth": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, -0.107),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "articulated_agent_jaw_rgb": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, -0.107),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "articulated_agent_jaw_panoptic": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, -0.107),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
    },
    "head_only": {
        "head_stereo_right": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(
                0.4164822634134684, 0.0, 0.03614789234067159
            ),
            cam_orientation=mn.Vector3(0.0290787, -0.940569, -0.38998877),
            attached_link_id=-1,
        ),
        "head_stereo_left": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(
                0.4164822634134684, 0.0, -0.03740343144695029
            ),
            cam_orientation=mn.Vector3(-3.1125141, -0.940569, 2.751605),
            attached_link_id=-1,
        ),
    },
    "arm_only": {
        "articulated_agent_arm_depth": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, 0.018),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "articulated_agent_arm_rgb": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.023, 0.03),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "articulated_agent_arm_panoptic": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, 0.018),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "articulated_agent_jaw_depth": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, -0.107),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "articulated_agent_jaw_rgb": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, -0.107),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "articulated_agent_jaw_panoptic": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.166, 0.0, -0.107),
            cam_orientation=mn.Vector3(0, -1.571, 0.0),
            attached_link_id=6,
            relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
    },
}

stretch_camera_params = {
    "default": {
        "head": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
            cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.1),
            attached_link_id=14,
            relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
            @ mn.Matrix4.rotation_z(mn.Deg(-90)),
        ),
        "third": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
            cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
            attached_link_id=-1,
        ),
    }
}

dji_camera_params = {
    "default": {
        "head": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(0.0, 0.0, 0.0),
            cam_look_at_pos=mn.Vector3(1.0, -1.0, 0.0),
            attached_link_id=-1,
        ),
        "third": ArticulatedAgentCameraParams(
            cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
            cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
            attached_link_id=-1,
        ),
    },
}


def get_robot_link_id2name_map(robot_name):
    """Get the robot link id to name map for the given robot name."""
    if robot_name == "FetchRobot":
        return robot_link_id2name_map["FetchRobot"]
    elif robot_name == "SpotRobot":
        return robot_link_id2name_map["SpotRobot"]
    elif robot_name == "StretchRobot":
        return robot_link_id2name_map["StretchRobot"]
    else:
        raise NotImplementedError(f"Robot {robot_name} not supported")
