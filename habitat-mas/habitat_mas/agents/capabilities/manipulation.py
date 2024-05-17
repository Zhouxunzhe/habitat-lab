from typing import List, Tuple, Dict
import os
import numpy as np
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import (
    ArticulatedAgentAction,
    ArmEEAction,
)
from habitat.articulated_agents.mobile_manipulator import MobileManipulator
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import get_aabb, IkHelper
from matplotlib import pyplot as plt

def get_arm_workspace(
    sim: RearrangeSim, agent_id: int, num_bins=5, geometry: str = "sphere", visualize=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the workspace of the arm of the agent.

    Args:
        sim: The rearrange simulator.
        agent_id: The agent id.
        joint_delta: The delta value for joint position sampling.
        geometry: The geometry of the workspace.
    """
    ik_helper: IkHelper = sim.agents_mgr[agent_id].ik_helper
    agent = sim.agents_mgr[agent_id].articulated_agent
    # TODO: Do we need ellipsoid culling for better accuracy?
    assert geometry in ["sphere", "box"]

    # Initialize an empty list to store the end effector positions
    end_effector_positions = []
    joint_limits_lower, joint_limits_uppper = ik_helper.get_joint_limits()

    # Generate all joint position combinations, which is an N-dimensional grid, N is the number of joints
    joint_positions = np.meshgrid(
        *[
            np.linspace(
                joint_limits_lower[i],
                joint_limits_uppper[i],
                num_bins
            )
            for i in range(len(joint_limits_lower))
        ]
    )

    # Flatten the joint positions
    joint_positions = np.array(
        [joint_position.flatten() for joint_position in joint_positions]
    ).T

    # Iterate over the joint positions
    for joint_position in joint_positions:

        # Calculate the end effector position
        # end_effector_position = ik_helper.calc_fk(joint_position) # offset from real position
        agent.arm_joint_pos = joint_position
        agent.fix_joint_values = joint_position
        end_effector_position = agent.ee_transform().translation
        
        # Append the end effector position to the list
        end_effector_positions.append(end_effector_position)

    # Convert the end effector positions to a numpy array
    end_effector_positions = np.array(end_effector_positions)

    # Visualize the end effector positions
    if visualize:
        for i, end_effector_position in enumerate(end_effector_positions):
            marker_name = f"end_effector_{i}"
            sim.viz_ids[marker_name] = sim.visualize_position(end_effector_position, r=0.01)

    # if visualize:
        
    #     # get current ee position
    #     cur_base_pos = agent.base_pos
    #     cur_ee_pos = agent.ee_transform().translation
        
    #     # Visualize the sampled end effector positions
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(end_effector_positions[:, 0], end_effector_positions[:, 1], end_effector_positions[:, 2], c='r', marker='o', s=0.1)
        
    #     # Visualize the current base and end effector positions
    #     ax.scatter(cur_base_pos[0], cur_base_pos[1], cur_base_pos[2], c='g', marker='o', s=10)
    #     ax.scatter(cur_ee_pos[0], cur_ee_pos[1], cur_ee_pos[2], c='b', marker='o', s=10)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     plt.show()

    if geometry == "sphere":
        # Calculate the center of the sphere
        center = np.mean(end_effector_positions, axis=0)

        # Calculate the radius of the sphere
        radius = np.max(
            np.linalg.norm(end_effector_positions - center, axis=1)
        )

        return center, radius

    elif geometry == "box":

        # Calculate the minimum and maximum x, y, and z coordinates
        min_coords = np.min(end_effector_positions, axis=0)
        max_coords = np.max(end_effector_positions, axis=0)
        
        return min_coords, max_coords