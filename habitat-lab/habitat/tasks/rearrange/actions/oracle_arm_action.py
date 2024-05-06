# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union
import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import (
    ArticulatedAgentAction,
    ArmEEAction
)
from habitat.tasks.rearrange.utils import (
    coll_name_matches,
    coll_link_name_matches,
    get_match_link
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_sim.physics import MotionType



@registry.register_task_action
class OraclePickAction(ArmEEAction, ArticulatedAgentAction):
    """
    Pick/drop action for the articulated_agent given the object/receptacle index.
    Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm.
    """

    def __init__(self, *args, sim: RearrangeSim, task, **kwargs):
        ArmEEAction.__init__(self, *args, sim=sim, **kwargs)
        self._sim = sim
        self._task = task
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self._prev_ep_id = None
        # Agent not initialized 
        # self._init_cord = self.cur_articulated_agent.ee_transform().translation
        self._init_cord = np.array([0.0, 0.0, 0.0])
        self._hand_pose_iter = 0

    def reset(self, *args, **kwargs):
        super().reset()
        self.set_desired_ee_pos(self._init_cord)
        

    @property
    def action_space(self):
        return spaces.Box(
                    shape=(2,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
    # NOTE: get_rigid_object_manager has different object idx than that of pddl planner
    # def _get_coord_for_idx(self, object_target_idx):
    #     obj_pos = (
    #         self._sim.get_rigid_object_manager()
    #         .get_object_by_id(object_target_idx)
    #         .translation
    #     )
    #     return obj_pos
    
    def _get_coord_for_pddl_idx(self, object_target_idx):
        pick_obj_entity = self._entities[int(object_target_idx)]
        obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
            pick_obj_entity
        )
        return obj_pos
    
    def get_scene_index_obj(self, object_target_idx):
        pick_obj_entity = self._entities[object_target_idx]
        entity_name = pick_obj_entity.name
        obj_id = self._task.pddl_problem.sim_info.obj_ids[entity_name]
        return self._sim.scene_obj_ids[obj_id]

    def _suction_grasp(self):
        """
        Grasp object using suction grasp, snap to object if contact is detected.
        """
        attempt_snap_entity: Optional[Union[str, int]] = None
        match_coll = None
        contacts = self._sim.get_physics_contact_points()

        # TODO: the two arguments below should be part of args
        ee_index = 0
        index_grasp_manager = 0

        robot_id = self._sim.articulated_agent.sim_obj.object_id
        all_gripper_links = list(
            self._sim.articulated_agent.params.gripper_joints
        )
        robot_contacts = [
            c
            for c in contacts
            if coll_name_matches(c, robot_id)
            and any(coll_link_name_matches(c, l) for l in all_gripper_links)
        ]

        if len(robot_contacts) == 0:
            return

        # Contacted any objects?
        for scene_obj_id in self._sim.scene_obj_ids:
            for c in robot_contacts:
                if coll_name_matches(c, scene_obj_id):
                    match_coll = c
                    break
            if match_coll is not None:
                attempt_snap_entity = scene_obj_id
                break

        if attempt_snap_entity is not None:
            rom = self._sim.get_rigid_object_manager()
            ro = rom.get_object_by_id(attempt_snap_entity)

            ee_T = self.cur_articulated_agent.ee_transform()
            obj_in_ee_T = ee_T.inverted() @ ro.transformation

            # here we need the link T, not the EE T for the constraint frame
            ee_link_T = self.cur_articulated_agent.sim_obj.get_link_scene_node(
                self.cur_articulated_agent.params.ee_links[ee_index]
            ).absolute_transformation()

            self._sim.grasp_mgr.snap_to_obj(
                int(attempt_snap_entity),
                force=False,
                # rel_pos is the relative position of the object COM in link space
                rel_pos=ee_link_T.inverted().transform_point(ro.translation),
                keep_T=obj_in_ee_T,
                should_open_gripper=False,
            )
            return

    def set_desired_ee_pos(self, ee_pos: np.ndarray) -> None:
        self.ee_target += np.array(ee_pos)

        self.apply_ee_constraints()

        joint_pos = np.array(self.cur_articulated_agent.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        self._ik_helper.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = self._ik_helper.calc_ik(self.ee_target)
        des_joint_pos = list(des_joint_pos)
        if self.cur_articulated_agent.sim_obj.motion_type == MotionType.DYNAMIC:
            self.cur_articulated_agent.arm_motor_pos = des_joint_pos
        if self.cur_articulated_agent.sim_obj.motion_type == MotionType.KINEMATIC:
            self.cur_articulated_agent.arm_joint_pos = des_joint_pos
            self.cur_articulated_agent.fix_joint_values = des_joint_pos

    def step(self, pick_action, **kwargs):
        
        object_pick_pddl_idx = pick_action[0]
        should_pick = pick_action[1]

        if object_pick_pddl_idx <= 0 or object_pick_pddl_idx > len(self._entities):
            return

        object_coord = self._get_coord_for_pddl_idx(object_pick_pddl_idx)
        cur_ee_pos = self.cur_articulated_agent.ee_transform().translation
        translation = object_coord - cur_ee_pos  
        
        # translation from object to end effector in base frame
        translation_base = self.cur_articulated_agent.base_transformation.inverted().transform_vector(translation)


        should_rest = False
        # if self.hand_state == HandState.APPROACHING:  # Approaching
        # Only move the hand to object if has to drop or object is not grabbed
        if should_pick == 0 or self.cur_grasp_mgr.snap_idx is None:
            translation_base = np.clip(translation_base, -1, 1)
            translation_base *= self._ee_ctrl_lim
            self.set_desired_ee_pos(translation_base)

            # DEBUG VISUALIZATION
            if self._render_ee_target:
                global_pos = self.cur_articulated_agent.base_transformation.transform_point(
                    self.ee_target
                )
                self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                    global_pos, self._sim.viz_ids["ee_target"]
                )
                
            # TODO: should we add retracting action here? 
        
            # Grasp & Ungrasp when we are close to the target
            # if np.linalg.norm(translation_base) < self._config.grasp_thresh_dist:
            #     self.cur_grasp_mgr.snap_to_obj(
            #         self._sim.scene_obj_ids[object_pick_idx],
            #     )
            #     return
            

    

        