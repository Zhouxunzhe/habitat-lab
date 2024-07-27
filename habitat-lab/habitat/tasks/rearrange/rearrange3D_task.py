import os
from typing import Any, Dict, List, Type

import attr
import numpy as np
from gym import spaces

from habitat.config.default import Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure, SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.core.utils import not_none_validator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.rearrange3D_sim import Rearrange3DSim, raycast
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationTask,
    PointGoalSensor,
    merge_sim_episode_config,
)
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum

@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""

        gripped_object_id = self._sim._prev_sim_obs["gripped_object_id"]
        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[HabitatSimActions.GRAB_RELEASE]

        # If already holding an agent
        if gripped_object_id != -1:
            agent_body_transformation = (
                self._sim._default_agent.scene_node.transformation
            )
            T = np.dot(agent_body_transformation, self._sim.grip_offset)

            self._sim.set_transformation(T, gripped_object_id)

            position = self._sim.get_translation(gripped_object_id)

            if self._sim.pathfinder.is_navigable(position):
                self._sim.set_object_motion_type(
                    MotionType.STATIC, gripped_object_id
                )
                gripped_object_id = -1
                self._sim.recompute_navmesh(
                    self._sim.pathfinder, self._sim.navmesh_settings, True
                )
        # if not holding an object, then try to grab
        else:
            gripped_object_id = raycast(
                self._sim,
                action_spec.actuation.visual_sensor_name,
                crosshair_pos=action_spec.actuation.crosshair_pos,
                max_distance=action_spec.actuation.amount,
            )

            # found a grabbable object.
            if gripped_object_id != -1:
                agent_body_transformation = (
                    self._sim._default_agent.scene_node.transformation
                )

                self._sim.grip_offset = np.dot(
                    np.array(agent_body_transformation.inverted()),
                    np.array(self._sim.get_transformation(gripped_object_id)),
                )
                self._sim.set_object_motion_type(
                    MotionType.KINEMATIC, gripped_object_id
                )
                self._sim.recompute_navmesh(
                    self._sim.pathfinder, self._sim.navmesh_settings, True
                )

        # step physics by dt
        super().step_world(1 / 60.0)

        # Sync the gripped object after the agent moves.
        self._sim._sync_gripped_object(gripped_object_id)

        # obtain observations
        self._sim._prev_sim_obs.update(self._sim.get_sensor_observations())
        self._sim._prev_sim_obs["gripped_object_id"] = gripped_object_id

        observations = self._sim._sensor_suite.get_observations(
            self._sim._prev_sim_obs
        )
        return observations