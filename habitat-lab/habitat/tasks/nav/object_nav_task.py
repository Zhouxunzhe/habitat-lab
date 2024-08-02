# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
from typing import TYPE_CHECKING, Any, List, Optional
import json
import copy
from cycler import K
from habitat.config.default import get_agent_config
from habitat.core.dataset import Dataset, Episode
from omegaconf import DictConfig, read_write
from typing import Any, Dict, List, Tuple, Union
from collections import OrderedDict

import attr
import numpy as np
from gym import spaces

from habitat.tasks.rearrange.utils import add_perf_timing_func
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator, SensorSuite
from habitat.core.utils import not_none_validator
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimV1
from habitat.tasks.rearrange.utils import (
    CacheHelper,
    CollisionDetails,
    UsesArticulatedAgentInterface,
    rearrange_collision,
    rearrange_logger,
)
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)
from habitat_mas.tasks.habitat_mas_sensors import get_text_context

try:
    from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float]


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_name_id: Optional[int] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None


@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            goal_spec that specifies which id use for goal specification,
            goal_spec_max_val the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.goal_spec_max_val - 1
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None
        category_name = episode.object_category
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.goal_spec == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong goal_spec specified for ObjectGoalSensor."
            )


@registry.register_task(name="ObjectNav-v1")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """

@registry.register_task(name="ObjectNav-v2")
class MultiAgentObjectNavTask(NavigationTask):
    r"""An Object Navigation Task class for a multi-agent navigation task specific methods.
    """

    _cur_episode_step: int
    _articulated_agent_pos_start: Dict[str, Tuple[np.ndarray, float]]


    def _duplicate_sensor_suite(self, sensor_suite: SensorSuite) -> None:
        """
        Modifies the sensor suite in place to duplicate articulated agent specific sensors
        between the two articulated agents.
        """

        task_new_sensors: Dict[str, Sensor] = {}
        task_obs_spaces = OrderedDict()
        for agent_idx, agent_id in enumerate(self._sim.agents_mgr.agent_names):
            for sensor_name, sensor in sensor_suite.sensors.items():
                if isinstance(sensor, UsesArticulatedAgentInterface):
                    new_sensor = copy.copy(sensor)
                    new_sensor.agent_id = agent_idx
                    full_name = f"{agent_id}_{sensor_name}"
                    task_new_sensors[full_name] = new_sensor
                    task_obs_spaces[full_name] = new_sensor.observation_space
                else:
                    task_new_sensors[sensor_name] = sensor
                    task_obs_spaces[sensor_name] = sensor.observation_space

        sensor_suite.sensors = task_new_sensors
        sensor_suite.observation_spaces = spaces.Dict(spaces=task_obs_spaces)

    def __init__(self, *args, sim, dataset=None, should_place_articulated_agent=True, **kwargs) -> None:
        super().__init__(sim=sim, dataset=dataset, **kwargs)
        self.is_gripper_closed = False
        self._sim: HabitatSimV1 = sim
        self._ignore_collisions: List[Any] = []
        self._desired_resting = np.array(self._config.desired_resting_position)
        self._sim_reset = True
        self._targ_idx: int = 0
        self._episode_id: str = ""
        self._cur_episode_step = 0
        self._should_place_articulated_agent = should_place_articulated_agent
        self._seed = self._sim.habitat_config.seed
        self._min_distance_start_agents = (
            self._config.min_distance_start_agents
        )
        # TODO: this patch supports hab2 benchmark fixed states, but should be refactored w/ state caching for multi-agent
        if (
            hasattr(self._sim.habitat_config.agents, "main_agent")
            and self._sim.habitat_config.agents[
                "main_agent"
            ].is_set_start_state
        ):
            self._should_place_articulated_agent = False

        # Get config options
        self._force_regenerate = True
        self._should_save_to_cache = self._config.should_save_to_cache
        self._obj_succ_thresh = self._config.obj_succ_thresh
        self._enable_safe_drop = self._config.enable_safe_drop
        self._constraint_violation_ends_episode = (
            self._config.constraint_violation_ends_episode
        )
        self._constraint_violation_drops_object = (
            self._config.constraint_violation_drops_object
        )
        self._count_obj_collisions = self._config.count_obj_collisions

        # data_path = dataset.config.data_path.format(split=dataset.config.split)
        # data_path = "dataset.data_path"
        data_path = dataset.content_scenes_path

        # TODO(YCC): robot config path
        # robot_config_path = dataset.config.robot_config.format(mode=dataset.config.mode)
        # with open(robot_config_path, "r") as robot_config_file:
        #     robot_config = json.load(robot_config_file)
        # self._robot_config = robot_config

        fname = data_path.split("/")[-1].split(".")[0]
        cache_path = osp.join(
            osp.dirname(data_path),
            f"{fname}_{self._config.type}_robot_start.pickle",
        )

        if self._config.should_save_to_cache or osp.exists(cache_path):
            self._articulated_agent_init_cache = CacheHelper(
                cache_path,
                def_val={},
                verbose=False,
            )
            self._articulated_agent_pos_start = (
                self._articulated_agent_init_cache.load()
            )
        else:
            self._articulated_agent_pos_start = None

        if len(self._sim.agents_mgr) > 1:
            # Duplicate sensors that handle articulated agents. One for each articulated agent.
            self._duplicate_sensor_suite(self.sensor_suite)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        return config
    
    def set_sim_reset(self, sim_reset):
        self._sim_reset = sim_reset

    def _get_ep_init_ident(self, agent_idx):
        return f"{self._episode_id}_{agent_idx}"


    def _get_cached_articulated_agent_start(self, agent_idx: int = 0):
        start_ident = self._get_ep_init_ident(agent_idx)
        if (
            self._articulated_agent_pos_start is None
            or start_ident not in self._articulated_agent_pos_start
            or self._force_regenerate
        ):
            return None
        else:
            return self._articulated_agent_pos_start[start_ident]
        
    def _cache_articulated_agent_start(self, cache_data, agent_idx: int = 0):
        if (
            self._articulated_agent_pos_start is not None
            and self._should_save_to_cache
        ):
            start_ident = self._get_ep_init_ident(agent_idx)
            self._articulated_agent_pos_start[start_ident] = cache_data
            self._articulated_agent_init_cache.save(
                self._articulated_agent_pos_start
            )
        
    def get_task_text_context(self) -> dict:
        return {}
        # current_episode_idx = self._sim.ep_info.episode_id
        # robot_config_path = "data/robots/json/perception.json"
        # with open(robot_config_path, "r") as f:
        #     self._robot_config = json.load(f)
        # robot_config = self._robot_config[current_episode_idx]["agents"]
        # return get_text_context(self._sim, robot_config)

    @add_perf_timing_func()
    def _get_observations(self, episode):
        # Fetch the simulator observations, all visual sensors.
        obs = self._sim.get_sensor_observations()

        if not self._sim.sim_config.enable_batch_renderer:
            # Post-process visual sensor observations
            obs = self._sim._sensor_suite.get_observations(obs)
        else:
            # Keyframes are added so that the simulator state can be reconstituted when batch rendering.
            # The post-processing step above is done after batch rendering.
            self._sim.add_keyframe_to_observations(obs)

        # Task sensors (all non-visual sensors)
        obs.update(
            self.sensor_suite.get_observations(
                observations=obs, episode=episode, task=self, should_time=True
            )
        )
        return obs

    def _is_violating_safe_drop(self, action_args):
        idxs, goal_pos = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        min_dist = np.min(
            np.linalg.norm(target_pos - goal_pos, ord=2, axis=-1)
        )
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] < 0
            and min_dist < self._obj_succ_thresh
        )

    def _set_articulated_agent_start(self, agent_idx: int) -> None:
        articulated_agent_start = self._get_cached_articulated_agent_start(
            agent_idx
        )
        if articulated_agent_start is None:
            filter_agent_position = None
            if self._min_distance_start_agents > 0.0:
                # Force the agents to start a minimum distance apart.
                prev_pose_agents = [
                    np.array(
                        self._sim.get_agent_data(
                            agent_indx_prev
                        ).articulated_agent.base_pos
                    )
                    for agent_indx_prev in range(agent_idx)
                ]

                def _filter_agent_position(start_pos, start_rot):
                    start_pos_2d = start_pos[[0, 2]]
                    prev_pos_2d = [
                        prev_pose_agent[[0, 2]]
                        for prev_pose_agent in prev_pose_agents
                    ]
                    distances = np.array(
                        [
                            np.linalg.norm(start_pos_2d - prev_pos_2d_i)
                            for prev_pos_2d_i in prev_pos_2d
                        ]
                    )
                    return np.all(distances > self._min_distance_start_agents)

                filter_agent_position = _filter_agent_position

            (
                articulated_agent_pos,
                articulated_agent_rot,
            ) = self._sim.set_articulated_agent_base_to_random_point(
                    agent_idx=agent_idx, filter_func=filter_agent_position
                )            
            self._cache_articulated_agent_start(
                (articulated_agent_pos, articulated_agent_rot), agent_idx
            )
        else:
            (
                articulated_agent_pos,
                articulated_agent_rot,
            ) = articulated_agent_start
        articulated_agent = self._sim.get_agent_data(
            agent_idx
        ).articulated_agent
        articulated_agent.base_pos = articulated_agent_pos
        articulated_agent.base_rot = articulated_agent_rot


    @add_perf_timing_func()
    def reset(self, episode: Episode):
        self._episode_id = episode.episode_id
        self._ignore_collisions = []

        if self._sim_reset:
            self._sim.reset()
            for action_instance in self.actions.values():
                action_instance.reset(episode=episode, task=self)
            self._is_episode_active = True

            if self._should_place_articulated_agent:
                for agent_idx in range(self._sim.num_articulated_agents):
                    self._set_articulated_agent_start(agent_idx)

        self.prev_measures = self.measurements.get_metrics()
        self._targ_idx = 0
        self.coll_accum = CollisionDetails()
        self.prev_coll_accum = CollisionDetails()
        self.should_end = False
        self._done = False
        self._cur_episode_step = 0
        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)

    def get_coll_forces(self, articulated_agent_id):
        grasp_mgr = self._sim.get_agent_data(articulated_agent_id).grasp_mgr
        articulated_agent = self._sim.get_agent_data(
            articulated_agent_id
        ).articulated_agent
        snapped_obj = grasp_mgr.snap_idx
        articulated_agent_id = articulated_agent.sim_obj.object_id
        contact_points = self._sim.get_physics_contact_points()

        def get_max_force(contact_points, check_id):
            match_contacts = [
                x
                for x in contact_points
                if (check_id in [x.object_id_a, x.object_id_b])
                and (x.object_id_a != x.object_id_b)
            ]

            max_force = 0
            if len(match_contacts) > 0:
                max_force = max([abs(x.normal_force) for x in match_contacts])

            return max_force

        forces = [
            abs(x.normal_force)
            for x in contact_points
            if (
                x.object_id_a not in self._ignore_collisions
                and x.object_id_b not in self._ignore_collisions
            )
        ]
        max_force = max(forces) if len(forces) > 0 else 0

        max_obj_force = get_max_force(contact_points, snapped_obj)
        max_articulated_agent_force = get_max_force(
            contact_points, articulated_agent_id
        )
        return max_articulated_agent_force, max_obj_force, max_force
    
    def get_cur_collision_info(self, agent_idx) -> CollisionDetails:
        _, coll_details = rearrange_collision(
            self._sim, self._count_obj_collisions, agent_idx=agent_idx
        )
        return coll_details
    

    def step(self, action: Dict[str, Any], episode: Episode):
        action_args = action["action_args"]
        if self._enable_safe_drop and self._is_violating_safe_drop(
            action_args
        ):
            action_args["grip_action"] = None
        obs = super().step(action=action, episode=episode)

        self.prev_coll_accum = copy.copy(self.coll_accum)
        self._cur_episode_step += 1
        for grasp_mgr in self._sim.agents_mgr.grasp_iter:
            if (
                grasp_mgr.is_violating_hold_constraint()
                and self._constraint_violation_drops_object
            ):
                grasp_mgr.desnap(True)

        return obs