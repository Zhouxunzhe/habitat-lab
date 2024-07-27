from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat_mas.perception.nav_mesh import NavMesh
from omegaconf import DictConfig
import numpy as np
import os
import time
import attr
import magnum as mn
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.articulated_agent_manager import (
    ArticulatedAgentData,
    ArticulatedAgentManager,
)
from habitat.datasets.rearrange.mobility_dataset import Rearrange3DEpisode, RearrangeDatasetV1, RearrangeObjectSpec, RearrangeSpec
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat.config import read_write
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.agent import ActionSpec, ActuationSpec
from habitat_sim.agent.agent import Agent, AgentConfiguration, AgentState
from habitat_sim.nav import NavMeshSettings
from habitat_sim.logging import logger
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum, quat_from_angle_axis
from habitat_sim.physics import CollisionGroups, JointMotorSettings, MotionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)


if TYPE_CHECKING:
    from omegaconf import DictConfig


def raycast(sim, sensor_name, crosshair_pos=[128, 128], max_distance=2.0):
    r"""Cast a ray in the direction of crosshair and check if it collides
        with another object within a certain distance threshold
        :param sim: Simulator object
        :param sensor_name: name of the visual sensor to be used for raycasting
        :param crosshair_pos: 2D coordiante in the viewport towards which the
            ray will be cast
        :param max_distance: distance threshold beyond which objects won't
            be considered
    """
    visual_sensor = sim._sensors[sensor_name]
    scene_graph = sim.get_active_scene_graph()
    scene_graph.set_default_render_camera_parameters(
        visual_sensor._sensor_object
    )
    render_camera = scene_graph.get_default_render_camera()
    center_ray = render_camera.unproject(mn.Vector2i(crosshair_pos))

    raycast_results = sim.cast_ray(center_ray, max_distance=max_distance)

    closest_object = -1
    closest_dist = 1000.0
    if raycast_results.has_hits():
        for hit in raycast_results.hits:
            if hit.ray_distance < closest_dist:
                closest_dist = hit.ray_distance
                closest_object = hit.object_id

    return closest_object


@registry.register_simulator(name="RearrangeSim-v1")
class Rearrange3DSim(HabitatSim):
    def __init__(self, config: "DictConfig"):
        self.did_reset = False
        if len(config.agents) > 1:
            with read_write(config):
                for agent_name, agent_cfg in config.agents.items():
                    # using list to create a copy of the sim_sensors keys since we will be
                    # editing the sim_sensors config
                    sensor_keys = list(agent_cfg.sim_sensors.keys())
                    for sensor_key in sensor_keys:
                        sensor_config = agent_cfg.sim_sensors.pop(sensor_key)
                        sensor_config.uuid = (
                            f"{agent_name}_{sensor_config.uuid}"
                        )
                        agent_cfg.sim_sensors[
                            f"{agent_name}_{sensor_key}"
                        ] = sensor_config
        super().__init__(config=config)
        self.ep_info: Optional[Rearrange3DEpisode] = None
        self.grip_offset = np.eye(4)
        self.prev_scene_id: Optional[str] = None
        self.prev_loaded_navmesh = None
        self.agents_mgr = ArticulatedAgentManager(self.habitat_config, self)
        self._kinematic_mode = self.habitat_config.kinematic_mode
        self._additional_object_paths = (
            self.habitat_config.additional_object_paths
        )
        self._load_objs = self.habitat_config.load_objs

        agent_config = None
        if hasattr(self.habitat_config.agents, "agent_0"):
            agent_config = self.habitat_config.agents.agent_0
        elif hasattr(self.habitat_config.agents, "main_agent"):
            agent_config = self.habitat_config.agents.main_agent
        else:
            raise ValueError(f"Cannot find agent parameters.")
        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.radius
        self.navmesh_settings.agent_height = agent_config.height



    def reconfigure(self, config: DictConfig, ep_info: Rearrange3DEpisode):
        self.ep_info = ep_info
        new_scene = self.prev_scene_id != ep_info.scene_id
        if new_scene:
            self._prev_obj_names = None
        ep_info.objects = sorted(ep_info.objects, key=lambda x: x.object_id)
        obj_names = [x["object_handle"].split("/")[-1] for x in ep_info.objects]
        should_add_objects = self._prev_obj_names != obj_names
        self._prev_obj_names = obj_names

        self.agents_mgr.pre_obj_clear()
        self._clear_objects(should_add_objects, new_scene)

        is_hard_reset = new_scene or should_add_objects

        if is_hard_reset:
            with read_write(config):
                config["scene"] = ep_info.scene_id
            super().reconfigure(config, should_close_on_new_scene=False)

        if new_scene:
            self.agents_mgr.on_new_scene()
        self.prev_scene_id = ep_info.scene_id
        self.agents_mgr.post_obj_load_reconfigure()

        self._initialize_objects()

        rom = self.get_rigid_object_manager()
        self._obj_orig_motion_types = {
            handle: ro.motion_type
            for handle, ro in rom.get_objects_by_handle_substring().items()
        }

        if new_scene:
            self._load_navmesh(ep_info)   
    

    def reset(self):
        sim_obs = super().reset()
        for i in range(len(self.agents)):
            self.reset_agent(i)
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        self.did_reset = True
        self.grip_offset = np.eye(4)
        return self._sensor_suite.get_observations(sim_obs)



    def _load_navmesh(self, ep_info: Rearrange3DEpisode):
        scene_name = ep_info.scene_id.split("/")[-1].split(".")[0]
        base_dir = os.path.join(*ep_info.scene_id.split("/")[:2])

        navmesh_path = os.path.join(base_dir, "navmeshes", scene_name + ".navmesh")

        if os.path.exists(navmesh_path):
            self.pathfinder.load_nav_mesh(navmesh_path)
            logger.info(f"Loaded navmesh from {navmesh_path}")
        else:
            logger.warning(
                f"Requested navmesh to load from {navmesh_path} does not exist. Recomputing from configured values and caching."
            )
            navmesh_settings = NavMeshSettings()
            navmesh_settings.set_defaults()

            agent_config = None
            if hasattr(self.habitat_config.agents, "agent_0"):
                agent_config = self.habitat_config.agents.agent_0
            elif hasattr(self.habitat_config.agents, "main_agent"):
                agent_config = self.habitat_config.agents.main_agent
            else:
                raise ValueError(f"Cannot find agent parameters.")
            navmesh_settings.agent_radius = agent_config.radius
            navmesh_settings.agent_height = agent_config.height
            navmesh_settings.agent_max_climb = agent_config.max_climb
            navmesh_settings.agent_max_slope = agent_config.max_slope
            navmesh_settings.include_static_objects = True
            self.recompute_navmesh(self.pathfinder, navmesh_settings)
            os.makedirs(os.path.dirname(navmesh_path), exist_ok=True)
            self.pathfinder.save_nav_mesh(navmesh_path)

        # NOTE: allowing indoor islands only
        self._largest_indoor_island_idx = get_largest_island_index(
            self.pathfinder, self, allow_outdoor=False
        )

    def _initialize_objects(self):
        objects = self.habitat_config.objects[0]
        obj_attr_mgr = self.get_object_template_manager()

        # first remove all existing objects
        existing_object_ids = self.get_existing_object_ids()

        if len(existing_object_ids) > 0:
            for obj_id in existing_object_ids:
                self.remove_object(obj_id)

        self.sim_object_to_objid_mapping = {}
        self.objid_to_sim_object_mapping = {}

        if objects is not None:
            object_template = objects["object_template"]
            object_pos = objects["position"]
            object_rot = objects["rotation"]

            object_template_id = obj_attr_mgr.load_object_configs(
                object_template
            )[0]
            object_attr = obj_attr_mgr.get_template_by_ID(object_template_id)
            obj_attr_mgr.register_template(object_attr)

            object_id = self.add_object_by_handle(object_attr.handle)
            self.sim_object_to_objid_mapping[object_id] = objects["object_id"]
            self.objid_to_sim_object_mapping[objects["object_id"]] = object_id

            self.set_translation(object_pos, object_id)
            if isinstance(object_rot, list):
                object_rot = quat_from_coeffs(object_rot)

            object_rot = quat_to_magnum(object_rot)
            self.set_rotation(object_rot, object_id)

            self.set_object_motion_type(MotionType.STATIC, object_id)
            if self._kinematic_mode:
                self.set_object_motion_type(MotionType.KINEMATIC, object_id)

        # Recompute the navmesh after placing all the objects.
        self.recompute_navmesh(self.pathfinder, self.navmesh_settings, True)
    
    
    def get_agent(self, agent_id: int) -> Agent:
        return self.agents[agent_id]

    def initialize_agent(
        self, agent_id: int, initial_state: Optional[AgentState] = None
    ) -> Agent:
        agent = self.get_agent(agent_id=agent_id)
        if initial_state is None:
            initial_state = AgentState()
            if self.pathfinder.is_loaded:
                initial_state.position = self.pathfinder.get_random_navigable_point()
                initial_state.rotation = quat_from_angle_axis(
                    self.random.uniform_float(0, 2.0 * np.pi), np.array([0, 1, 0])
                )

        agent.set_state(initial_state, is_initial=True)
        self.__last_state[agent_id] = agent.state
        return agent
    
    def _sync_gripped_object(self, gripped_object_id):
        r"""
        Sync the gripped object with the object associated with the agent.
        """
        if gripped_object_id != -1:
            agent_body_transformation = (
                self._default_agent.scene_node.transformation
            )
            self.set_transformation(
                agent_body_transformation, gripped_object_id
            )
            translation = agent_body_transformation.transform_point(
                np.array([0, 2.0, 0])
            )
            self.set_translation(translation, gripped_object_id)

    @property
    def gripped_object_id(self):
        return self._prev_sim_obs.get("gripped_object_id", -1)
    
    def step(self, action: int):
        dt = 1 / 60.0
        self._num_total_frames += 1
        collided = False
        gripped_object_id = self.gripped_object_id

        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[action]

        if action_spec.name == "grab_or_release_object_under_crosshair":
            # If already holding an agent
            if gripped_object_id != -1:
                agent_body_transformation = (
                    self._default_agent.scene_node.transformation
                )
                T = np.dot(agent_body_transformation, self.grip_offset)

                self.set_transformation(T, gripped_object_id)

                position = self.get_translation(gripped_object_id)

                if self.pathfinder.is_navigable(position):
                    self.set_object_motion_type(
                        MotionType.STATIC, gripped_object_id
                    )
                    gripped_object_id = -1
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )
            # if not holding an object, then try to grab
            else:
                gripped_object_id = raycast(
                    self,
                    action_spec.actuation.visual_sensor_name,
                    crosshair_pos=action_spec.actuation.crosshair_pos,
                    max_distance=action_spec.actuation.amount,
                )

                # found a grabbable object.
                if gripped_object_id != -1:
                    agent_body_transformation = (
                        self._default_agent.scene_node.transformation
                    )

                    self.grip_offset = np.dot(
                        np.array(agent_body_transformation.inverted()),
                        np.array(self.get_transformation(gripped_object_id)),
                    )
                    self.set_object_motion_type(
                        MotionType.KINEMATIC, gripped_object_id
                    )
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )

        else:
            collided = self._default_agent.act(action)
            self._last_state = self._default_agent.get_state()

        # step physics by dt
        super().step_world(dt)

        # Sync the gripped object after the agent moves.
        self._sync_gripped_object(gripped_object_id)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations()
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations