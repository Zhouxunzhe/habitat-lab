import math
from tqdm import tqdm
from habitat.datasets.rearrange.samplers import receptacle
import magnum as mn
from habitat.core.dataset import Episode
from habitat.core.simulator import Simulator
from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode, RearrangeDatasetV0
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
    is_accessible,
)
from collections import defaultdict
import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.core.logging import logger
from habitat.utils.common import cull_string_list_by_substrings
import habitat_sim
import json
import os
import os.path as osp
from omegaconf import OmegaConf
from typing import(
    Dict, Generator, List, Optional, Sequence, Tuple, Union, Any
)
from habitat.config import DictConfig
from habitat.datasets.rearrange.samplers.receptacle import (
    OnTopOfReceptacle,
    Receptacle,
    ReceptacleSet,
    ReceptacleTracker,
    find_receptacles,
    get_navigable_receptacles,
)
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
import numpy as np
import random
from habitat_sim.nav import NavMeshSettings
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2

def get_sample_region_ratios(load_dict) -> Dict[str, float]:
    sample_region_ratios: Dict[str, float] = defaultdict(lambda: 1.0)
    sample_region_ratios.update(
        load_dict["params"].get("sample_region_ratio", {})
    )
    return sample_region_ratios


# function in HabitatSim class for finding geodesic distance
def geodesic_distance(
        sim: "HabitatSim",
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode: Optional[Episode] = None,
    ) -> float:
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], (Sequence, np.ndarray)):
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else:
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)]
                )
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)

        sim.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

def is_compatible_episode(
    s: Sequence[float],
    t: Sequence[float],
    sim: "HabitatSim",
    near_dist: float,
    far_dist: float,
    geodesic_to_euclid_ratio: float,
) -> Union[Tuple[bool, float], Tuple[bool, int]]:
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)

    #TODO(YCC): In mobility task, s and t may not be on the same floor, 
    # we need to check height difference
    height_dist = np.abs(s[1] - t[1])
    d_separation = geodesic_distance(sim, s, [t])
    if d_separation == np.inf:
        return False, 0, height_dist
    if not near_dist <= d_separation <= far_dist:
        return False, 0, height_dist
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0, height_dist

    return True, d_separation, height_dist


class MobilityGenerator:
    
    def __enter__(self) -> "MobilityGenerator":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.sim != None:
            self.sim.close(destroy=True)
            del self.sim

    def __init__(
        self,
        cfg: DictConfig,
        debug_visualization: bool = False,
        num_episodes: int = 1,
    ) -> None:

        # load and cache the config
        self.cfg = cfg
        self.num_episodes = num_episodes

        # hold a habitat Simulator object for efficient re-use
        self.sim: habitat_sim.Simulator = None
        # initialize an empty scene and load the SceneDataset
        self.initialize_sim()



# TODO(YCC): sample the scene from config and initialize the Simulator
    def initialize_sim(self, dataset_path: str = "/home/yuchecheng/habitat-lab/data/scene_datasets/mp3d/") -> None:
        backend_cfg = habitat_sim.SimulatorConfiguration()
        # TODO(YCC): randomly sample the scene from config
        sub_scene_paths = [d for d in os.listdir(dataset_path) if osp.isdir(osp.join(dataset_path, d))]
        if not sub_scene_paths or len(sub_scene_paths) == 1:
            raise ValueError("No sub scene paths found in dataset path.")
        else:
            selected_sub_scene = random.choice(sub_scene_paths)
            scene_glb_path = osp.join(dataset_path, selected_sub_scene, f"{selected_sub_scene}.glb")

        backend_cfg.scene_id = scene_glb_path
        backend_cfg.scene_dataset_config_file = cfg.dataset_path
        # backend_cfg.additional_obj_config_paths = cfg.additional_object_paths
        backend_cfg.enable_physics = True
        backend_cfg.load_semantic_mesh = True

        sensor_specs = []
        
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        sensor_specs.append(color_sensor_spec)
        
        sem_cfg = habitat_sim.CameraSensorSpec()
        sem_cfg.uuid = "semantic"
        sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
        sensor_specs.append(sem_cfg)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        if self.sim is None:
            self.sim = habitat_sim.Simulator(sim_cfg)
            object_attr_mgr = self.sim.get_object_template_manager()
            for object_path in cfg.additional_object_paths:
                object_attr_mgr.load_configs(osp.abspath(object_path))
        else:
            if self.sim.config.sim_cfg.scene_id != scene_glb_path:
                self.sim.close(destroy=True)
            if self.sim.config.sim_cfg.scene_id == scene_glb_path:
                # we need to force a reset, so reload the NONE scene
                proxy_backend_cfg = habitat_sim.SimulatorConfiguration()
                proxy_backend_cfg.scene_id = "NONE"
                proxy_backend_cfg.gpu_device_id = cfg.gpu_device_id
                proxy_hab_cfg = habitat_sim.Configuration(
                    proxy_backend_cfg, [agent_cfg]
                )
                self.sim.reconfigure(proxy_hab_cfg)
            self.sim.reconfigure(sim_cfg)

        return scene_glb_path


# TODO(YCC):generate single episode, return episode as NavigationEpisode
    def generate_single_episode(
        self,
        episode_id: int,
        closest_dist_limit: float = 1,
        furthest_dist_limit: float = 30,
        geodesic_to_euclid_min_ratio: float = 1.1,
        number_retries_per_target: int = 10,
        max_placement_tries: int = 10,
    ) -> Optional[RearrangeEpisode]:
        
        ep_scene_handle = self.initialize_sim()
        scene_base_dir = osp.dirname(osp.dirname(ep_scene_handle))
        scene_name = ep_scene_handle.split(".")[0].split("/")[-1]
        navmesh_path = osp.join(
            scene_base_dir, scene_name, scene_name + ".navmesh"
        )

        # Load navmesh
        regenerate_navmesh = self.cfg.regenerate_new_mesh
        if not regenerate_navmesh and not self.sim.pathfinder.load_nav_mesh(
            navmesh_path
        ):
            # if loading fails, regenerate instead
            regenerate_navmesh = True
            logger.error(
                f"Failed to load navmesh '{navmesh_path}', regenerating instead."
            )
        if regenerate_navmesh:
            navmesh_settings = NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_radius = self.cfg.agent_radius
            navmesh_settings.agent_height = self.cfg.agent_height
            navmesh_settings.include_static_objects = True
            navmesh_settings.agent_max_climb = self.cfg.agent_max_climb
            navmesh_settings.agent_max_slope = self.cfg.agent_max_slope
            self.sim.recompute_navmesh(
                self.sim.pathfinder,
                navmesh_settings,
            )

        if len(self.sim.semantic_scene.levels) < 2:
            return None
        
        largest_indoor_island_id = get_largest_island_index(
            self.sim.pathfinder, self.sim, allow_outdoor=False
        )

        # TODO(YCC): randomly sample the target receptacles
        target_receptacles: Dict[str, Any] = {}
        expected_list_keys = ["included_object_substrings", "excluded_object_substrings"]
        sampled_target_receptacles = {}
        for target_recep_set in cfg.receptacle_sets:
            assert "name" in target_recep_set
            for list_key in expected_list_keys:
                assert list_key in target_recep_set
                assert isinstance(target_recep_set[list_key], Sequence)
            target_receptacles[target_recep_set["name"]] = cull_string_list_by_substrings(
                self.sim.get_object_template_manager().get_template_handles(),
                target_recep_set["included_object_substrings"],
                target_recep_set["excluded_object_substrings"],
            )
            sampled_target_receptacle = target_receptacles[target_recep_set['name']][np.random.randint(0, len(target_receptacles[target_recep_set['name']]))]
            sampled_target_receptacles[target_recep_set['name']] = sampled_target_receptacle

        # TODO(YCC): randomly sample the goal receptacles
        goal_receptacles: Dict[str, Any] = {}
        sampled_goal_receptacles = {}
        for goal_recep_set in cfg.receptacle_sets:
            assert "name" in goal_recep_set
            for list_key in expected_list_keys:
                assert list_key in goal_recep_set
                assert isinstance(goal_recep_set[list_key], Sequence)
            goal_receptacles[goal_recep_set["name"]] = cull_string_list_by_substrings(
                self.sim.get_object_template_manager().get_template_handles(),
                goal_recep_set["included_object_substrings"],
                goal_recep_set["excluded_object_substrings"],
            )
            sampled_goal_receptacle = None
            while sampled_goal_receptacle is None or sampled_goal_receptacle in sampled_target_receptacles[goal_recep_set['name']]:
                sampled_goal_receptacle = goal_receptacles[goal_recep_set['name']][np.random.randint(0, len(goal_receptacles[goal_recep_set['name']]))]
            sampled_goal_receptacles[goal_recep_set['name']] = sampled_goal_receptacle


        # TODO(YCC): randomly sample the object set
        obj_sets: Dict[str, List[str]] = {}
        sampled_objects = {}
        expected_list_keys = ["included_substrings", "excluded_substrings"]
        for obj_set in cfg.object_sets:
            assert "name" in obj_set
            for list_key in expected_list_keys:
                assert list_key in obj_set
                assert isinstance(obj_set[list_key], Sequence)
            obj_sets[obj_set["name"]] = cull_string_list_by_substrings(
                self.sim.get_object_template_manager().get_template_handles(),
                obj_set["included_substrings"],
                obj_set["excluded_substrings"],
            )
            sampled_object = obj_sets[obj_set['name']][np.random.randint(0, len(obj_sets[obj_set['name']]))]
            sampled_objects[obj_set['name']] = sampled_object


        rom = self.sim.get_rigid_object_manager()
        # try to place the target receptacles to target position
        num_placement_tries = 0
        new_target_receptacles = {}
        for name, target_receptacle in sampled_target_receptacles.items():
            while num_placement_tries < max_placement_tries:
                num_placement_tries += 1
                target_position = self.sim.pathfinder.get_random_navigable_point().tolist()
                while self.sim.pathfinder.island_radius(target_position) < 2.0:
                    target_position = self.sim.pathfinder.get_random_navigable_point().tolist()
                if new_target_receptacles == {}:
                    assert self.sim.get_object_template_manager().get_library_has_handle(
                        target_receptacle
                    ), f"Receptacle handle {target_receptacle} not found in library."
                    new_target_receptacles[name] = rom.add_object_by_template_handle(
                        target_receptacle
                    )
                new_target_receptacles[name].translation = target_position
                new_target_receptacles[name].rotation = mn.Quaternion.rotation(
                    mn.Rad(random.uniform(0, math.pi * 2.0)), mn.Vector3.y_axis()
                )
                if not is_accessible(
                    sim=self.sim,
                    point=new_target_receptacles[name].translation,
                    height=1,
                    nav_to_min_distance=-1.0,
                    nav_island=largest_indoor_island_id,
                    target_object_id=new_target_receptacles[name].object_id
                ):
                    new_target_receptacles = {}
                    continue
                break
        
        # try to place the goal receptacles to goal position
        num_placement_tries = 0
        new_goal_receptacles = {}
        for name, goal_receptacle in sampled_goal_receptacles.items():
            while num_placement_tries < max_placement_tries:
                num_placement_tries += 1
                for _retry in range(100):
                    goal_position = self.sim.pathfinder.get_random_navigable_point().tolist()
                    while self.sim.pathfinder.island_radius(goal_position) < 2.0:
                        goal_position = self.sim.pathfinder.get_random_navigable_point().tolist()
                    is_compatible, dist, height_dist = is_compatible_episode(
                        goal_position,
                        target_position,
                        self.sim,
                        near_dist=closest_dist_limit,
                        far_dist=furthest_dist_limit,
                        geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
                    )
                    if is_compatible:
                        break
                if new_goal_receptacles == {}:
                    assert self.sim.get_object_template_manager().get_library_has_handle(
                        goal_receptacle
                    ), f"Receptacle handle {goal_receptacle} not found in library."
                    new_goal_receptacles[name] = rom.add_object_by_template_handle(
                        goal_receptacle
                    )
                new_goal_receptacles[name].translation = goal_position
                new_goal_receptacles[name].rotation = mn.Quaternion.rotation(
                    mn.Rad(random.uniform(0, math.pi * 2.0)), mn.Vector3.y_axis()
                )
                if not is_accessible(
                    sim=self.sim,
                    point=new_goal_receptacles[name].translation,
                    height=1,
                    nav_to_min_distance=-1.0,
                    nav_island=largest_indoor_island_id,
                    target_object_id=new_goal_receptacles[name].object_id
                ):
                    new_goal_receptacles = {}
                    continue
        if height_dist < 0.1:
            return None

        # TODO(YCC): try to place the object to target receptacle
        new_object = {}
        object_position = [target_position[0], target_position[1] + 0.2, target_position[2]]

        for name, object_handle in sampled_objects.items():
            assert self.sim.get_object_template_manager().get_library_has_handle(
                object_handle
            ), f"Object handle {object_handle} not found in library."
            new_object[name] = rom.add_object_by_template_handle(
                object_handle
            )
            new_object[name].translation = object_position
            new_object[name].rotation = mn.Quaternion.rotation(
                mn.Rad(random.uniform(0, math.pi * 2.0)), mn.Vector3.y_axis()
            )
            if not is_accessible(
                sim=self.sim,
                point=new_object[name].translation,
                height=1,
                nav_to_min_distance=-1.0,
                nav_island=largest_indoor_island_id,
                target_object_id=new_object[name].object_id
            ):
                return None

        objects = []
        targets = {}
        object_filename, object_name = "", ""
        for name, object_handle in sampled_objects.items():
            object_filename = object_handle.split("/")[-1]
            object_name = new_object[name].handle
            matrix_list = np.array([[new_object[name].transformation[i][j] for j in range(4)] for i in range(4)])
            obj = (object_filename, matrix_list)
            objects.append(obj)

        # TODO(YCC): try to place the object to goal receptacle
        object_target_position = [goal_position[0], goal_position[1] + 0.2, goal_position[2]]
        new_object_target = {}
        for name, object_handle in sampled_objects.items():
            assert self.sim.get_object_template_manager().get_library_has_handle(
                object_handle
            ), f"Object handle {object_handle} not found in library."
            rom.remove_object_by_id(
                new_object[name].object_id
            )
            new_object_target[name] = rom.add_object_by_template_handle(
                object_handle
            )
            new_object_target[name].translation = object_target_position
            new_object_target[name].rotation = mn.Quaternion.rotation(
                mn.Rad(random.uniform(0, math.pi * 2.0)), mn.Vector3.y_axis()
            )
            if not is_accessible(
                sim=self.sim,
                point=new_object_target[name].translation,
                height=1,
                nav_to_min_distance=-1.0,
                nav_island=largest_indoor_island_id,
                target_object_id=new_object_target[name].object_id
            ):
                return None
        

        for name, object_handle in sampled_objects.items():
            object_name = new_object_target[name].handle
            target_matrix_list = np.array([[new_object_target[name].transformation[i][j] for j in range(4)] for i in range(4)])
            targets[object_name] = target_matrix_list

        target_receps = []
        for name, target_recep_handle in sampled_target_receptacles.items():
            target_recep_name = new_target_receptacles[name].handle
            matrix_list = np.array([[new_target_receptacles[name].transformation[i][j] for j in range(4)] for i in range(4)])
            tar_translation = [x for x in new_target_receptacles[name].translation]
            target_recep = (target_recep_name, matrix_list, tar_translation)
            target_receps.append(target_recep)
        goal_receps = []
        for name, goal_recep_handle in sampled_goal_receptacles.items():
            goal_recep_name = new_goal_receptacles[name].handle
            matrix_list = np.array([[new_goal_receptacles[name].transformation[i][j] for j in range(4)] for i in range(4)])
            goal_translation = [x for x in new_goal_receptacles[name].translation]
            goal_recep = (goal_recep_name, matrix_list, goal_translation)
            goal_receps.append(goal_recep)
        

        name_to_receptacle = {}
        for name, object_handle in sampled_objects.items():
            object_name = new_object_target[name].handle
            tr = rom.get_object_by_handle(new_target_receptacles["table"].handle)
            tr_name = tr.user_attributes.get_subconfig_keys()[0]
            tr_handle = new_target_receptacles["table"].handle
            name_to_receptacle[object_name] = f"{tr_handle}|{tr_name}"

        same_floor = abs(target_position[1] - goal_position[1]) < 0.5

        return RearrangeEpisode(
            scene_dataset_config=self.sim.config.sim_cfg.scene_dataset_config_file,
            additional_obj_config_paths=cfg.additional_object_paths,
            episode_id=str(episode_id),
            start_position=[0.0, 0.0, 0.0],
            start_rotation=[0.0, 0.0, 0.0, 1.0],
            scene_id=ep_scene_handle,
            rigid_objs=objects,
            ao_states={},
            target_receptacles=target_receps,
            targets=targets,
            goal_receptacles=goal_receps,
            name_to_receptacle=name_to_receptacle,
            markers={},
            info={
                "object_labels": {object_name: "any_targets|0"},
                "dataset": "mp3d",
                "geodesic_distance": dist, 
                "target_goal_height_difference": height_dist, 
                "same_floor": same_floor
                },
        )

#TODO(YCC):generate HM3D/MP3D episodes
    def generate_mobility_episodes(self, output_dir):


        logger.info(f"\n\nConfig:\n{cfg}\n\n")
        dataset = RearrangeDatasetV0()
        episodes = []
        with tqdm(total=self.num_episodes) as pbar:
            while len(episodes) < self.num_episodes:
                episode = self.generate_single_episode(episode_id=len(episodes))
                if episode:
                    episodes.append(episode)
                    pbar.update(1)
        dataset.episodes += episodes
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "mobility_episodes.json.gz")
        
        if output_path is None:
            output_path = "mobility_episodes.json.gz"
        elif osp.isdir(output_path) or output_path.endswith("/"):
            output_path = (
                osp.abspath(output_path) + "/mobility_episodes.json.gz"
            )
        else:
            if not output_path.endswith(".json.gz"):
                output_path += ".json.gz"

        if (
            not osp.exists(osp.dirname(output_path))
            and len(osp.dirname(output_path)) > 0
        ):
            os.makedirs(osp.dirname(output_path))
        import gzip

        with gzip.open(output_path, "wt") as f:
            f.write(dataset.to_json())

        print(f"Episodes saved to {output_path}")

if __name__ == "__main__":
    config_path = "/home/yuchecheng/habitat-lab/habitat-lab/habitat/datasets/rearrange/configs/mp3d.yaml" 
    output_dir = "/home/yuchecheng/habitat-lab/data/datasets/mobility"
    scene_dataset_path = "/home/yuchecheng/habitat-lab/data/scene_datasets/mp3d/"     
    num_episodes = 10

    assert num_episodes > 0, "Number of episodes must be greater than 0."
    assert osp.exists(
            config_path
        ), f"Provided config, '{config_path}', does not exist."
        
    cfg = get_config_defaults()
    override_config = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, override_config)

    with MobilityGenerator(
        cfg=cfg,
        debug_visualization=False,
        num_episodes=num_episodes,
    ) as mo_gen:
        mo_gen.generate_mobility_episodes(output_dir)
    pass
