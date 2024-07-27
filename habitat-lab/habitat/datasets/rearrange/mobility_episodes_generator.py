from re import S, X
import math
import magnum as mn
from habitat.config.default import get_config
from habitat.core.dataset import Episode
import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.core.simulator import Simulator
from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
from habitat.datasets.rearrange.mobility_dataset import Rearrange3DEpisode, RearrangeDatasetV1, RearrangeSpec, RearrangeObjectSpec
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
    is_accessible,
)
from habitat.core.logging import logger
from habitat.utils.common import cull_string_list_by_substrings
import habitat_sim
import json
import os
import os.path as osp
from omegaconf import OmegaConf
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union, Any
from habitat.config import DictConfig
from sympy import root
from torch import mm

import numpy as np
import random

try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException
try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    habitat_sim = BaseException
ISLAND_RADIUS_LIMIT = 1.5

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


# TODO(YCC): sample the scene from config and initialize the Simulator
def initialize_sim(sim: "HabitatSim", cfg: DictConfig, dataset_path: str) -> "HabitatSim":
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
    if sim is None:
        sim = habitat_sim.Simulator(sim_cfg)
        object_attr_mgr = sim.get_object_template_manager()
        for object_path in cfg.additional_object_paths:
            object_attr_mgr.load_configs(osp.abspath(object_path))
    else:
        if sim.config.sim_cfg.scene_id != scene_glb_path:
            sim.close(destroy=True)
        if sim.config.sim_cfg.scene_id == scene_glb_path:
            # we need to force a reset, so reload the NONE scene
            proxy_backend_cfg = habitat_sim.SimulatorConfiguration()
            proxy_backend_cfg.scene_id = "NONE"
            proxy_backend_cfg.gpu_device_id = cfg.gpu_device_id
            proxy_hab_cfg = habitat_sim.Configuration(
                proxy_backend_cfg, [agent_cfg]
            )
            sim.reconfigure(proxy_hab_cfg)
        sim.reconfigure(sim_cfg)
    return sim

# TODO(YCC):generate single episode, return episode as NavigationEpisode
def generate_single_episode(
    sim: "HabitatSim",
    episode_id: int,
    cfg: DictConfig,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
    max_placement_tries: int = 10,
    scene_dataset_path: str = "/home/yuchecheng/habitat-lab/data/scene_datasets/mp3d_habitat/mp3d/",
) -> Generator[Rearrange3DEpisode, None, None]:
    
    sim = initialize_sim(sim, cfg, scene_dataset_path)

    # TODO(YCC): randomly sample the object set
    obj_sets: Dict[str, List[str]] = {}
    expected_list_keys = ["included_substrings", "excluded_substrings"]
    sampled_objects = {}
    for obj_set in cfg.object_sets:
        assert "name" in obj_set
        for list_key in expected_list_keys:
            assert list_key in obj_set
            assert isinstance(obj_set[list_key], Sequence)
        obj_sets[obj_set["name"]] = cull_string_list_by_substrings(
            sim.get_object_template_manager().get_template_handles(),
            obj_set["included_substrings"],
            obj_set["excluded_substrings"],
        )
        sampled_object = obj_sets[obj_set['name']][np.random.randint(0, len(obj_sets[obj_set['name']]))]
        sampled_objects[obj_set['name']] = sampled_object

    # TODO(YCC): flag the object placement: 1. robot start position and object start position are not in the same floor
    # 2. robot start position and object goal position are not in the same floor
    same_floor = True

    target_position = sim.pathfinder.get_random_navigable_point().tolist()

    # if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
    while sim.pathfinder.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
        target_position = sim.pathfinder.get_random_navigable_point().tolist()

    # TODO(YCC): randomly sample the robot start position and test if it is compatible with the target position
    for _retry in range(number_retries_per_target):
        # source_position = sim.sample_navigable_point()
        source_position = sim.pathfinder.get_random_navigable_point().tolist()
        while sim.pathfinder.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
            source_position = sim.pathfinder.get_random_navigable_point().tolist()
        is_compatible, dist, height_dist = is_compatible_episode(
            source_position,
            target_position,
            sim,
            near_dist=closest_dist_limit,
            far_dist=furthest_dist_limit,
            geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
        )
        if is_compatible:
            break
    if not is_compatible:
        return None
    
    if height_dist > 0.5:
        same_floor = False

    num_placement_tries = 0
    new_object = {}
    for name, object_handle in sampled_objects.items():
        while num_placement_tries < max_placement_tries:
            num_placement_tries += 1
            # TODO(YCC): randomly sample the object location, and try to place it
            object_position = sim.pathfinder.get_random_navigable_point().tolist()
            while sim.pathfinder.island_radius(object_position) < ISLAND_RADIUS_LIMIT:
                object_position = sim.pathfinder.get_random_navigable_point().tolist()
            if new_object == {}:
                assert sim.get_object_template_manager().get_library_has_handle(
                    object_handle
                ), f"Object handle {object_handle} not found in library."
                new_object[name] = sim.get_rigid_object_manager().add_object_by_template_handle(
                    object_handle
                )
            new_object[name].translation = object_position
            new_object[name].rotation = mn.Quaternion.rotation(
                mn.Rad(random.uniform(0, math.pi * 2.0)), mn.Vector3.y_axis()
            )
            if not is_accessible(
                sim=sim,
                point=new_object[name].translation,
                height=1,
                nav_to_min_distance=-1.0,
                nav_island=get_largest_island_index(sim.pathfinder, sim, allow_outdoor=False),
                target_object_id=new_object[name].object_id
            ):
                new_object = {}
                continue
            break
    objects = []
    object_num = 0
    for name, object_handle in sampled_objects.items():
        # object_filename = object_handle.split("/")[-1]
        object_filename = osp.relpath(object_handle)
        matrix_list = np.array([[new_object[name].transformation[i][j] for j in range(4)] for i in range(4)])
        obj = RearrangeObjectSpec(
            position=[new_object[name].translation.x, new_object[name].translation.y, new_object[name].translation.z],
            rotation=[0.0, 0.0, 0.0, 1.0],
            object_id=str(object_num),
            object_handle=object_filename,
            object_transform=matrix_list
        )
        # TODO(YCC): calculate the height difference between robot source position and object source position
        robot_object_height_diff = abs(obj.position[1] - source_position[1])
        if robot_object_height_diff > 0.5:
            same_floor = False
        obj1 = (object_filename, matrix_list)
        objects.append(obj1)
        # objects.append(obj)
        object_num += 1

    angle = np.random.uniform(0, 2 * np.pi)
    source_rotation = [0.0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    goals = []
    goal = RearrangeSpec(
        position=target_position,
        rotation=[0.0, 0.0, 0.0, 1.0]
    )
    goals.append(goal)

    return Rearrange3DEpisode(
        scene_dataset_config=sim.config.sim_cfg.scene_dataset_config_file,
        additional_obj_config_paths=cfg.additional_object_paths,
        episode_id=str(episode_id),
        start_position=source_position,
        start_rotation=source_rotation,
        scene_id=sim.config.sim_cfg.scene_id,
        rigid_objs=objects,
        ao_states={},
        target_receptacles=None,
        targets=None,
        goal_receptacles=None,
        name_to_receptacle=None,
        markers={},
        info={
            "object_labels": None,
            "geodesic_distance": dist, 
            "source_target_height_difference": height_dist, 
            "robot_object_height_difference": robot_object_height_diff,
            "same_floor": same_floor
            },
        goals=goals,
        # objects=objects
    )

#TODO(YCC):generate HM3D/MP3D episodes
def generate_mobility_episodes(config_path, output_dir, num_episodes, dataset = "mp3d", scene_dataset_path = "/home/yuchecheng/habitat-lab/data/scene_datasets/mp3d_habitat/mp3d/"):
    assert num_episodes > 0, "Number of episodes must be greater than 0."
    assert osp.exists(
            config_path
        ), f"Provided config, '{config_path}', does not exist."
        
    cfg = OmegaConf.load(config_path)

    logger.info(f"\n\nConfig:\n{cfg}\n\n")
    sim = None
    dataset = RearrangeDatasetV1()
    episodes = []
    while len(episodes) < num_episodes:
        episode = generate_single_episode(sim, episode_id=len(episodes), cfg=cfg)
        if episode:
            episodes.append(episode)
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
    scene_dataset_path = "/home/yuchecheng/habitat-lab/data/scene_datasets/mp3d_habitat/mp3d/"
    dataset = "mp3d"       
    num_episodes = 10  

    generate_mobility_episodes(config_path, output_dir, num_episodes, dataset, scene_dataset_path)
    pass
