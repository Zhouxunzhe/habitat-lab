import math
from tqdm import tqdm
import magnum as mn
from habitat.datasets.rearrange.run_episode_generator import get_config_defaults
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode, RearrangeDatasetV0
from habitat.datasets.rearrange.samplers.receptacle import Receptacle, AABBReceptacle, parse_receptacles_from_user_config
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
    is_accessible,
)
from habitat.core.logging import logger
import habitat_sim
import json
import os
import os.path as osp
from omegaconf import OmegaConf
from typing import(
    Dict, Generator, List, Optional, Sequence, Tuple, Union, Any
)
from habitat.config import DictConfig
import numpy as np
import random
from habitat_sim.nav import NavMeshSettings
from habitat_sim.utils.common import quat_from_two_vectors as qf2v
from habitat_sim.utils.common import quat_to_magnum as qtm
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

def geodesic_distance(
        sim: "HabitatSim",
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[Sequence[float], np.ndarray],
    ) -> float:
        path = habitat_sim.ShortestPath()
        path.requested_end = np.array(position_b, dtype=np.float32).reshape(3,1)
        path.requested_start = np.array(position_a, dtype=np.float32)
        found_path = sim.pathfinder.find_path(path)

        return found_path, path.geodesic_distance

def is_compatible_episode(
    s: Sequence[float],
    t: Sequence[float],
    sim: "HabitatSim",
    near_dist: float,
    far_dist: float,
    geodesic_to_euclid_ratio: float,
) -> Union[Tuple[bool, float], Tuple[bool, int]]:
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)

    # In mobility task, s and t may not be on the same floor, 
    # we need to check height difference
    height_dist = np.abs(s[1] - t[1])
    found_path, d_separation = geodesic_distance(sim, s, t)
    if not found_path:
        return False, 0, height_dist
    if not near_dist <= d_separation <= far_dist:
        return False, 0, height_dist
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0, height_dist

    return True, float(d_separation), float(height_dist)

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
        num_episodes: int = 1,
        scene_dataset_path: str = "data/scene_datasets/mp3d/"
    ) -> None:

        # load and cache the config
        self.cfg = cfg
        self.num_episodes = num_episodes
        self.dataset_path = scene_dataset_path
        # hold a habitat Simulator object for efficient re-use
        self.sim: habitat_sim.Simulator = None
        # initialize an empty scene and load the SceneDataset
        self.initialize_sim(self.dataset_path)

        self.failed_episodes_num = 0

# sample the scene from config and initialize the Simulator
    def initialize_sim(self, dataset_path: str = "data/scene_datasets/mp3d/") -> None:
        backend_cfg = habitat_sim.SimulatorConfiguration()
        # randomly sample the scene from config
        sub_scene_paths = [d for d in os.listdir(dataset_path) if osp.isdir(osp.join(dataset_path, d))]
        if not sub_scene_paths or len(sub_scene_paths) == 1:
            raise ValueError("No sub scene paths found in dataset path.")
        else:
            selected_sub_scene = random.choice(sub_scene_paths)
            scene_glb_path = osp.join(dataset_path, selected_sub_scene, f"{selected_sub_scene}.glb")

        backend_cfg.scene_id = scene_glb_path
        backend_cfg.scene_dataset_config_file = self.cfg.dataset_path
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
            for object_path in self.cfg.additional_object_paths:
                object_attr_mgr.load_configs(osp.abspath(object_path))
        else:
            if self.sim.config.sim_cfg.scene_id != scene_glb_path:
                self.sim.close(destroy=True)
            if self.sim.config.sim_cfg.scene_id == scene_glb_path:
                # we need to force a reset, so reload the NONE scene
                proxy_backend_cfg = habitat_sim.SimulatorConfiguration()
                proxy_backend_cfg.scene_id = "NONE"
                proxy_backend_cfg.gpu_device_id = self.cfg.gpu_device_id
                proxy_hab_cfg = habitat_sim.Configuration(
                    proxy_backend_cfg, [agent_cfg]
                )
                self.sim.reconfigure(proxy_hab_cfg)
            self.sim.reconfigure(sim_cfg)

        return scene_glb_path

    def safe_snap_point(self, point, island_idx) -> np.ndarray:
        new_pos = self.sim.pathfinder.snap_point(
            point, island_idx
        )

        num_sample_points = 2000
        max_iter = 10
        offset_distance = 0.5
        distance_per_iter = 0.5
        regen_i = 0

        while np.isnan(new_pos[0]) and regen_i < max_iter:
            new_pos = self.sim.pathfinder.get_random_navigable_point_near(
                point,
                offset_distance + regen_i * distance_per_iter,
                num_sample_points,
                island_index=island_idx,
            )
            regen_i += 1

        return new_pos
    def get_global_transform(self, translation: mn.Vector3, rotation = mn.Quaternion()) -> mn.Matrix4:
        stage_config = self.sim.get_stage_initialization_template()
        r_frameup_worldup = qf2v(
            habitat_sim.geo.UP, stage_config.orient_up
        )
        v_prime = qtm(r_frameup_worldup).transform_vector(
            mn.Vector3(habitat_sim.geo.FRONT)
        )
        world_to_local = (
            qf2v(np.array(v_prime), np.array(stage_config.orient_front))
            * r_frameup_worldup
        )
        world_to_local = habitat_sim.utils.common.quat_to_magnum(
            world_to_local
        )
        local_to_world = world_to_local.inverted()
        l2w4 = mn.Matrix4.from_(local_to_world.to_matrix(), mn.Vector3())

        # apply the receptacle rotation from the bb center
        T = mn.Matrix4.from_(mn.Matrix3(), translation)
        R = mn.Matrix4.from_(rotation.to_matrix(), mn.Vector3())
        # translate frame to center, rotate, translate back
        l2w4 = l2w4.__matmul__(T.__matmul__(R).__matmul__(T.inverted()))
        return l2w4
    
    def generate_single_episode(
        self,
        episode_id: int,
        closest_dist_limit: float = 3.0,
        furthest_dist_limit: float = 80.0,
        geodesic_to_euclid_min_ratio: float = 1.1,
        max_placement_tries: int = 100,
    ) -> Optional[RearrangeEpisode]:
        
        ep_scene_handle = self.initialize_sim(self.dataset_path)
        scene_base_dir = osp.dirname(osp.dirname(ep_scene_handle))
        scene_name = ep_scene_handle.split(".")[0].split("/")[-1]
        navmesh_path = osp.join(
            scene_base_dir, scene_name, scene_name + ".navmesh"
        )

        assert osp.exists(navmesh_path), f"Navmesh does not exist at {navmesh_path}"

        # Load navmesh
        self.sim.pathfinder.load_nav_mesh(navmesh_path)
        logger.info(f"Loaded navmesh from {navmesh_path}")

        if len(self.sim.semantic_scene.levels) < 2:
            return None
        
        largest_island_idx = get_largest_island_index(
            self.sim.pathfinder, self.sim, allow_outdoor=False
        ) 

        semantic_scene = self.sim.semantic_scene
        rom = self.sim.get_rigid_object_manager()

        is_compatible = False
        for _retry in range(max_placement_tries):
            target_level = random.choice(semantic_scene.levels)
            goal_level = target_level
            while goal_level == target_level:
                goal_level = random.choice(semantic_scene.levels)
            
            target_region = random.choice(target_level.regions)
            goal_region = random.choice(goal_level.regions)
            
            target_object = random.choice(target_region.objects)
            goal_object = random.choice(goal_region.objects)

            target_pos = self.safe_snap_point(target_object.aabb.center, largest_island_idx)
            goal_pos = self.safe_snap_point(goal_object.aabb.center, largest_island_idx)

            if np.isnan(target_pos[0]) or np.isnan(goal_pos[0]):
                continue
            
            is_compatible, dist, height_dist = is_compatible_episode(
                goal_pos, 
                target_pos,
                self.sim, 
                closest_dist_limit, 
                furthest_dist_limit,
                geodesic_to_euclid_min_ratio
            )
            if is_compatible:
                break
        
        if np.isnan(target_pos[0]) or np.isnan(goal_pos[0]):
            return None
        if not is_compatible or height_dist <= 2.0:
            return None
        
        target_transform = self.get_global_transform(translation=target_pos)
        goal_transform = self.get_global_transform(translation=goal_pos)
        return RearrangeEpisode(
            episode_id=episode_id,

        )



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
    config_path = "habitat-lab/habitat/datasets/rearrange/configs/mp3d.yaml" 
    output_dir = "data/datasets/mobility"
    scene_dataset_path = "data/scene_datasets/mp3d/"     
    num_episodes = 1

    assert num_episodes > 0, "Number of episodes must be greater than 0."
    assert osp.exists(
            config_path
        ), f"Provided config, '{config_path}', does not exist."
        
    cfg = get_config_defaults()
    override_config = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, override_config)

    with MobilityGenerator(
        cfg=cfg,
        num_episodes=num_episodes,
        scene_dataset_path=scene_dataset_path
    ) as mo_gen:
        mo_gen.generate_mobility_episodes(output_dir)
    pass
