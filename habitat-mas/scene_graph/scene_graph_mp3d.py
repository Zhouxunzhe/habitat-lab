from functools import partial
import os
import sys
import time
from typing import List
import pathlib
import numpy as np
import pickle
import open3d as o3d
import copy 
import plotly.graph_objects as go
from scipy import stats
from scipy.ndimage.morphology import binary_dilation
import quaternion as qt
from habitat_sim import Simulator

from utils.constants import coco_categories, coco_label_mapping
from scene_graph.scene_graph_base import SceneGraphBase
from scene_graph.utils import (
    visualize_scene_graph,
)
from perception.grid_map import GridMap
from perception.nav_mesh import NavMesh
from perception.mesh_utils import (
    compute_triangle_adjacency,
    propagate_triangle_region_ids,
    build_region_triangle_adjacency_graph
)

class SceneGraphMP3D(SceneGraphBase):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.sim: Simulator = None
        self.gt_point_cloud = None
        
        self.meters_per_grid = kwargs.get('meters_per_grid', 0.05)
        self.object_grid_scale = kwargs.get('object_grid_scale', 1)
        self.aligned_bbox = kwargs.get('aligned_bbox', True)
        self.enable_region_layer = kwargs.get('enable_region_layer', True)
        # Util habitat-sim v0.3.1, HM3D region annotation missing
        # self.compute_region_bbox = kwargs.get('compute_region_bbox', True)

    def load_gt_scene_graph(self, sim: Simulator):
        # register habitat simulator
        self.sim = sim
        # get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = self.sim.pathfinder.get_bounds()
        self.current_height = sim.get_agent(0).state.position[1]

        # 1. load navmesh from habitat simulator
        navmesh_vertices: List[np.ndarray] = self.sim.pathfinder.build_navmesh_vertices()
        navmesh_indices: List[int]= self.sim.pathfinder.build_navmesh_vertex_indices()
        self.nav_mesh = NavMesh(
            vertices=np.stack(navmesh_vertices, axis=0),
            triangles=np.array(navmesh_indices).reshape(-1, 3),
        )
        
        semantic_scene = self.sim.semantic_scene
        # 2. load region layer from habitat simulator
        if self.enable_region_layer: # matterport 3D has region annotations 
            for region in semantic_scene.regions:
            # add region node to region layer
                region_id = int(region.id.split("_")[-1]) # counting from 0, -1 for background
                if region_id < 0:
                    continue

                region_bbox = np.stack(
                    [
                        region.aabb.center - region.aabb.sizes / 2,
                        region.aabb.center + region.aabb.sizes / 2,
                    ],
                    axis=0,
                )
                
                # Add region node to region layer 
                region_class_name = None
                region_label = None
                if region.category is not None:
                    region_class_name = region.category.name()
                    region_label = region.category.index()
                
                region_node = self.region_layer.add_region(
                    region_bbox,
                    id=region_id,
                    class_name=region_class_name,
                    label=region_label,
                )

                # 3. load object layer from habitat simulator
                for obj in region.objects:
                    if obj is not None:
                        object_id = int(obj.id)  
                        if self.aligned_bbox:
                            center = obj.aabb.center
                            rot_quat = np.array([0, 0, 0, 1])  # identity transform
                            size = obj.aabb.sizes
                        else:  # Use obb, NOTE: quaternion is [w,x,y,z] from habitat, need to convert to [x,y,z,w]
                            center = obj.obb.center
                            rot_quat = obj.obb.rotation[1, 2, 3, 0]
                            size = obj.obb.sizes
                            size = obj.aabb.sizes

                        node_size = (
                            self.meters_per_grid / self.object_grid_scale
                        )  # treat object as a point
                        node_bbox = np.stack(
                            [center - node_size / 2, center + node_size / 2], axis=0
                        )
                        object_node = self.object_layer.add_object(
                            center,
                            rot_quat,
                            size,
                            id=object_id,
                            class_name=obj.category.name(),
                            label=obj.category.index(),
                            bbox=node_bbox,
                        )

                        # connect object to region
                        region_node.add_object(object_node)
                    
    def load_gt_geometry(self):
        # load the ply file of the scene
        scene_id = self.sim.config.sim_cfg.scene_id
        ply_file_path = scene_id[:-4] + "_semantic.ply"
        
        # load the ply file with open3d 
        if os.path.exists(ply_file_path):
            self.gt_point_cloud = o3d.io.read_point_cloud(ply_file_path)
        else:
            print(f"PLY file {ply_file_path} not found")


if __name__ == "__main__":
    import habitat_sim
    import os 

    data_dir = "/home/junting/repo/habitat-lab/data"

    # initialize habitat sim
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = f"{data_dir}/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
    backend_cfg.scene_dataset_config_file = f"{data_dir}/scene_datasets/mp3d/mp3d.scene_dataset_config.json"

    sem_cfg = habitat_sim.CameraSensorSpec()
    sem_cfg.uuid = "semantic"
    sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sem_cfg]

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)

    # initialize scene graph
    sg = SceneGraphMP3D()
    sg.load_gt_scene_graph(sim)
    sg.load_gt_geometry()

    # Segment navigation mesh by region bounding box
    region_bbox_dict = {
        region_id: region.bbox for region_id, region in sg.region_layer.region_dict.items()
    }

    # save the navmesh triangle mesh and region bounding box dict to files
    # o3d.io.write_triangle_mesh("1LXtFkjw3qL.obj", sg.nav_mesh.mesh)
    # pickle.dump(region_bbox_dict, open("1LXtFkjw3qL_region_bbox_dict.pkl", "wb"))

    # algorithm to build abstract scene graph:
    # 1) segment navmesh with region bbox; 2) propogate non-labeled triangles 3) build adjaceny graph
    sg.nav_mesh.segment_by_region_bbox(region_bbox_dict)
    sg.nav_mesh.triangle_region_ids = propagate_triangle_region_ids(
        sg.nav_mesh.triangle_region_ids, sg.nav_mesh.triangle_adjacency_list
    )
    abs_graph = build_region_triangle_adjacency_graph(
        sg.nav_mesh.triangle_region_ids, sg.nav_mesh.triangle_adjacency_list
    )

    # assign region name and bbox center to each region node
    for region_id in abs_graph.nodes:
        abs_graph.nodes[region_id]["name"] = sg.region_layer.region_dict[region_id].class_name + str(region_id)
        abs_graph.nodes[region_id]["bbox_center"] = sg.region_layer.region_dict[region_id].bbox.mean(axis=0)

    sim.close()
    
    ############# Visualization ##################
    
    # visualize_scene_graph(
    #     scene_graph=sg,
    #     scene_o3d=sg.gt_point_cloud,
    #     vis_region_bbox=True,
    #     vis_object_bbox=False,
    #     vis_navmesh=True,
    #     navmesh_shift=[0, 0, -8.0],
    #     vis_region_graph=True,
    #     region_graph=abs_graph,
    #     region_graph_shift=[0, 0, 10.0],
    #     mp3d_coord=True
    # )

    ############ Generate scene description ###########
    def generate_region_scene_graph_description(abs_graph):
        num_regions = len(abs_graph.nodes)
        description = "There are {} regions in the scene.\n".format(num_regions)
        
        for region_id in abs_graph.nodes:
            region = abs_graph.nodes[region_id]
            description += "The {}-th region is named {} with a center at ({}).\n".format(
                region_id, region["name"], region["bbox_center"]
            )
            for neighbor_id in abs_graph.neighbors(region_id):
                description += "{} is connected to {}.\n".format(region["name"], abs_graph.nodes[neighbor_id]["name"])

        return description

    def generate_region_description(region_layer):
        description = "The following  text contains the objects contained in each region:\n"
        for region_id, region in region_layer.region_dict.items():
            region_full_name = region.class_name + str(region_id)
            description += "Region {} contains the following objects:\n".format(region_full_name)
            for obj in region.objects:
                description += f"{obj.class_name}_{obj.id}; "
            description += "\n"
            
        return description
        

    # Generate scene descriptions
    region_scene_graph_description = generate_region_scene_graph_description(abs_graph)
    region_description = generate_region_description(sg.region_layer)

    print(region_scene_graph_description)
    print("\n")
    print(region_description)
