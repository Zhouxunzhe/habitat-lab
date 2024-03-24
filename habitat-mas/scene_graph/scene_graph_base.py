import json
from math import dist
import os
from abc import ABC, abstractmethod, abstractproperty
from turtle import color
from typing import List
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from habitat_sim import Simulator, scene

# local import
from scene_graph.config import SceneGraphHabitatConfig
from scene_graph.region_layer import RegionLayer, RegionNode
from scene_graph.object_layer import ObjectLayer, ObjectNode
from scene_graph.agent_layer import AgentLayer, AgentNode
from scene_graph.utils import get_corners
from perception.grid_map import GridMap
from perception.nav_mesh import NavMesh

"""
########################## Update log ###################################
2021/11/06, Junting Chen: Only consider the scene graph of one level in one building 
"""


class SceneGraphBase:

    """Presume that the scene graph have three layers"""
    """ Set a layer to None if it does not exist in class instance """

    def __init__(self) -> None:
        """Initialize scene graph on different 3D datasets"""
        """Parsing of 3D datasets should be implemented in dataset module"""
        self.region_layer = RegionLayer()
        self.object_layer = ObjectLayer()
        self.agent_layer = AgentLayer()
        
        # Optional members
        self.sim: Simulator = None
        self.grid_map: GridMap = None
        self.nav_mesh: NavMesh = None
        
    def get_full_graph(self):
        """Return the full scene graph"""
        raise NotImplementedError

    def generate_scene_description(self):
        """Generate scene description for the scene graph"""
        raise NotImplementedError
    
    def sample_graph(self, method, *args, **kwargs):
        """Return the sub-sampled scene graph
        
        Assume all method-dependent variables passed through kwargs 
        
        """
        if method == "radius_sampling":
            # get params
            sample_centers = kwargs.get("center")
            sample_radius = kwargs.get("radius")
            # by default, calculate distance on x-y plane
            dist_dims = kwargs.get("dist_dims", [0,1])
            if len(sample_centers.shape) == 1:
                # add dummy dim 
                sample_centers = sample_centers[np.newaxis, :]
                
            # build the kdtree to query objects inside the ball/circle
            obj_ids = self.object_layer.obj_ids
            if len(obj_ids) == 0: # empty scene graph
                return [[] for _ in range(sample_centers.shape[0])]
            
            obj_centers = self.object_layer.get_centers(obj_ids)
            kdtree = KDTree(obj_centers[:, dist_dims])
            sample_idx_list = kdtree.query_ball_point(
                sample_centers[:, dist_dims], sample_radius)
            sample_obj_ids_list = [[obj_ids[idx] for idx in sample_idx] 
                                   for sample_idx in sample_idx_list]
            return sample_obj_ids_list
        
        elif method == "soft_radius_sampling":
            sample_centers = kwargs.get("center")
            if len(sample_centers.shape) == 1:
                # add dummy dim 
                sample_centers = sample_centers[np.newaxis, :]
                
            obj_ids = self.object_layer.obj_ids
            if len(obj_ids) == 0: # empty scene graph
                return [[] for _ in range(sample_centers.shape[0])], \
                    [[] for _ in range(sample_centers.shape[0])]
            obj_centers = self.object_layer.get_centers(obj_ids)
            
            sample_obj_ids_list = []
            sample_obj_dists_list = []
            if self.object_layer.flag_grid_map and False: 
                # if there is grid map, dist represented by geodesic distance 
                # TODO: consider how to calculate obj distance to current pos
                pass
            else:
                # if the grid map is not initialized, use manhattan distance instead
                
                for center in sample_centers:  
                    sample_obj_ids_list.append(obj_ids)
                    dists = abs(center[0] - obj_centers[:, 0]) + \
                        abs(center[1] - obj_centers[:, 1])
                    sample_obj_dists_list.append(dists)
                return sample_obj_ids_list, sample_obj_dists_list
        else:
            raise NotImplementedError
        return 

