import copy
from typing import List, Dict, Tuple
import numpy as np
import open3d as o3d
from scene_graph.utils import project_points_to_grid_xz


class AgentNode:
    def __init__(
        self,
        agent_id,
        position,
        orientation,
        bbox,
        description="",
    ):

        # required attributes
        self.agent_id = agent_id
        self.position = position  # [x,y,z]
        self.orientation = orientation  # [x,y,z,w]
        self.bbox = bbox  # [x_min, y_min, z_min, x_max, y_max, z_max]  
        self.description = description
        
    def generate_agent_description(self):
        
        raise NotImplementedError

class AgentLayer:
    def __init__(self):

        self.flag_grid_map = False
        self.agent_ids: List = []
        self.agent_dict: Dict[str, AgentNode] = {}

    def __len__(self):
        return len(self.agent_ids)

    def init_map(self, bounds, grid_size, free_space_grid, project_mode="xz"):

        self.bounds = (
            bounds  # the real map area corresponding to free space grid
        )
        self.grid_size = grid_size
        self.segment_grid = np.zeros_like(free_space_grid, dtype=int) - 1
        self.free_space_grid = np.array(free_space_grid).astype(bool)
        self.project_mode = (
            project_mode  # 'xz' means project xz axes in 3D to xy axes in map
        )

        # implement the data structure for closest grid point searching
        # self.grid_kd_tree = cKDTree(self.free_space_grid)
        self.flag_grid_map = True
        return

    def add_agent(
        self,
        agent_id,
        position,
        orientation,
        bbox,
        description="",
    ):

        agent = AgentNode(agent_id, position, orientation, bbox, description)
        self.agent_ids.append(agent_id)
        self.agent_dict[agent_id]= agent

        return agent
    
    def segment_agent_on_grid_map_xz(self, agent_id):
        # TODO: deprecate grid map or add floor node to manage grid map 
        assert (
            self.flag_grid_map
        ), "called 'segment_region_on_grid_map_xz()' before grid map being initialized"
        agent_bbox = self.agent_dict[agent_id].bbox
        agent_2d_bbox = project_points_to_grid_xz(
            self.bounds, agent_bbox, self.grid_size
        )

        agent_mask = np.zeros_like(self.free_space_grid).astype(bool)
        # NOTE: (row_idx, col_idx) corresponds to (y, x) in 2d grid map
        agent_mask[
            int(np.ceil(agent_2d_bbox[0][1])) : int(np.floor(agent_2d_bbox[1][1])),
            int(np.ceil(agent_2d_bbox[0][0])) : int(np.floor(agent_2d_bbox[1][0])),
        ] = True
        # color the region on global grid map
        if (self.segment_grid[agent_mask] == -1).all():
            self.segment_grid[agent_mask] = agent_id
        else:
            print(
                f"Warning: agentect {agent_id} overlap with agentect {self.segment_grid[agent_mask]}"
            )
            self.segment_grid[agent_mask] = agent_id

        return agent_mask
    