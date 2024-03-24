# %%
import os
import numpy as np
import habitat_sim
from typing import List
from numpy.typing import ArrayLike
import open3d as o3d 
import math
from scene_graph.scene_graph_hm3d import SceneGraphHM3D

os.chdir("/home/junting/repo/habitat-lab")

# %%

# # initialize habitat sim 
# backend_cfg = habitat_sim.SimulatorConfiguration()
# backend_cfg.scene_id = "data/scene_datasets/hm3d/val/00891-cvZr5TUy5C5/cvZr5TUy5C5.basis.glb"
# backend_cfg.scene_dataset_config_file = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

# sem_cfg = habitat_sim.CameraSensorSpec()
# sem_cfg.uuid = "semantic"
# sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

# agent_cfg = habitat_sim.agent.AgentConfiguration()
# agent_cfg.sensor_specifications = [sem_cfg]

# sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
# sim = habitat_sim.Simulator(sim_cfg)

# # %%
# path_finder = sim.pathfinder
# navmesh_vertices: List[np.ndarray] = path_finder.build_navmesh_vertices()
# navmesh_indices:  List[int]= path_finder.build_navmesh_vertex_indices()

# # %%

# # save to o3d mesh 
# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(np.stack(navmesh_vertices, axis=0))
# mesh.triangles = o3d.utility.Vector3iVector(np.array(navmesh_indices).reshape(-1, 3))
# save to file 
# o3d.io.write_triangle_mesh("./00891-cvZr5TUy5C5_navmesh.obj", mesh)


# %%
# Load the triangle mesh
mesh = o3d.io.read_triangle_mesh("./00891-cvZr5TUy5C5_navmesh.obj")

# Check if the mesh was loaded successfully
if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
    print("Failed to load the mesh file.")
    exit(1)

# Compute triangle normals
mesh.compute_triangle_normals()

# Get the triangle normals as a NumPy array
triangle_normals = np.asarray(mesh.triangle_normals)

# Define the up direction (assuming y-up coordinate frame)
up = np.array([0, 1, 0])

# Set the angle threshold for flat-ground surfaces (in degrees)
flat_threshold = 20.0

# Create a list to store the triangle colors
triangle_colors = []

# Iterate over each triangle
for normal in triangle_normals:
    # Compute the angle between the triangle normal and the up direction
    angle = math.degrees(math.acos(np.dot(normal, up)))
    
    # Check if the angle is within the flat-ground threshold
    if angle < flat_threshold:
        # Flat-ground surface (green color)
        triangle_colors.append([0, 1, 0])
    else:
        # Stairs (red color)
        triangle_colors.append([1, 0, 0])

# Assign vertex colors by max voting of triangle colors
vertex_colors = np.zeros_like(np.asarray(mesh.vertices))
for i, triangle in enumerate(mesh.triangles):
    for j in range(3):
        vertex_colors[triangle[j]] += triangle_colors[i]
 
# Normalize the vertex colors
vertex_colors /= np.linalg.norm(vertex_colors, axis=1)[:, None]
    
# Set the vertex colors
mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
# Visualize the segmented mesh
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


