import numpy as np
import open3d as o3d

class NavMesh:
    """Wrapper for navmesh stored as triangle meshes and helper function"""
    
    def __init__(self, vertices, triangles, **kwargs):
        self.vertices = vertices
        self.triangles = triangles.reshape(-1, 3)
        
        self.mesh = self._build_triangle_mesh(vertices, self.triangles)
        
        # by default, compute navmesh sgementation 
        self.slope_threshold = kwargs.get("slope_threshold", 20)
        self.is_flat_ground = self.segment_by_slope(slope_threshold=self.slope_threshold)
       
    def _build_triangle_mesh(self, vertices, triangles):
        """Build a triangle mesh from vertices and triangles"""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        return mesh
    
    def segment_by_slope(self, slope_threshold=20):
        """Segment the navmesh by slope angle"""
        
        assert self.mesh is not None, "Mesh is not initialized"
        
        # Define the up direction (assuming y-up coordinate frame)
        up = np.array([0, 1, 0])
        
        # Get the triangle normals as a NumPy array
        triangle_normals = np.asarray(self.mesh.triangle_normals)
        
        # Create a list to store the triangle colors
        triangle_colors = []
        
        # Compute the angle between the triangle normal and the up direction for all triangles
        angles = np.degrees(np.arccos(np.dot(triangle_normals, up)))

        # Check if the angles are within the flat-ground threshold
        is_flat_ground = angles < slope_threshold

        # Flat-ground surfaces (green color)
        triangle_colors = np.where(is_flat_ground[:, None], [0, 1, 0], [1, 0, 0])
        
        
        # Assign vertex colors by max voting of triangle colors
        vertex_colors = np.zeros_like(np.asarray(self.mesh.vertices))
        for i, triangle in enumerate(self.mesh.triangles):
            for j in range(3):
                vertex_colors[triangle[j]] += triangle_colors[i]
        
        # Normalize the vertex colors
        vertex_colors /= np.linalg.norm(vertex_colors, axis=1)[:, None]
        
        # Set the vertex colors
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        return is_flat_ground
    

