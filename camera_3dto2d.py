import numpy as np
import magnum as mn
project_size = 2.0
projection_matrix = np.array([[1, 0, 0, 0],
       [0, 1, 0, 0],
      [0, 0, -1.00002, -0.0200002],
       [0, 0, -1, 0]])
viewport = [256,256]
def _3d_to_2d(matrix, point_3d):
        # get the scene render camera and sensor object
        W, H = viewport[0], viewport[1]

        # use the camera and projection matrices to transform the point onto the near plane
        project_mar = projection_matrix
        # print("mar:",np.append(point_3d,1),np.array(matrix).reshape(4, 4))s
        cam_mat = mn.Matrix4(matrix)
        # point_transform = cam_mat.transform_point(point_3d)
        point_transform = np.dot(matrix,np.append(point_3d,1))
        point_transs = mn.Vector3(point_transform[:3])
        point_mat = mn.Matrix4(project_mar)
        projected_point_3d = point_mat.transform_point(point_transs)
        # convert the 3D near plane point to integer pixel space
        point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
        point_2d = point_2d / 2.0
        point_2d += mn.Vector2(0.5)
        point_2d *= mn.Vector2(256,256)
        out_bound = 10
        point_2d = np.nan_to_num(point_2d, nan=W+out_bound, posinf=W+out_bound, neginf=-out_bound)
        return point_2d.astype(int).tolist()