import numpy as np
import magnum as mn
# project_size = 2.0
projection_matrix = np.array([[1, 0, 0, 0],
       [0, 1, 0, 0],
      [0, 0, -1.00002, -0.0200002],
       [0, 0, -1, 0]])
viewport = [512,512]   #because the new sampling range is [512,512]
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
        point_2d *= mn.Vector2(W,H)
        out_bound = 10
        point_2d = np.nan_to_num(point_2d, nan=W+out_bound, posinf=W+out_bound, neginf=-out_bound)
        return point_2d.astype(int).tolist()

def _2d_to_3d(matrix, point_2d, depth=10000.0):
    W, H = viewport[0], viewport[1]
    
    # 将2D点归一化到[-1, 1]范围
    point_2d_normalized = (np.array(point_2d) / np.array([W, H])) * 2.0 - 1.0
    point_2d_normalized = np.array([point_2d_normalized[0], -point_2d_normalized[1], 1.0])  # Z为1.0
    
    # 使用投影矩阵的逆矩阵
    point_2d_normalized = mn.Vector3(point_2d_normalized)
    inv_proj_matrix = mn.Matrix4(projection_matrix).inverted()
    point_3d_camera = inv_proj_matrix.transform_point(point_2d_normalized)
    
    # 使用相机矩阵（视图矩阵）的逆矩阵
    inv_camera_matrix = mn.Matrix4(matrix).inverted()
    point_3d_world = inv_camera_matrix.transform_point(point_3d_camera * depth)
    
    return list(point_3d_world)

# martix = [
#           [
#             -0.8103205561637878,
#             -0.0,
#             0.585986852645874,
#             3.3535726070404053
#           ],
#           [
#             0.21763013303279877,
#             0.9284766912460327,
#             0.30094560980796814,
#             -0.2418685257434845
#           ],
#           [
#             -0.5440751314163208,
#             0.3713907301425934,
#             -0.7523638010025024,
#             -3.0943806171417236
#           ]
# ]
# worldpoc = [
#         1.2668886184692383,
#         0.17379358410835266,
#         -3.1019668579101562,
#         -0.6113674640655518
#     ]
# print("transbefore:",worldpoc[:3])
# d_point = _3d_to_2d(matrix=martix,point_3d=worldpoc[:3])
# print("dp:",d_point)
# print("trans:",_2d_to_3d(matrix=martix,point_2d=d_point))