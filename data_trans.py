import json
import numpy as np
import cv2
import os
import magnum as mn
from PIL import Image
def trans_worldloc_to_robotloc(trans,loc):
    #用于将世界坐标系的点转换成机器人坐标系下的点
    trans = np.linalg.inv(trans)
    locs= np.append(loc,1)
    robotloc_pos = np.dot(trans,locs)
    return np.array(robotloc_pos)[:3]
def _project_points_to_image(points_3d,camera_info, cam_trans):
        #ArmRGBWorkspaceSensor的投影方式
        """
        Projects 3D points to 2D image pixel coordinates.

        Args:
            points_3d (np.ndarray): Nx3 array of 3D points.
            camera_info (dict): Dictionary containing camera intrinsics with keys 'fx', 'fy', 'cx', 'cy'.
            cam_trans (np.ndarray): 4x4 camera transformation matrix.

        Returns:
            np.ndarray: Nx2 array of 2D pixel coordinates.
        """
        # Extract camera intrinsics
        fx = camera_info['K'][0]
        fy = camera_info['K'][4]
        cx = camera_info['K'][2]
        cy = camera_info['K'][5]
        # Transform 3D points using the camera transformation matrix
        # points_3d_hom_match = np.append(point_3d_match,1)
        points_3d_hom = np.append(points_3d,1)
        cam_trans = np.asarray(cam_trans)
        # print("cam_trans",cam_trans)
        points_3d_transformed =  np.dot(cam_trans,points_3d_hom.T)
        # points_3d_transformed = cam_trans.inverted().transform_point(points_3d)
        # print("points_3d_transformed",points_3d_transformed)
        # Project the transformed 3D points onto the 2D image plane
        # x = points_3d_transformed[0, :]
        # y = points_3d_transformed[1, :]
        # z = points_3d_transformed[2, :]
        # point_camera_ho = np.array([points_3d_transformed[0],points_3d_transformed[1],points_3d_transformed[2],1])
        # pixel_coords = np.dot(np.asarray(camera_in),point_camera_ho[:3]/points_3d_transformed[2])
        x = points_3d_transformed[0]
        y = points_3d_transformed[1]
        z = np.abs(points_3d_transformed[2])
        # x,y,z = points_3d_transformed
        if z == 0:
             return np.array([0,0])
        else:
            u = (x * fx / z) + cx
            v = (y * fy / z) + cy
            # Stack u and v to get Nx2 array of 2D pixel coordinates
            # pixel_coords = np.vstack((u, v)).T
        pixel_coords = np.array([u,v])
        return pixel_coords[0],pixel_coords[1]
#camera_info一直不变
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
        return point_2d.astype(int)

data_path = './video_dir/image_dir/episode_159'
data_trans = []
skip_len = 60
with open(os.path.join(data_path,'sum_data.json'), 'r') as file:
    data = json.load(file)

for i in range(0,len(data['entities'])-skip_len):
    print(f"___________________________:{i}")
    step_data = data['entities'][i]
    next_step_data = data['entities'][i + skip_len] #取当前step的20步后的info，用于预测导航目标信息

    agent_0_trans_matrix = step_data['data']['agent_0_robot_trans_martix']
    # agent_1_trans_matrix = step_data['data']['agent_1_robot_trans_martix'] #获取agent当前step的转换矩阵信息
    # agent_0_nowloc = step_data['data']['agent_0_localization_sensor']
    # agent_1_nowloc = step_data['data']['agent_1_localization_sensor']
    agent_0_pre_worldloc = next_step_data['data']['agent_0_localization_sensor']
    # agent_1_pre_worldloc = next_step_data['data']['agent_1_localization_sensor']

    # agent_0_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_pre_worldloc[:3])
    # agent_1_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_pre_worldloc[:3])
    agent_0_objpos = step_data['data']['agent_0_obj_pos']
    # agent_1_objpos = step_data['data']['agent_1_obj_pos'] #获取当前目标物体在世界坐标系下的坐标
    agent_0_obj_ro = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_objpos[:3])
    # agent_1_obj_ro = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_objpos[:3]) #转移到机器人坐标系下
    
    agent_0_eeglobal = step_data['data']['agent_0_ee_global_pos_sensor']
    # agent_1_eeglobal = step_data['data']['agent_1_ee_global_pos_sensor']
    agent_0_eepos = step_data['data']['agent_0_ee_pos']
    # agent_1_eepos = step_data['data']['agent_1_ee_pos'] #末端执行器在机器人坐标系下的坐标
    agent_0_cam = step_data['data']['agent_0_camera_extrinsic']
    agent_0_obj_ = step_data['data']['agent_0_obj_bounding_box']
    agent_0_target_ = step_data['data']['agent_0_target_bounding_box']
    # agent_1_cam = step_data['data']['agent_1_camera_extrinsic'] #当前step的camera_extrinsic
    # print("agent_1_cam:",np.array(agent_1_cam))
    # agent_0_prepix = _3d_to_2d(matrix = agent_0_cam,point_3d = agent_0_objpos)
    agent_0_prepix = _3d_to_2d(matrix = agent_0_cam,point_3d = agent_0_pre_worldloc[:3])
    # agent_0_prepix = _project_points_to_image(points_3d =agent_0_eeglobal,camera_info=camera_info,cam_trans=agent_0_cam)
    # agent_1_prepix = _project_points_to_image(points_3d =agent_1_eeglobal, camera_info=camera_info,cam_trans=agent_1_cam) 
    #3D投影到2D,想要更改投影点为目标物体的话改points_3d为agent_0_objpos
    
    result = {
        "step":i+1,
        # "agent_0_camera_point":agent_0_prepix.tolist(),
        "agent_1_camera_point":agent_0_prepix.tolist()
        
    }
    data_trans.append(result)
    point = agent_0_prepix
    
    step_name = i+1
    picture_name = ("frame_"+str(step_name)+"_agent_0_head_rgbFetchRobot_head_rgb.png")
    image = Image.open(os.path.join(data_path,picture_name))
    image = image.resize((256,256))
    image = image.convert('RGB')
    rgb_matrix = np.array(image)
    bgr = cv2.cvtColor(rgb_matrix,cv2.COLOR_RGB2BGR)

    cv2.circle(bgr, point,2,(0, 0, 255),-1)
    if agent_0_target_!= [[-1,-1,-1,-1]]:
            x,y,w,h = list(map(lambda t: int((t*256)/1024),agent_0_target_[0]))
            
            cv2.rectangle(bgr,(x,y),(x+w,y+h),(255,0,0),1)
    output_image_path = os.path.join(data_path,'point')
    if not os.path.exists(output_image_path):
                    os.makedirs(output_image_path)
    output_image_name = os.path.join(output_image_path,"frame_"+str(step_name)+".png")
    cv2.imwrite(output_image_name, bgr)
    # except:
    #      print("skip")
    

with open(os.path.join(data_path,'data_trans.json'), 'w') as file:
    json.dump(data_trans, file, indent=2)

