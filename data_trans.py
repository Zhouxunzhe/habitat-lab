import json
import numpy as np
import cv2
import os
import magnum
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
camera_info={
    'width': 256, 
    'height': 256, 
    'K': [128.00000000000003, 0, 128.0, 0, 128.00000000000003, 128.0, 0, 0, 1], 
    'max_depth': 10.0, 
    'min_depth': 0.0, 
    'normalize_depth': False
} 
data_path = './video_dir_point/image_dir/episode_91'
data_trans = []
skip_len = 20
with open(os.path.join(data_path,'sum_data.json'), 'r') as file:
    data = json.load(file)

for i in range(0,len(data['entities'])-skip_len):
    step_data = data['entities'][i]
    next_step_data = data['entities'][i + skip_len] #取当前step的20步后的info，用于预测导航目标信息

    agent_0_trans_matrix = step_data['data']['agent_0_robot_trans_martix']
    agent_1_trans_matrix = step_data['data']['agent_1_robot_trans_martix'] #获取agent当前step的转换矩阵信息
    # agent_0_nowloc = step_data['data']['agent_0_localization_sensor']
    # agent_1_nowloc = step_data['data']['agent_1_localization_sensor']
    # agent_0_pre_worldloc = next_step_data['data']['agent_0_localization_sensor']
    # agent_1_pre_worldloc = next_step_data['data']['agent_1_localization_sensor']

    # agent_0_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_pre_worldloc[:3])
    # agent_1_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_pre_worldloc[:3])
    agent_0_objpos = step_data['data']['agent_0_obj_pos']
    agent_1_objpos = step_data['data']['agent_1_obj_pos'] #获取当前目标物体在世界坐标系下的坐标
    agent_0_obj_ro = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_objpos[:3])
    agent_1_obj_ro = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_objpos[:3]) #转移到机器人坐标系下
    
    agent_0_eeglobal = step_data['data']['agent_0_ee_global_pos_sensor']
    agent_1_eeglobal = step_data['data']['agent_1_ee_global_pos_sensor']
    agent_0_eepos = step_data['data']['agent_0_ee_pos']
    agent_1_eepos = step_data['data']['agent_1_ee_pos'] #末端执行器在机器人坐标系下的坐标
    agent_0_cam = step_data['data']['agent_0_camera_extrinsic']
    agent_1_cam = step_data['data']['agent_1_camera_extrinsic'] #当前step的camera_extrinsic

    agent_0_prepix = _project_points_to_image(points_3d =agent_0_eeglobal,camera_info=camera_info,cam_trans=agent_0_cam)
    agent_1_prepix = _project_points_to_image(points_3d =agent_1_eeglobal, camera_info=camera_info,cam_trans=agent_1_cam) 
    #3D投影到2D,想要更改投影点为目标物体的话改points_3d为agent_0_objpos

    result = {
        "step":i+1,
        "agent_0_camera_point":agent_0_prepix,
        "agent_1_camera_point":agent_1_prepix
        
    }
    data_trans.append(result)
    point = agent_0_prepix
    step_name = i+1
    picture_name = ("frame_"+str(step_name)+"_agent_0_head_rgbFetchRobot_head_rgb.png")
    image = cv2.imread(os.path.join(data_path, picture_name))
    try:
        cv2.circle(image, (int(point[0]), int(point[1])),2,(0, 255, 0), -1)
        output_image_path = os.path.join(data_path,'point')
        output_image_name = os.path.join(output_image_path,"frame_"+str(step_name)+".png")
        cv2.imwrite(output_image_name, image)
    except:
         print("skip")
    

with open(os.path.join(data_path,'data_trans.json'), 'w') as file:
    json.dump(data_trans, file, indent=2)

