import json
import numpy as np
import cv2
import os
import magnum
with open('./video_dir/image_dir/episode_138/sum_data.json', 'r') as file:
    data = json.load(file)
camera_extrinsic_old = np.array([[1, 0, 0, 0.25],
       [0, 1, 0, 1.2],
       [0, 0, 1, 0],
       [0, 0, 0, 1]])
camera_extrinsic = magnum.Matrix4(
        magnum.Vector4(*camera_extrinsic_old[0]),
        magnum.Vector4(*camera_extrinsic_old[1]),
        magnum.Vector4(*camera_extrinsic_old[2]),
        magnum.Vector4(*camera_extrinsic_old[3]),
    )
camera_info={
    'width': 256, 
    'height': 256, 
    'K': [128.00000000000003, 0, 128.0, 0, 128.00000000000003, 128.0, 0, 0, 1], 
    'max_depth': 10.0, 
    'min_depth': 0.0, 
    'normalize_depth': False
} 
camera_in = np.array([[128.00000000000003, 0, 128.0],
            [0, 128.00000000000003, 128.0],
            [0, 0, 1]]              
)
result = []
def trans_worldloc_to_robotloc(trans,loc):
    trans = np.linalg.inv(trans)
    # tran_ = magnum.Matrix4(
    #     magnum.Vector4(*trans[0]),
    #     magnum.Vector4(*trans[1]),
    #     magnum.Vector4(*trans[2]),
    #     magnum.Vector4(*trans[3]),
    # )
    locs= np.append(loc,1)
    robotloc_pos = np.dot(trans,locs)
    return np.array(robotloc_pos)[:3]
data_trans = []
skip_len = 20

def _project_points_to_image(points_3d,point_3d_match ,camera_info, cam_trans):
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
        z = points_3d_transformed[2]
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
# datas = data['entities'][2]['data']['agent_0_localization_sensor']
# print(datas)
action = ["nav_to_point","pick","nav_to_point","none"]
agent_0_action = 0
agent_1_action = 0
for i in range(0,len(data['entities'])-skip_len):
    step_data = data['entities'][i]
    next_step_data = data['entities'][i + skip_len]
    # print("i+skip_len",i+skip_len)
    # print("now_data",step_data)
    # print("match_data:",next_step_data)
    agent_0_trans_matrix = step_data['data']['agent_0_robot_trans_martix']
    agent_0_nowloc = step_data['data']['agent_0_localization_sensor']
    agent_1_nowloc = step_data['data']['agent_1_localization_sensor']
    agent_1_trans_matrix = step_data['data']['agent_1_robot_trans_martix']
    agent_0_eeglobal = step_data['data']['agent_0_ee_global_pos_sensor']
    agent_1_eeglobal = step_data['data']['agent_1_ee_global_pos_sensor']
    agent_0_eepos = step_data['data']['agent_0_ee_pos']
    agent_1_eepos = step_data['data']['agent_1_ee_pos']
    agent_0_pre_worldloc = next_step_data['data']['agent_0_localization_sensor']
    agent_1_pre_worldloc = next_step_data['data']['agent_1_localization_sensor']
    agent_0_objpos = step_data['data']['agent_0_obj_pos']
    agent_1_objpos = step_data['data']['agent_1_obj_pos']
    # print("agent_0_pre_worldloc",agent_0_pre_worldloc)
    # print("agent_1_pre_worldloc",agent_1_pre_worldloc)
    agent_0_pre_eepos= trans_worldloc_to_robotloc(agent_0_trans_matrix, agent_0_eeglobal)
    agent_1_pre_eepos= trans_worldloc_to_robotloc(agent_0_trans_matrix, agent_1_eeglobal)
    agent_0_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_pre_worldloc[:3])
    agent_1_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_pre_worldloc[:3])
    agent_0_obj_ro = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_objpos[:3])
    agent_1_obj_ro = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_objpos[:3])
    agent_0_prepix = _project_points_to_image(points_3d =agent_0_pre_robotloc,point_3d_match=agent_0_pre_robotloc[:3], camera_info=camera_info,cam_trans=camera_extrinsic_old)
    agent_1_prepix = _project_points_to_image(points_3d =agent_1_eepos,point_3d_match=agent_1_pre_robotloc[:3], camera_info=camera_info,cam_trans=camera_extrinsic_old)
    if step_data['data']['agent_0_has_finished_oracle_nav']==[1.0]:

        print(step_data['data']['agent_0_has_finished_oracle_nav']) 
    #判断是什么动作
    if (agent_0_action == 0 or agent_0_action == 2)and step_data['data']['agent_0_has_finished_oracle_nav']== [1.0]:
        agent_0_action+=1
    if agent_0_action == 1 and data['entities'][i]['data']['agent_0_localization_sensor']!= data['entities'][i-1]['data']['agent_0_localization_sensor']:
        agent_0_action+=1
    if (agent_1_action == 0 or agent_1_action == 2)and step_data['data']['agent_1_has_finished_oracle_nav']== [1.0]:
        agent_1_action+=1
    if agent_1_action == 1 and data['entities'][i]['data']['agent_1_localization_sensor']!= data['entities'][i-1]['data']['agent_1_localization_sensor']:
        agent_1_action+=1

    result = {
        "step":i+1,
        # "agent_0_nowloc": agent_0_nowloc,
        # "agent_0_pre_robotloc":agent_0_pre_robotloc.tolist(),
        "agent_1_nowloc": agent_1_nowloc,
        "agent_1_pre_robotloc":agent_1_pre_robotloc.tolist(),
        # "agent_0_obj_ro":agent_0_obj_ro.tolist(),
        "agent_1_obj_position(robot_xyz)":agent_1_obj_ro.tolist(),
        "agent_1_ee_position(robot_xyz)":agent_1_eepos,
        "agent_0_camera_point":agent_1_prepix
        # "agent_0_action": action[agent_0_action],
        # "agent_1_action": action[agent_1_action]
    }
    # print("flag")
    data_trans.append(result)
    point = agent_1_prepix
    step_name = i+1
    picture_name = ("frame_"+str(step_name)+"_agent_1_head_rgbFetchRobot_head_rgb.png")
    image = cv2.imread(os.path.join('./video_dir/image_dir/episode_24/',picture_name))
    try:
        cv2.circle(image, (int(point[0]), int(point[1])),2,(0, 255, 0), -1)
        output_image_path = './video_dir/image_dir/episode_24/data'
        output_image_name = os.path.join(output_image_path,"frame_"+str(step_name)+".png")
        cv2.imwrite(output_image_name, image)
    except:
         print("skip")
    

with open('./video_dir/image_dir/episode_24/data/data_trans.json', 'w') as file:
    json.dump(data_trans, file, indent=2)

