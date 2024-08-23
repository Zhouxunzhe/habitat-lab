import json
import numpy as np
import cv2
import os
import magnum

def process_directory(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            json_path = os.path.join(dir_path, 'sum_data.json')
            
            if os.path.exists(json_path) and dir_name != "episode_0":
                with open(json_path, 'r') as file:
                    data = json.load(file)

                camera_extrinsic_old = np.array([
                    [1, 0, 0, 0.25],
                    [0, 1, 0, 1.2],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                camera_extrinsic = magnum.Matrix4(
                    magnum.Vector4(*camera_extrinsic_old[0]),
                    magnum.Vector4(*camera_extrinsic_old[1]),
                    magnum.Vector4(*camera_extrinsic_old[2]),
                    magnum.Vector4(*camera_extrinsic_old[3]),
                )
                camera_info = {
                    'width': 256,
                    'height': 256,
                    'K': [128.0, 0, 128.0, 0, 128.0, 128.0, 0, 0, 1],
                    'max_depth': 10.0,
                    'min_depth': 0.0,
                    'normalize_depth': False
                }
                camera_in = np.array([
                    [128.0, 0, 128.0],
                    [0, 128.0, 128.0],
                    [0, 0, 1]
                ])
                result = []
                data_trans = []
                skip_len = 20

                def trans_worldloc_to_robotloc(trans, loc):
                    trans = np.linalg.inv(trans)
                    locs = np.append(loc, 1)
                    robotloc_pos = np.dot(trans, locs)
                    return np.array(robotloc_pos)[:3]

                def _project_points_to_image(points_3d, point_3d_match, camera_info, cam_trans):
                    fx = camera_info['K'][0]
                    fy = camera_info['K'][4]
                    cx = camera_info['K'][2]
                    cy = camera_info['K'][5]
                    points_3d_hom = np.append(points_3d, 1)
                    cam_trans = np.asarray(cam_trans)
                    points_3d_transformed = np.dot(cam_trans, points_3d_hom)
                    x = points_3d_transformed[0]
                    y = points_3d_transformed[1]
                    z = points_3d_transformed[2]
                    if z == 0:
                        return np.array([0, 0])
                    else:
                        u = (x * fx / z) + cx
                        v = (y * fy / z) + cy
                    pixel_coords = np.array([u, v])
                    return pixel_coords[0], pixel_coords[1]

                action = ["nav_to_point", "pick", "nav_to_point", "none"]
                agent_0_action = 0
                agent_1_action = 0
                for i in range(0, len(data['entities']) - skip_len):
                    step_data = data['entities'][i]
                    next_step_data = data['entities'][i + skip_len]
                    agent_0_trans_matrix = step_data['data']['agent_0_robot_trans_martix']
                    agent_0_nowloc = step_data['data']['agent_0_localization_sensor']
                    agent_1_nowloc = step_data['data']['agent_1_localization_sensor']
                    agent_1_trans_matrix = step_data['data']['agent_1_robot_trans_martix']
                    agent_0_eeglobal = step_data['data']['agent_0_ee_global_pos_sensor']
                    agent_1_eeglobal = step_data['data']['agent_1_ee_global_pos_sensor']
                    agent_0_pre_worldloc = next_step_data['data']['agent_0_localization_sensor']
                    agent_1_pre_worldloc = next_step_data['data']['agent_1_localization_sensor']
                    agent_0_objpos = step_data['data']['agent_0_obj_pos']
                    agent_1_objpos = step_data['data']['agent_1_obj_pos']

                    # agent_0_pre_eepos = trans_worldloc_to_robotloc(agent_0_trans_matrix, agent_0_eeglobal)
                    # agent_1_pre_eepos = trans_worldloc_to_robotloc(agent_0_trans_matrix, agent_1_eeglobal)
                    agent_0_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_pre_worldloc[:3])
                    agent_1_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_pre_worldloc[:3])
                    agent_0_obj_ro = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_objpos[:3])
                    agent_1_obj_ro = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_objpos[:3])
                    agent_0_prepix = _project_points_to_image(points_3d=agent_0_obj_ro, point_3d_match=agent_0_pre_robotloc[:3], camera_info=camera_info, cam_trans=camera_extrinsic_old)
                    agent_1_prepix = _project_points_to_image(points_3d=agent_1_obj_ro, point_3d_match=agent_1_pre_robotloc[:3], camera_info=camera_info, cam_trans=camera_extrinsic_old)

                    if (agent_0_action == 0 or agent_0_action == 2) and step_data['data']['agent_0_has_finished_oracle_nav'] == [1.0]:
                        agent_0_action += 1
                    if agent_0_action == 1 and data['entities'][i]['data']['agent_0_localization_sensor'] != data['entities'][i-1]['data']['agent_0_localization_sensor']:
                        agent_0_action += 1
                    if (agent_1_action == 0 or agent_1_action == 2) and step_data['data']['agent_1_has_finished_oracle_nav'] == [1.0]:
                        agent_1_action += 1
                    if agent_1_action == 1 and data['entities'][i]['data']['agent_1_localization_sensor'] != data['entities'][i-1]['data']['agent_1_localization_sensor']:
                        agent_1_action += 1

                    result = {
                        "step": i + 1,
                        "agent_0_nowloc": agent_0_nowloc,
                        "agent_0_pre_worldloc":agent_0_pre_worldloc,
                        "agent_1_pre_worldloc":agent_1_pre_worldloc,
                        "agent_0_pre_robotloc": agent_0_pre_robotloc.tolist(),
                        "agent_0_trans_matrix": agent_0_trans_matrix,
                        "agent_1_trans_matrix": agent_1_trans_matrix,
                        "agent_1_nowloc": agent_1_nowloc,
                        "agent_1_pre_robotloc": agent_1_pre_robotloc.tolist(),
                        "agent_0_obj_ro": agent_0_obj_ro.tolist(),
                        "agent_1_obj_ro": agent_1_obj_ro.tolist(),
                        "agent_0_action": action[agent_0_action],
                        "agent_1_action": action[agent_1_action]
                    }
                    data_trans.append(result)
                
                data_dir_path = os.path.join(dir_path, 'data')
                if not os.path.exists(data_dir_path):
                    os.makedirs(data_dir_path)
                output_json_path = os.path.join(data_dir_path, 'data_trans.json')
                # output_json_path = os.path.join(dir_path, 'data_trans.json')
                with open(output_json_path, 'w') as file:
                    json.dump(data_trans, file, indent=2)

base_directory = './video_dir/'
# gz_name = f"manipulation_eval_process_0.json.gz"
num_gz = 250
# for i in range(0,num_gz):
#     process_directory(os.path.join(base_directory,f"process_{i}.json.gz"))
process_directory('./video_dir/image_dir')