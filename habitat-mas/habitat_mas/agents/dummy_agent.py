import matplotlib.pyplot as plt
import os
import numpy as np
import json
import re,shutil
import base64
import requests
import time
from PIL import Image
from io import BytesIO
import os
from torchvision import transforms
import torch


# from habitat_mas.agents.llm_agent_base import LLMAgentBase
class DummyAgent:
    def __init__(self, **kwargs):
        self.agent_name = kwargs.get("agent_name", "agent_0")
        self.pddl_problem = kwargs.get("pddl_problem", None)
        # self.all_entities = self.pddl_problem.all_entities
        self.initilized = True
        self.return_num = -1
        self.step_num = 0
        self.episode_id = -1
        self.url = "http://0.0.0.0:10077/robot-chat"
        self.have_finished_action = []
        self.have_reset_arm = False
        self.have_wait = True
        self.trial_time = 0
        self.rag_truly_output = False
        self.rag_truly_output_point = None
        self.oracle_set = True
        

    # def send_and_receive(self, image_list, episode_id):
        
    def init_agent(self, **kwargs):
        return 
    def get_image_paths(self,root_dir):
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def images_to_tensors(self,image_paths):
        tensors = []
        names = []
        transform = transforms.Compose([
            transforms.Resize((448, 448)),  
            transforms.ToTensor()
        ])
        for path in image_paths:
            image = Image.open(path).convert('RGB') 
            tensor = transform(image)
            tensors.append(tensor.squeeze(0))
            names.append(os.path.basename(path))
        return tensors,names
    def send_and_receive(self,image_list,episode_id,prompt_text,is_rag = False):
        images = []
        headers = {
            "Content-Type": "application/json"
        }
        for image in image_list:
            image = image.numpy()
            # print("image_in_send_beforesqueeze:",image.shape)
            if is_rag:
                image = (image.transpose(1, 2, 0))*255
            image = image.astype(np.uint8)
            # print("image_in_send:",image)
            image_PIL = Image.fromarray(image)
            buffered = BytesIO()
            image_PIL.save(buffered,format = 'PNG')
            images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        data = {
            "id": f"{episode_id}_{self.step_num}",
            "width": [256, 256],
            "height": [256, 256],
            "image": images,
            "prompt": prompt_text
                # f"describe what is in the picture."
                # f"You are an AI visual assistant that can manage a single robot. You receive the robot's task, one image representing the robot's current view and what the robot has completed so far. You need to output the robot's next action. Actions the robot can perform are \"nav_to_point\", \"turn\", \"pick\" and \"place\". If you cannot determine the next action based on the robot's current view, you can command the robot to \"turn\". Your output format should be either \"turn\" or \"action_name<box>[[x1, y1, x2, y2]]</box>\".Robot's Task: The robot need to navigate to the {target_name} where the box is located, pick it up,navigate to the {goal_name} and place the box.\n    Robot's current view: <image>\n    ."
                # f"You are an AI visual assistant that can manage a single robot. You receive the robot's task, one image representing the robot's current view and what the robot has completed so far. You need to output the robot's next action. Actions the robot can perform are \"nav_to_point\", \"turn\", \"pick\" and \"place\". If you cannot determine the next action based on the robot's current view, you can command the robot to \"turn\". Your output format should be either \"turn\" or \"action_name<box>[[x1, y1, x2, y2]]</box>\".Robot's Task: The robot need to navigate to the {target_name} where the box is located, pick it up,navigate to the {goal_name} and place the box.\n    Robot's current view: <image>\n    .The robot has finished: navigate to the kitchen,pick the box,navigate to the chair."
##记得改prompt

# prompt: aaa <image> <image>
# image: [base64(image1), base64(image2)]
        }
        while True:
            response = requests.post(self.url, headers=headers, json=data)
            if response.status_code >= 200 and response.status_code < 300:
                return_json = response.json()
                try:
                    response_json = response.json()
                    return response_json
                except:
                    print("invalid json")
            else:
                print("Wrong status_code")
            time.sleep(0.2)
    def find_rec_name(self,episode_id,scene_name_path,data_path,scene_dir):
        data_json_path = data_path+'.json'
        with open(data_json_path, 'r') as f:
            data = json.load(f)
        episodes_info = []
        for episode in data.get('episodes', []):
            episode_info = {
                'episode_id': episode.get('episode_id'),
                'target_receptacles': episode.get('target_receptacles'),
                'goal_receptacles': episode.get('goal_receptacles'),
                'target_object': episode.get('name_to_receptacle',{})
            }
            episodes_info.append(episode_info)
        matched_episode = None
        for episode in episodes_info:
            if int(episode['episode_id']) == episode_id:
                matched_episode = episode
                break
        goal_item = matched_episode['goal_receptacles'][0][0].split('_')[0]
        target_item = matched_episode['target_receptacles'][0][0].split('_')[0]
        obj_item = ""
        for key in matched_episode['target_object'].keys():
            key_replace_front = re.sub(r'^\d+_', '', key)
            key_replace_back = re.sub(r'[:_]\d+$', '', key_replace_front)
            obj_item = key_replace_back.replace('_', ' ')
        with open(scene_name_path, 'r') as f:
            scene_name_data = json.load(f)
        result_find = []
        for sc_name in scene_name_data:
            if goal_item == sc_name["template_name"]:
                goal_name = sc_name["name"]
                break
        for sc_name in scene_name_data:
            if target_item == sc_name["template_name"]:
                target_name = sc_name["name"]
                break
        return target_name, goal_name, obj_item
    def _2d_to_3d_single_point(self, depth_obs, depth_rot,depth_trans,pixel_x, pixel_y):
        # depth_camera = self._sim._sensors[depth_name]._sensor_object.render_camera

        # hfov = float(self._sim._sensors[depth_name]._sensor_object.hfov) * np.pi / 180.
        # W, H = depth_camera.viewport[0], depth_camera.viewport[1]
        W = 256
        H = 256
        hfov = 1.5707963267948966
        # Intrinsic matrix K
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]
        ])
        
        # Normalize pixel coordinates
        xs = (2.0 * pixel_x / (W - 1)) - 1.0  # normalized x [-1, 1]
        ys = 1.0 - (2.0 * pixel_y / (H - 1))  # normalized y [1, -1]

        # Depth value at the pixel
        depth = depth_obs[0, pixel_x,pixel_y,0]

        # Create the homogeneous coordinates for the pixel in camera space
        xys = np.array([xs * depth, ys * depth, -depth, 1.0]).reshape(4, 1)
        
        # Apply the inverse of the intrinsic matrix to get camera space coordinates
        xy_c = np.matmul(np.linalg.inv(K), xys)

        # Get the rotation and translation of the camera
        depth_rotation = np.array(depth_rot)
        depth_translation = np.array(depth_trans)

        # Get camera-to-world transformation
        T_world_camera = np.eye(4)
        T_world_camera[0:3, 0:3] = depth_rotation
        T_world_camera[0:3, 3] = depth_translation

        # Apply transformation to get world coordinates
        T_camera_world = np.linalg.inv(T_world_camera)
        points_world = np.matmul(T_camera_world, xy_c)

        # Get non-homogeneous points in world space
        points_world = points_world[:3, :] / points_world[3, :]
        return points_world.flatten()
    def seen_process(self,vlm_output,depth_info,depth_rot,depth_trans):
        depth_info = depth_info
        depth_rot = depth_rot
        depth_trans = depth_trans
        is_not_match_training = True
        match = re.search(r'(\w+)\s*\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]', vlm_output)
        if match and False:
            func_name = match.group(1)  # 提取函数名
            x = int(match.group(2))      # 提取 x
            y = int(match.group(3))      # 提取 y
            w = int(match.group(4))      # 提取 w
            h = int(match.group(5))
        elif is_not_match_training:
            if "nav_to_point" in vlm_output:
                func_name = "nav_to_point"
            elif "pick" in vlm_output:
                func_name = "pick"
            else:
                func_name = "place"
            numbers = re.findall(r'\d+', vlm_output)
            numbers = [int(num) for num in numbers]
            x,y,w,h = numbers
            print(f"x:{x},y:{y},w:{w},h:{h},func:{func_name}")
            point_2d = (int((x+(w/2))*256/1000),int((y+(h/2))*256/1000))
            # print("point2d:",point_2d,flush = True)
            # point2d = (128,200)
            point_3d = self._2d_to_3d_single_point(depth_info,depth_rot, depth_trans,point_2d[0],point_2d[1])
            print("point3d:",point_3d,flush = True)
            if func_name == "nav_to_point":
                if point_3d[1]<0:
                    point_3d[1] = 0
                return {
                    "name": "nav_to_position",
                    "arguments": {
                        "target_position": point_3d.tolist()[:3],
                        "robot": self.agent_name,
                    }
                }
            if func_name == "pick":
                self.have_finished_action.append("pick")
                return {
                    "name": "pick",
                    "arguments": ['any_targets|0', 'agent_0']
                }
            if func_name == "place":
                return {
                    "name": "place",
                    "arguments": ['any_targets|0','TARGET_any_targets|0', 'agent_0']
                }
        else:
            return {
                "name": "wait",
                "arguments": ['1','agent_0']
            }
    def get_localization_sensor(self,data, episode_id, index):
        for item in data:
            if item['episode_id'] == str(episode_id):
                for file in item['obs_files']:
                    if f"head_rgb_{str(index)}" in file:
                        print(f"choose_rag_point:head_rgb_{str(index)}/episode_id:{episode_id}")
                        return item['localization_sensor'][:4]
        return None
    def action_check_and_get(self,observations,eval_info):
        if not self.have_reset_arm:
            self.have_reset_arm = True
            return {
                "name": "reset_arm",
                "arguments": ['agent_0']
            }
        if not self.have_wait:
            self.have_wait = True
            return {
                "name": "wait",
                "arguments": ['1','agent_0']
            }
        if self.oracle_set:
            self.oracle_set = False
            return {
                    "name": "nav_to_position",
                    "arguments": {
                        "target_position": nav_point_new,
                        "robot": self.agent_name,
                    }
                }
        
        action_dict = self.get_vlm_policy_info(observations,eval_info)
        self.have_wait = False
        return action_dict
    def get_vlm_policy_info(self,observations,eval_info):
        debug_seen = True
        episode_id,data_path = eval_info
        if self.episode_id != episode_id:
            self.episode_id = episode_id
            self.trial_time = 0
        if(self.agent_name == "agent_1"):
            return {
                "name": "wait",
                "arguments": ['3000','agent_1']
            }
        head_rgb = observations["head_rgb"].cpu()
        depth_info = observations["depth_inf"].cpu()
        depth_rot = observations["depth_rot"].cpu()
        depth_trans = observations["depth_trans"].cpu()
        print("depth_infoshape:",depth_info.shape)
        print("depth_rotshape:",depth_rot)
        print("depth_transshape:",depth_trans)
        # print("depth_rot:",depth_rot)
        # print("depth_trans:",depth_trans)
        # print("head_rgb:",head_rgb)
        print("observations[""]",observations["rec_bounding_box"])
        _,_,w_rec,h_rec = observations["rec_bounding_box"].cpu()[0][0]
        _,_,w_tag,h_tag = observations["target_bounding_box"].cpu()[0][0]
        print(f"w_rec:{w_rec},h_rec:{h_rec}")
        scene_dir = 'data/datasets/test_scene_graph/scene_dir/108736689_177263340.json'
        target_name, goal_name, obj_item = self.find_rec_name(episode_id,scene_dir,data_path,scene_dir)
        print(f"target_name:{target_name},goal_name:{goal_name},obj_item:{obj_item}")
        seen_divide = 0
        if "pick" in self.have_finished_action:
            seen_divide = int(w_tag)*int(h_tag)
        else:
            
            seen_divide = int(w_rec)*int(h_rec)
        prompt = ""
        images_list = []
        scene_graph_path = f'data/scene_graph/108736689_177263340/data_8/episode_{episode_id}'
        print("seen_divide:",seen_divide)
        if debug_seen:
            min_seen = 0
        if seen_divide > min_seen:
            prompt = f"""You are an AI visual assistant that can manage a single robot.
            You receive the robot's task and one image representing the robot's current view.
            You need to output the robot's next action. Actions the robot can perform are \"nav_to_point\", \"pick\" and \"place\".
            Your output format should be \"action_name<box>[[x1, y1, x2, y2]]</box>\".
            Robot's Task: The robot need to navigate to the {target_name} where the {obj_item} is located,
            pick the {obj_item}up,navigate to the {goal_name} and place the {obj_item}.Robot's current view: <image>"""
            images_list = [head_rgb.squeeze(0)]
            vlm_output = self.send_and_receive(images_list,episode_id,prompt)
            print("vlm_output:",vlm_output)
            return self.seen_process(vlm_output,depth_info,depth_rot,depth_trans)
        else:
            if self.rag_truly_output:
                self.rag_truly_output = False
                nav_point_new = self.rag_truly_output_point
                nav_point_new[4] = 0
                return {
                    "name": "nav_to_position",
                    "arguments": {
                        "target_position": nav_point_new,
                        "robot": self.agent_name,
                    }
                }
            rag_target = goal_name if "pick" in self.have_finished_action else target_name
            prompt = f"""\nYou are a robot operating in a scene and your task is to respond to the user 
            command about going to a specific location by finding the closest frame in the tour video to navigate to. \n
            These frames are from the robot's tour of the scene:\nFrame1: <image>\nFrame2: <image>\nFrame3: <image>\nFrame4: <image>\nFrame5: <image>\nFrame6: <image>\nFrame7: <image>\nFrame8: <image>\nFrame9: <image>\nFrame10: <image>\nFrame11: <image>\nFrame12: <image>\n
            Now the robot task is:navigate to the {rag_target}.\nPlease identify and find the frame that the robot should navigate to complete the task. """
            image_paths = self.get_image_paths(scene_graph_path)
            print("image_paths:",image_paths)
            images_list,names_list = self.images_to_tensors(image_paths)
            scene_json_path = 'data/scene_graph/108736689_177263340/data_8/metadata.json'
            image_index_test = 1 if "pick" in self.have_finished_action else 0
            if not debug_seen:
            # print("images_list:",images_list.shape)
                vlm_output = self.send_and_receive(images_list,episode_id,prompt,is_rag = True)
                match = re.search(r'"image_index": (\d+)', vlm_output)
                if match:
                    image_index = int(match.group(1)) - 1
                    print("output_index:",image_index)
            with open(scene_json_path, 'r') as f:
                data_index = json.load(f)
            nav_position = self.get_localization_sensor(data_index,episode_id,image_index_test)
            self.trial_time +=1
            if self.trial_time > 4:
                return {
                    "name": "wait",
                    "arguments": ['3000','agent_0']
                }
            self.rag_truly_output = True
            nav_position.append(1)
            self.rag_truly_output_point = nav_position
            return {
                "name": "nav_to_position",
                "arguments": {
                    "target_position": nav_position,
                    "robot": self.agent_name,
                }
            }
        return None
    def agent_output(self,skill_name_to_idx,observations,eval_info,**kwargs):
        # print("kwargs",kwargs)
        episode_id,data_path = eval_info
        print(f"_______________________{self.agent_name}")
        for key in observations.keys():
            print(key)
        info = [
            # {
            #     "name": "nav_to_position",
            #     "arguments": {
            #         "target_position": [-6.0114569664001465,0.1160174161195755,0.8003081679344177,1.4639922380447388,1],
            #         "robot": self.agent_name,
            #     }
            # },
            #!!!!!!!!!if the next_loc_z < now_loc_z, then it is 0
            {
                "name": "reset_arm",
                "arguments": ['agent_0']
            }, #非常tmd重要！！！！！

            # {
            #     "name": "nav_to_position",
            #     "arguments": {
            #         "target_position": [-5.921825885772705,0.17830851674079895,2.79310941696167,-0.8817002773284912,0],
            #         "robot": self.agent_name,
            #     }
            # },
            # {
            #     "name": "wait",
            #     "arguments": ['1','agent_0']
            # },
            {
                "name": "nav_to_position",
                "arguments": {
                    "target_position": [-4.954258918762207,0.17830851674079895,3.402430534362793],
                    "robot": self.agent_name,
                }
            },
            #after finish nav,you should force the robot wait for 1sc,so that it could do the next action.
            {
                "name": "wait",
                "arguments": ['1','agent_0']
            },
            {
                "name": "pick",
                "arguments": ['any_targets|0', 'agent_0']
            },
            {
                "name": "reset_arm",
                "arguments": ['agent_0']
            },
            {
                "name": "nav_to_position",
                "arguments": {
                    "target_position": [-2.1182525157928467,0.1160174161195755,-0.19527381658554077,-0.7443385124206543,1],
                    "robot": self.agent_name,
                }
            },
            # {
            #     "name": "wait",
            #     "arguments": ['1','agent_0']
            # },
            # {
            #     "name": "wait",
            #     "arguments": ['3000','agent_0']
            # },
            {
                "name": "wait",
                "arguments": ['1','agent_0']
            },
            {
                "name": "nav_to_position",
                "arguments": {
                    "target_position":  [-6.866964340209961,0.17830851674079895,1.3594081401824951,0],
                    "robot": self.agent_name,
                }
            },
            {
                "name": "wait",
                "arguments": ['1','agent_0']
            },
            {
                "name": "nav_to_position",
                "arguments": {
                    "target_position":  [-7.28494930267334,0.17830851674079895,0.008822225034236908],
                    "robot": self.agent_name,
                }
            },
            {
                "name": "wait",
                "arguments": ['1','agent_0']
            },
            {
                "name": "nav_to_position",
                "arguments": {
                    "target_position": [-7.70682430267334,0.17830851674079895,-4.249903678894043,0],
                    "robot": self.agent_name,
                }
            },
            {
                "name": "wait",
                "arguments": ['1','agent_0']
            },
            {
                "name": "nav_to_position",
                "arguments": {
                    "target_position":  [-7.229156970977783,0.17830851674079895,-4.4799323081970215],
                    "robot": self.agent_name,
                }
            },
            {
                "name": "wait",
                "arguments": ['3000','agent_0']
            },

            {
                "name": "place",
                "arguments": ['any_targets|0','TARGET_any_targets|0', 'agent_0']
            },
            {
                "name": "wait",
                "arguments": ['3000','agent_0']
            },

        ]
        # print("self.agent_name:",self.agent_name)
        if self.agent_name == "agent_0":
            self.return_num += 1
        else:
            return {
                "name": "nav_to_position",
                "arguments": {
                    "target_position": [0,0,0,0],
                    "robot": self.agent_name,
                }
            }
        print("self.return_num:",self.return_num)
        # target_position = [[-3.7735161781311035,0.1160174161195755,-1.5337603092193604],[-2.0869526863098145,
        #   0.1160174161195755,-0.19013549387454987],[-1.985478401184082,0.1160174161195755,3.3729171752929688],
        #   [-3.7735161781311035,0.1160174161195755,-1.5337603092193604],[-1.985478401184082,0.1160174161195755,3.3729171752929688]]
        return info[self.return_num]

if __name__ == "__main__":
    vlm_test = DummyAgent()
    image_paths = ['data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_10.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_11.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_1.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_9.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_4.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_5.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_2.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_7.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_0.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_3.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_6.png', 'data/scene_graph/103997970_171031287/data_8/episode_2/episode_2_head_rgb_8.png']
    images_list,names_list = vlm_test.images_to_tensors(image_paths)
    print("images_list:",images_list)
    # vlm_output = self.send_and_receive(images_list,episode_id,prompt,is_rag = True)
    for image in images_list:
        image = image.numpy()
        print("image_in_send_beforesqueeze:",image.shape)
        if True:
            image = (image.transpose(1, 2, 0))*255
        image = image.astype(np.uint8)
        print("image_in_send:",image)
        image_PIL = Image.fromarray(image)
        buffered = BytesIO()
        image_PIL.save(buffered,format = 'PNG')
        images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))