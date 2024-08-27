import matplotlib.pyplot as plt
import os
import numpy as np
import json
import re
import base64
import requests
import time
from PIL import Image
from io import BytesIO
class VLMAgent():
    def __init__(self,agent_num,image_dir,url,json_dir=None) -> None:
        self.image_dir = image_dir
        self.step_num = 0
        self.json_dir = json_dir
        self.url = url
    def vlm_inference(self,image_dir):
        output = {}
        return output
    def test_inference(self,json_dir):
        with open(json_dir,'r') as file:
            data = json.load(file)
        step = self.step_num
        return data[step]
    def pos_trans(self,trans,pos):
        poss = np.append(pos,1)
        coord_pos = np.dot(trans, poss)
        return np.array(coord_pos)[:3]
    def send_and_receive(self,image_list,episode_id):
        images = []
        headers = {
            "Content-Type": "application/json"
        }
        for image in image_list:
            image = image.squeeze(0).numpy().astype(np.uint8)
            image_PIL = Image.fromarray(image)
            buffered = BytesIO()
            image_PIL.save(buffered,format = 'PNG')
            images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        data = {
            "id": f"{episode_id}_{self.step_num}",
            "width": [256, 256],
            "height": [256, 256],
            "image": images,
            "prompt": (
                "You are now managing two robots. The actions they can perform are "
                "\"nav_to_point,\" \"pick,\" \"continue\" and \"place.\" They need to locate "
                "all the boxes in the scene, pick them up, and place them on the table next "
                "to the sofa in the living room. The first robot's view is <image>\n. "
                "The second robot's view is <image>\n. Please output the next action for each robot respectively."
            ) ##记得改prompt
        }
        self.step_num+=1
        while True:
            response = requests.post(self.url, headers=headers, json=data)
            if response.status_code >= 200 and response.status_code < 300:
                return_json = response.json()
                try:
                    response_json = response.json()
                    if self.check_vlm_ans(response_json):
                        return response_json
                except:
                    print("invalid json")
            else:
                print("Wrong status_code")
            time.sleep(0.2)

    def answer_vlm(self,agent_trans_list,agent_query,image,episode_id):
        image = image
        output = self.send_and_receive(image_list= image,episode_id=episode_id)
        result_dict = {}
        print("yuanshi:",output)
        pattern = re.compile(r'robot_(\d+):(\w+)_\[(\-?\d+),(\-?\d+)\]')
        pattern = re.compile(r'robot_(\d+):(\w+)_\[(\-?\d+),(\-?\d+)(?:,(\-?\d+))?\]')
        matches = re.findall(pattern,output)
        for match in matches:
            robotw,action,x,y,z = match
            z = z if z else None
            robot_num = int(robotw) - 1
            print("action",action)
            a = int(x)
            b = int(y)
            if z:
                c = int(z)
            if a == 0 and b == 0 and action == "nav_to_point":
                agent_key = f"agent_{robot_num}"
                result_dict[agent_key] = {
                    "name": action,
                    "position": [a,b] 
                }
                
            else:
                x = float(x) * 0.01
                y = float(y) * 0.01
                if z:
                    z = float(z) * 0.01
                agent_key = f"agent_{robot_num}"
                if z:
                    result_dict[agent_key] = {
                        "name": action,
                        "position": [x,y,z] 
                        }
                else:
                    result_dict[agent_key] = {
                        "name": action,
                        "position": [x,y] 
                        }
        num_query = 0
        output = {}
        print("agent___:",agent_query)
        print("vlm_out:",result_dict)
        for item in result_dict:
            print("agent_query[num_query]:",agent_query[num_query])
            print("item:",result_dict[item])
            if agent_query[num_query] == 1:
                pos = result_dict[item]['position']
                print(f"pos_0:{pos[0]}_pos_1:{pos[1]}")
                if pos == [0,0]:
                    pos.append(0)
                    output[item] = result_dict[item]
                else:
                    if len(pos) == 2:
                        pos.append(0.0)
                    print("pos:",pos)
                    if len(pos) == 3:
                        match = re.search(r'\d+',item)
                        num = -1
                        if match:
                            num = int(match.group())
                            result_dict[item]['position'] = self.pos_trans(pos=pos,
                                                                                trans=agent_trans_list[num]).flatten()[:3].tolist()
                    output[item] = result_dict[item]
                print("temp:",output[item])
            num_query+=1       
        return output 
    def check_vlm_ans(self,json):
        return True
    def answer(self,agent_trans_list,agent_query,image_list = None):
        if image_list != None:
            for i, image in enumerate(image_list):
                obs_k = image
                obs_k = obs_k.squeeze(0)
                if not isinstance(obs_k, np.ndarray):
                    obs_k = obs_k.cpu().numpy()
                if obs_k.dtype != np.uint8:
                    obs_k = obs_k * 255.0
                    obs_k = obs_k.astype(np.uint8)
                if obs_k.shape[2] == 1:
                    obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
                plt.imshow(obs_k)
                plt.axis('off')
                plt.savefig(os.path.join(self.image_dir, str(self.step_num)+'_agent_'+str(i)+'.png'),
                            bbox_inches='tight', pad_inches=0)
            # vlm_output = self.vlm_inference(self.image_dir)
        test_output = self.test_inference(self.json_dir)
        self.step_num+=1
        query = [0] + agent_query
        filter_output = {key:value for i ,(key,value) in enumerate(test_output.items()) if 
                         i < len(query) and query[i] == 1}
        for item in filter_output:
            pos = filter_output[f"{item}"]['position']
            
            if len(pos) == 2:
                pos.append(0.0)
            print("pos:",pos)
            if len(pos) == 3:
                
                match = re.search(r'\d+',item)
                num = -1
                if match:
                    num = int(match.group())
                    filter_output[f"{item}"]['position'] = self.pos_trans(pos=pos,trans=agent_trans_list[num]).flatten()[:3].tolist()
        return filter_output
    
if __name__ == "__main__":
    test_vlmagent = VLMAgent(2,image_dir='./video_dir',json_dir='./video_dir/image_dir/episode_91/episode_91_test.json')
    with open('./video_dir/image_dir/episode_91/episode_91_test.json','r') as file:
        data = json.load(file)
    i = 1
    for item in data:
        agent_trans_list = []
        ele = []
        agent_trans_list.append(item['agent_0_trans_matrix'])
        agent_trans_list.append(item['agent_1_trans_matrix'])
        ele.append(item['agent_0']['name'])
        ele.append(item['agent_1']['name'])
        agent_query = [0 if elem == "continue" else 1 for elem in ele]
        print(f"step:{i}")
        i+=1
        print(test_vlmagent.answer(agent_trans_list=agent_trans_list,agent_query = agent_query))
        
