import matplotlib.pyplot as plt
import os
import numpy as np
import json
import re
class VLMAgent():
    def __init__(self,agent_num,image_dir,json_dir=None) -> None:
        self.image_dir = image_dir
        self.step_num = 0
        self.json_dir = json_dir
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
        
