import json,jsonlines
import glob
import os,shutil
def process_array(arr):
    processed_arr = [str(int(x * 100)) for x in arr]
    return '[' + ','.join(processed_arr) + ']'
# 设定文件所在的目录
directory = './video_dir'
num_step = 0
num_id = -1
meta = {    "robotdata_demo":{
        "root":"/mnt/hwfile/gveval/dulingxiao/final_eval",
        "annotation":"/mnt/hwfile/gveval/dulingxiao/final_eval/robotdata_demo.jsonl",
        "data_augment":False,
        "repeat_time":1,
        "length":0
    }
}
combined_data = []
output_path = os.path.join('./final_train', 'robotdata_demo.jsonl')
folder_list = glob.glob(os.path.join(directory, '*.json'))
# 获取目录中所有 JSON 文件
file_list = glob.glob(os.path.join(directory, '*.json'))
for process_folder in os.listdir(directory):
    process_path = os.path.join(directory, process_folder)
    
    # 检查是否为process_{i}文件夹
    if os.path.isdir(process_path) and process_folder.startswith('process_'):
        # 遍历每个process_{i}文件夹中的文件夹
        for episode_folder in os.listdir(process_path):
            episode_path = os.path.join(process_path, episode_folder)
            
            # 检查是否为episode_i文件夹
            if os.path.isdir(episode_path) and episode_folder.startswith('episode_') and episode_folder!= 'episode_0':
                json_path = os.path.join(episode_path,f'{episode_folder}.json')
                if os.path.exists(json_path):
                    num_id+=1    
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                    flag = 0
                    for item in data:
                        image_0 = item.get("image_0")
                        image_1 = item.get("image_1")
                        image_0_exists = os.path.exists(os.path.join(episode_path, image_0)) if image_0 else False
                        image_1_exists = os.path.exists(os.path.join(episode_path, image_1)) if image_1 else False
                        if image_0_exists!=1 or image_1_exists!=1:
                            flag = 1
                    if flag == 0:
                        for sup in data:
                            item = {}
                            item['id'] = f"episode_{num_id}_{sup['step']}"
                            agent_0_action = sup["agent_0"]["name"]
                            agent_1_action = sup["agent_1"]["name"]
                            image_list = []
                            if agent_0_action == "continue" and agent_1_action != "continue":
                                image_list.append("")
                                image_list.append(f"image/{num_id}/{sup['image_1']}")
                            if agent_1_action == "continue" and agent_0_action != "continue":
                                image_list.append(f"image/{num_id}/{sup['image_0']}")
                                image_list.append("")
                            if agent_0_action == "continue" and agent_1_action == "continue":
                                image_list.append("")
                                image_list.append("")
                            if agent_0_action != "continue" and agent_1_action != "continue":
                                image_list.append(f"image/{num_id}/{sup['image_0']}")
                                image_list.append(f"image/{num_id}/{sup['image_1']}")
                            item['image'] = image_list
                            item['height'] = [256,256]
                            item['weight'] = [256,256]          
                            agent_0_pos = process_array(sup["agent_0"]["position"])
                            agent_1_pos = process_array(sup["agent_1"]["position"])
                            conversations = [
                                {
                                    "from":"human",
                                    "value":"You are now managing two robots. The actions they can perform are \"nav_to_point,\" \"pick,\" \"continue\" and \"place.\" They need to locate all the boxes in the scene, pick them up, and place them on the table next to the sofa in the living room. The first robot's view is <image>\n. The second robot's view is <image>\n. Please output the next action for each robot respectively."
                                },
                                {
                                    "from":"gpt",
                                    "value":f"robot_1:{agent_0_action}_{agent_0_pos}\nrobot_2:{agent_1_action}_{agent_1_pos}"
                                }
                            ]
                            item["conversations"] = conversations
                            for file_name in os.listdir(episode_path):
                                if sup['image_0'] in file_name or sup['image_1'] in file_name:
                                    source_file = os.path.join(episode_path, file_name)
                                    destination_folder = f'./final_train/image/{num_id}'
                                    os.makedirs(destination_folder, exist_ok=True)
                                    destination_file = os.path.join(destination_folder, file_name)
                                    shutil.copy(source_file, destination_file)
                            num_step+=1
                            combined_data.append(item)
                        
meta["robotdata_demo"]["length"] = num_step            
with jsonlines.open(output_path,mode='w') as writer:
     writer.write_all(combined_data)

with open(os.path.join('./final_train','robotdemo_meta.json'),'w') as file:
    json.dump(meta,file,indent=4)
                    








               