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
        "root":"/mnt/hwfile/gveval/dulingxiao/single_agent_train",
        "annotation":"/mnt/hwfile/gveval/dulingxiao/single_agent_train/robotdata_demo.jsonl",
        "data_augment":False,
        "repeat_time":1,
        "length":0
    }
}
combined_data = []
output_path = os.path.join('./single_agent_train', 'robotdata_demo.jsonl')
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
                        image_0 = item.get("image")
                        # image_1 = item.get("image_1")
                        image_0_exists = os.path.exists(os.path.join(episode_path, image_0)) if image_0 else False
                        # image_1_exists = os.path.exists(os.path.join(episode_path, image_1)) if image_1 else False
                        # if image_0_exists!=1 or image_1_exists!=1:
                        if image_0_exists!=1:
                            flag = 1
                    if flag == 0:
                        action_before = "nav_to_point"
                        action_has_finished = ""
                        flag_ifpick = 0
                        for sup in data:
                            item = {}
                            item['id'] = f"episode_{num_id}_{sup['step']}"
                            agent_0_action = sup["action"]["name"]
                            if action_before == "pick":
                                flag_ifpick = 1
                            if action_before != agent_0_action:
                                if action_before == "nav_to_point":
                                    if flag_ifpick:
                                        action_has_finished+=f"navigate to the sofa,"
                                    else:
                                        action_has_finished+=f"navigate to the box,"
                                else:
                                    action_has_finished+=f"{action_before} the box,"
                            # agent_1_action = sup["agent_1"]["name"]
                            image_list = []
                            # if agent_0_action == "continue" and agent_1_action != "continue":
                            #     image_list.append("")
                            #     image_list.append(f"image/{num_id}/{sup['image_1']}")
                            # if agent_1_action == "continue" and agent_0_action != "continue":
                            #     image_list.append(f"image/{num_id}/{sup['image_0']}")
                            #     image_list.append("")
                            # if agent_0_action == "continue" and agent_1_action == "continue":
                            #     image_list.append("")
                            #     image_list.append("")
                            # if agent_0_action != "continue" and agent_1_action != "continue":
                            if agent_0_action != "continue":
                                image_list.append(f"image/{num_id}/{sup['image']}")
                                # image_list.append(f"image/{num_id}/{sup['image_1']}")
                            item['image'] = image_list
                            item['height'] = [256,256]
                            item['weight'] = [256,256]          
                            # agent_0_pos = process_array(sup["agent_0"]["position"])
                            agent_0_pos = sup["action"]["position"]

                            # agent_1_pos = process_array(sup["agent_1"]["position"])
                            if action_has_finished:
                                name_finished = f"The robot has finished:{action_has_finished}"
                            else:
                                name_finished = ""
                            x,y= agent_0_pos
                            if agent_0_action == "nav_to_point":
                                if not (0<=x<=256 and 0<=y<=256):
                                    agent_0_action = "turn"
                                    agent_0_pos = [0,0]
                            elif(agent_0_action == "place"):
                                if not (0<=x<=256 and 0<=y<=256):
                                    agent_0_pos = [max(2,min(x,254)),max(2,min(y,254))]
                            question = f"""You are now managing single robot. The actions it can perform are \"nav_to_point\", \"turn\",\"pick\" and \"place\".The robot's view is <image>\n.The robot need to locate the box in the scene, pick them up, and place them on the table next to the sofa in the living room.If you can not know what action to do next from the current robot's view,you can let the robot \"turn_[0,0]\".{name_finished}Please output the next action for the robot in the format like \"robot:pick_[100,100]\".Your output should include the task coordinates."""
                            conversations = [
                                {
                                    "from":"human",
                                    "value":f"{question}"
                                },
                                {
                                    "from":"gpt",
                                    "value":f"robot:{agent_0_action}_{agent_0_pos}\n"
                                }
                            ]
                            item["conversations"] = conversations
                            for file_name in os.listdir(episode_path):
                                # if sup['image_0'] in file_name or sup['image_1'] in file_name:
                                if sup['image'] in file_name:
                                    source_file = os.path.join(episode_path, file_name)
                                    destination_folder = f'./single_agent_train/image/{num_id}'
                                    os.makedirs(destination_folder, exist_ok=True)
                                    destination_file = os.path.join(destination_folder, file_name)
                                    shutil.copy(source_file, destination_file)
                            num_step+=1
                            action_before = sup["action"]["name"]
                            combined_data.append(item)
                        
meta["robotdata_demo"]["length"] = num_step            
with jsonlines.open(output_path,mode='w') as writer:
     writer.write_all(combined_data)

with open(os.path.join('./single_agent_train','robotdemo_meta.json'),'w') as file:
    json.dump(meta,file,indent=4)
                    








               