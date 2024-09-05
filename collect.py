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
        "root":"/mnt/hwfile/gveval/lianghaotian/data/single_agent_train_waypoint",
        "annotation":"/mnt/hwfile/gveval/lianghaotian/data/single_agent_train_waypoint/robotdata_demo.jsonl",
        "data_augment":False,
        "repeat_time":1,
        "length":0
    }
}
combined_data = []
output_path = os.path.join('./single_agent_train_waypoint', 'robotdata_demo.jsonl')
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
                            figure_history = agent_0_action if agent_0_action!="turn" else "nav_to_point"
                            if action_before == "pick":
                                flag_ifpick = 1
                            if action_before != figure_history:
                                if action_before == "nav_to_point":
                                    if flag_ifpick:
                                        action_has_finished+=f"navigate to the table,"
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
                            
                            if agent_0_action == "nav_to_point":
                                x,y= agent_0_pos
                                if not (0<=x<=256 and 0<=y<=256):
                                    agent_0_action = "turn"
                                    agent_0_pos = None
                                else:
                                    agent_0_pos = [[int(x*1000/256),int(y*1000/256),int(10*1000/256) if x+10<256 else 255-x,10 if y+10<256 else 255-y]]
                            elif agent_0_action == "pick" or agent_0_action == "place":
                                x,y,w,h = agent_0_pos[0]
                                agent_0_pos = [[int(x*1000/256),int(y*1000/256),int(w*1000/256),int(h*1000/256)]]
                            if "pick"  in action_has_finished and agent_0_action == "pick":
                                agent_0_action = "place"
                            question = f"""You are an AI visual assistant that can manage a single robot. You receive the robot's task, one image representing the robot's current view and what the robot has completed so far. You need to output the robot's next action. Actions the robot can perform are "nav_to_point", "turn", "pick" and "place". If you cannot determine the next action based on the robot's current view, you can command the robot to "turn". Your output format should be either "turn" or "action_name<box>[[x1, y1, x2, y2]]</box>".Robot's Task: The robot need to navigate to the shelf where the box is located, pick it up,navigate to the table next to the sofa in the living room and place the box.
Robot's current view: <image>
{name_finished}."""
                            conversations = [
                                {
                                    "from":"human",
                                    "value":f"{question}"
                                },
                                {
                                    "from":"gpt",
                                    "value":f"robot:{agent_0_action}<box>{agent_0_pos}</box>\n" if agent_0_action != "turn" else "robot:turn"
                                }
                            ]
                            item["conversations"] = conversations
                            for file_name in os.listdir(episode_path):
                                # if sup['image_0'] in file_name or sup['image_1'] in file_name:
                                if sup['image'] in file_name:
                                    source_file = os.path.join(episode_path, file_name)
                                    destination_folder = f'./single_agent_train_waypoint/image/{num_id}'
                                    os.makedirs(destination_folder, exist_ok=True)
                                    destination_file = os.path.join(destination_folder, file_name)
                                    shutil.copy(source_file, destination_file)
                            num_step+=1
                            action_before = figure_history
                            combined_data.append(item)
                        
meta["robotdata_demo"]["length"] = num_step            
with jsonlines.open(output_path,mode='w') as writer:
     writer.write_all(combined_data)

with open(os.path.join('./single_agent_train_waypoint','robotdemo_meta.json'),'w') as file:
    json.dump(meta,file,indent=4)
                    








               