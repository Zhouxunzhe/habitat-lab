import json
ans = ['episode_157', 'episode_134', 'episode_9', 'episode_52', 'episode_123', 'episode_32', 
       'episode_28', 'episode_145', 'episode_105', 'episode_71', 'episode_54', 'episode_88', 
       'episode_111', 'episode_50', 'episode_18', 'episode_90', 'episode_34', 'episode_86', 
       'episode_60', 'episode_81', 'episode_159', 'episode_156', 'episode_13', 'episode_147', 
       'episode_139', 'episode_39', 'episode_151', 'episode_146', 'episode_25', 'episode_153',
         'episode_161', 'episode_47', 'episode_144', 'episode_115']
name = 'episode_105'
with open(f'./video_dir/image_dir/{name}/data/data_trans.json', 'r') as file:
    data = json.load(file)

data_final_0 = []
action = ["nav_to_point","pick","nav_to_point","none"]
action_index = 0
skip_len = 20
i = 0
result_agent_0 = []
while i < len(data):
    if action_index == 0:
        if data[i+skip_len]["agent_0_action"] == "pick":
            result = {
                "step":data[i]["step"],
                "action":{
                    "name":"nav_to_point",
                    "position":data[i]["agent_0_pre_robotloc"][:2]
                },
                "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
            }
            j = 0
            for j in range(i,i+skip_len):
                if(data[j]["agent_0_action"] == "pick"):
                    break
            i = j
            action_index = 1
        else:
            result = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"nav_to_point",
                        "position":data[i]["agent_0_pre_robotloc"][:2]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                }
            i+= skip_len
        data_final_0.append(result)
        continue
    if action_index == 1:
        result = {
            "step":data[i]["step"],
            "action":{
                "name":"pick",
                "position":data[i]["agent_0_obj_ro"]
            },
            "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
        }
        data_final_0.append(result)
        j = i
        while(True):
            if(data[j]["agent_0_action"]== "nav_to_point"):
                break
            j+=1
        i = j
        action_index = 2
        continue
    if action_index ==2:
        if i+skip_len<len(data):
            if(data[i+skip_len] == "none"):
                j = i
                for j in range(i,i+skip_len):
                    if(data[j]["agent_0_action"] == "none"):
                        break
                    j+=1
                result_1 = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"nav_to_point",
                        "position":data[i]["agent_0_pre_robotloc"][:2]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                }
                result_2 = {
                    "step":data[i+skip_len]["step"],
                    "action":{
                        "name":"none",
                        "position":[0,0]
                    },
                    "image":f"frame_"+str(data[j]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                }
                data_final_0.append(result_1)
                data_final_0.append(result_2)
                break
            else:
                result = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"nav_to_point",
                        "position":data[i]["agent_0_pre_robotloc"][:2]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                    }
                data_final_0.append(result)
                i+= skip_len
                continue
        else:
            result_1 = {
                "step":data[i]["step"],
                "action":{
                    "name":"nav_to_point",
                    "position":data[i]["agent_0_pre_robotloc"][:2]
                },
                "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
            }
            result_2 = {
                "step":len(data),
                "action":{
                    "name":"none",
                    "position":[0,0]
                },
                "image":f"frame_"+str(data[len(data)-1]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
            }
            data_final_0.append(result_1)
            data_final_0.append(result_2)
            break
action_index = 0
with open(f'./video_dir/image_dir/{name}/data/agent0.json', 'w') as file:
    json.dump(data_final_0, file, indent=2)
    
data_final_1 = []
i = 0
while i < len(data):
    if action_index == 0:
        if data[i+skip_len]["agent_1_action"] == "pick":
            j = 0
            result = {
                "step":data[i]["step"],
                "action":{
                    "name":"nav_to_point",
                    "position":data[i]["agent_1_pre_robotloc"][:2]
                },
                "image":f"frame_"+str(data[i]["step"])+"_agent_1_head_rgbFetchRobot_head_rgb.png"
            }
            for j in range(i,i+skip_len):
                if(data[j]["agent_1_action"] == "pick"):
                    break
            i = j
            action_index = 1
        else:
            
            result = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"nav_to_point",
                        "position":data[i]["agent_1_pre_robotloc"][:2]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_1_head_rgbFetchRobot_head_rgb.png"
                }
            i+= skip_len
        data_final_1.append(result)
        continue
    if action_index == 1:
        result = {
            "step":data[i]["step"],
            "action":{
                "name":"pick",
                "position":data[i]["agent_1_obj_ro"]
            },
            "image":f"frame_"+str(data[i]["step"])+"_agent_1_head_rgbFetchRobot_head_rgb.png"
        }
        data_final_1.append(result)
        j = i
        while(True):
            if(data[j]["agent_1_action"]== "nav_to_point"):
                break
            j+=1
        i = j
        action_index = 2
        continue
    if action_index ==2:
        if i+skip_len<len(data):
            if(data[i+skip_len] == "none"):
                j = i
                for j in range(i,i+skip_len):
                    if(data[j]["agent_1_action"] == "none"):
                        break
                    j+=1
                result_1 = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"nav_to_point",
                        "position":data[i]["agent_1_pre_robotloc"][:2]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_1_head_rgbFetchRobot_head_rgb.png"
                }
                result_2 = {
                    "step":data[i+skip_len]["step"],
                    "action":{
                        "name":"none",
                        "position":[0,0]
                    },
                    "image":f"frame_"+str(data[j]["step"])+"_agent_1_head_rgbFetchRobot_head_rgb.png"
                }
                data_final_1.append(result_1)
                data_final_1.append(result_2)
                break
            else:
                result = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"nav_to_point",
                        "position":data[i]["agent_1_pre_robotloc"][:2]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_1_head_rgbFetchRobot_head_rgb.png"
                    }
                data_final_1.append(result)
                i+= skip_len
                continue
        else:
            result_1 = {
                "step":data[i]["step"],
                "action":{
                    "name":"nav_to_point",
                    "position":data[i]["agent_1_pre_robotloc"][:2]
                },
                "image":f"frame_"+str(data[i]["step"])+"_agent_1_head_rgbFetchRobot_head_rgb.png"
            }
            result_2 = {
                "step":len(data),
                "action":{
                    "name":"none",
                    "position":[0,0]
                },
                "image":f"frame_"+str(data[len(data)-1]["step"])+"_agent_1_head_rgbFetchRobot_head_rgb.png"
            }
            data_final_1.append(result_1)
            data_final_1.append(result_2)
            break        

with open(f'./video_dir/image_dir/{name}/data/agent1.json', 'w') as file:
    json.dump(data_final_1, file, indent=2)
    
index_0 = 0
index_1 = 0
max_0 = len(data_final_0)
max_1 = len(data_final_1)
data_final = []
num_step = 0
action_index = 0
index_pick_0 = 0
index_pick_1 = 1
for index_0 in range(0,max_0):
    if data_final_0[index_0]["action"]["name"] == "pick":
        index_pick_0=index_0
        break
for index_1 in range(0,max_1):
    if data_final_1[index_1]["action"]["name"] == "pick":
        index_pick_1=index_1
        break

if(index_pick_0>=index_pick_1):
    for i in range(0,index_pick_1):
        result = {
            "step":num_step,
            "agent_0":data_final_0[i]["action"],
            "agent_1":data_final_1[i]["action"],
            "image_0":data_final_0[i]["image"],
            "image_1":data_final_1[i]["image"],
        }
        data_final.append(result)
        num_step +=1
    for i in range(index_pick_1,index_pick_0):
        result = {
            "step":num_step,
            "agent_0":data_final_0[i]["action"],
            "agent_1":{
                "action":"none",
                "position":[0,0]
            },
            "image_0":data_final_0[i]["image"],
            "image_1":f"frame_"+str(data_final_0[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
        }
        data_final.append(result)
        num_step+=1
else:
    for i in range(0,index_pick_0):
        result = {
            "step":num_step,
            "agent_0":data_final_0[i]["action"],
            "agent_1":data_final_1[i]["action"],
            "image_0":data_final_0[i]["image"],
            "image_1":data_final_1[i]["image"],
        }
        data_final.append(result)
        num_step +=1
    for i in range(index_pick_0,index_pick_1):
        result = {
            "step":num_step,
            "agent_0":{
                "action":"none",
                "position":[0,0]
            },
            "agent_1":data_final_1[i]["action"],
            "image_0":f"frame_"+str(data_final_1[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
            "image_1":data_final_1[i]["image"]
        }
        data_final.append(result)
        num_step+=1
result = {
    "step":num_step,
    "agent_0":data_final_0[index_pick_0]["action"],
    "agent_1":data_final_1[index_pick_1]["action"],
    "image_0":data_final_0[index_pick_0]["image"],
    "image_1":data_final_1[index_pick_1]["image"]
}
data_final.append(result)
num_step+=1
a = index_pick_0+1
b = index_pick_1+1
print(f"temp:a = {a},b = {b},max0 = {max_0},max1 = {max_1}")
print(data_final)
if(max_0-index_pick_0>=max_1-index_pick_1):
    for i in range(index_pick_1+1,max_1):
        result = {
            "step":num_step,
            "agent_0":data_final_0[i+index_pick_0-index_pick_1]["action"],
            "agent_1":data_final_1[i]["action"],
            "image_0":data_final_0[i+index_pick_0-index_pick_1]["image"],
            "image_1":data_final_1[i]["image"],
        }
        data_final.append(result)
        num_step +=1
    for i in range(max_1,max_0):
        result = {
            "step":num_step,
            "agent_0":data_final_0[i+index_pick_0-index_pick_1]["action"],
            "agent_1":{
                "action":"none",
                "position":[0,0]
            },
            "image_0":data_final_0[i+index_pick_0-index_pick_1]["image"],
            "image_1":f"frame_"+str(data_final_0[max_1]["step"]+skip_len-2)+"_agent_1_head_rgbFetchRobot_head_rgb.png"
        }
        data_final.append(result)
        num_step+=1
else:
    for i in range(index_pick_0+1,max_0):
        print(f"index:{i+index_pick_0-index_pick_1}")
        print()
        result = {
            "step":num_step,
            "agent_0":data_final_0[i]["action"],
            "agent_1":data_final_1[i+index_pick_1-index_pick_0]["action"],
            "image_0":data_final_0[i]["image"],
            "image_1":data_final_1[i+index_pick_1-index_pick_0]["image"],
        }
        data_final.append(result)
        num_step +=1
    for i in range(max_0,max_1):
        result = {
            "step":num_step,
            "agent_0":{
                "action":"none",
                "position":[0,0]
            },
            "agent_1":data_final_1[i+index_pick_1-index_pick_0]["action"],
            "image_0":f"frame_"+str(data_final_1[max_0]["step"]+skip_len-2)+"_agent_0_head_rgbFetchRobot_head_rgb.png",
            "image_1":data_final_1[i+index_pick_1-index_pick_0]["image"]
        }
        data_final.append(result)
        num_step+=1

with open(f'./video_dir/image_dir/{name}/data/end.json', 'w') as file:
    json.dump(data_final, file, indent=2)
    
# import shutil,os
# image_files = set()
# source_path = './video_dir/image_dir/{name}/'
# destination_folder= './video_dir/image_dir/{name}/data/ans'
# for i in range(0,len(data_final)):
#     image_file_0 = data_final[i]["image_0"]
#     image_file_1 = data_final[i]["image_1"]
#     # print(image_file_0)
#     image_path_0 = os.path.join(source_path, image_file_0)
#     image_path_1 = os.path.join(source_path, image_file_1)

#     image_name_0 = f"agent_0_step_{i}.png"
#     image_name_1 = f"agent_1_step_{i}.png"
#     desire_path_0 = os.path.join(destination_folder, image_name_0)
#     desire_path_1 = os.path.join(destination_folder, image_name_1)
#     shutil.copy(image_path_0, desire_path_0)
#     shutil.copy(image_path_1, desire_path_1)

# for i in range(0,len(data_final)):
#     data_final[i]["image_0"] = f"agent_0_step_{i}.png"
#     data_final[i]["image_1"] = f"agent_1_step_{i}.png"

# with open('./video_dir/image_dir/{name}/data/ans/end.json', 'w') as file:
#     json.dump(data_final, file, indent=2)