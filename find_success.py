import os,json,re
import shutil


video_dir = './video_dir_TRUE/manipulation_new'  # Path to the directory with mp4 files
episode_dir = './video_dir_TRUE/image_dir'  # Path to the directory with episode folders
find_episode = []
episode_id_sum = []
from camera_3dto2d import _3d_to_2d
def datatrans_2_end_single_agent(process_dir:str,skip_len:int) -> list:
    find_episode = []
    process_dir_path = os.path.join('./video_dir',process_dir)
    for folder_name in os.listdir(process_dir_path):
        json_path = os.path.join(process_dir_path,folder_name,"sum_data.json")
        if os.path.exists(json_path):
            with open(json_path,'r',encoding='utf-8') as f:
                data = json.load(f)
                if 'entities' in data:
                    entity_count = len(data['entities'])
                    if 5< entity_count < 496:
                        find_episode.append(folder_name)
    temp_q = 0
    sample_info = []
    for name in find_episode:

        with open(os.path.join(process_dir_path,name,"data/data_trans.json"), 'r') as file:
            data = json.load(file)

        data_final_0 = []
        action = ["nav_to_point","pick","nav_to_point","place"]
        action_index = 0
        i = 0
        result_agent_0 = []
        flag = 0
        while i < len(data):
            if(flag == 1):
                break
            if action_index == 0:
                if data[i+skip_len]["agent_0_action"] == "pick":
                    result = {
                        "step":data[i]["step"],
                        "action":{
                            "name":"nav_to_point",
                            "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
                        },
                        "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                    }
                    j = 0
                    for j in range(i,i+skip_len+1):
                        if(data[j]["agent_0_action"] == "pick"):
                            break
                    i = j
                    action_index = 1
                else:
                    result = {
                            "step":data[i]["step"],
                            "action":{
                                "name":"nav_to_point",
                                "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
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
                        "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_objpos"])
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
                # print("pick_j:",i)
                action_index = 2
                continue
            if action_index == 2:
                if i +skip_len < len(data):
                    if data[i+skip_len]["agent_0_action"] == "place":
                        result = {
                            "step":data[i]["step"],
                            "action":{
                                "name":"nav_to_point",
                                "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
                            },
                            "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                        }
                        j = 0
                        for j in range(i,i+skip_len+1):
                            if(data[j]["agent_0_action"] == "place"):
                                break
                        i = j
                        action_index = 3
                    else:
                        result = {
                                "step":data[i]["step"],
                                "action":{
                                    "name":"nav_to_point",
                                    "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
                                },
                                "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                            }
                        i+= skip_len
                else:
                    result = {
                                "step":data[i]["step"],
                                "action":{
                                    "name":"nav_to_point",
                                    "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
                                },
                                "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                            }
                    j = i
                    while(True):
                        j+=1
                        if(data[j]["agent_0_action"] == "place"):
                            break
                    i = j
                    action_index = 3

                data_final_0.append(result)
                continue
            if action_index == 3:
                result = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"place",
                        "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_target"])
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                }
                data_final_0.append(result)
                flag = 1

        action_index = 0
        temp_info = {
                "episode_id":int(name.replace('episode_', '')),
                "sample_frame":[],
            }
        for i in range(len(data_final_0)):
            match = re.search(r"frame_(\d+)_agent_(\d+)", data_final_0[i]["image"])
            if match:
                frame_number = match.group(1)
                agent_number = match.group(2)
            result_0 = [int(frame_number), int(agent_number)]
            temp_info["sample_frame"].append(result_0)
            
        sample_info.append(temp_info)
        with open(os.path.join(process_dir_path,name,f"{name}.json"), 'w') as file:
            json.dump(data_final_0, file, indent=2)
    return sample_info
def datatrans_2_end(process_dir:str) -> list:
    find_episode = []
    process_dir_path = os.path.join('./video_dir',process_dir)
    for folder_name in os.listdir(process_dir_path):
        json_path = os.path.join(process_dir_path,folder_name,"sum_data.json")
        if os.path.exists(json_path):
            with open(json_path,'r',encoding='utf-8') as f:
                data = json.load(f)
                if 'entities' in data:
                    entity_count = len(data['entities'])
                    if 5< entity_count < 496:
                        find_episode.append(folder_name)
    temp_q = 0
    sample_info = []
    for name in find_episode:
        try:
            with open(os.path.join(process_dir_path,name,"data/data_trans.json"), 'r') as file:
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
                        for j in range(i,i+skip_len+1):
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
                    # print("pick_j:",i)
                    action_index = 2
                    continue
                if action_index ==2:
                    if i+skip_len<len(data):
                        if(data[i+skip_len] == "none"):
                            j = i
                            for j in range(i,i+skip_len+1):
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
                            "image":f"frame_"+str(data[len(data)-4]["step"]+skip_len)+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                        }
                        data_final_0.append(result_1)
                        data_final_0.append(result_2)
                        break
            action_index = 0
            # with open(f'./video_dir/manipulation_eval_process_0.json.gz/{name}/agent0.json', 'w') as file:
            #     json.dump(data_final_0, file, indent=2)
                
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
                        for j in range(i,i+skip_len+1):
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
                            for j in range(i,i+skip_len+1):
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
                            "image":f"frame_"+str(data[len(data)-4]["step"]+skip_len)+"_agent_1_head_rgbFetchRobot_head_rgb.png"
                        }
                        data_final_1.append(result_1)
                        data_final_1.append(result_2)
                        break        
                
            # with open(f'./video_dir/manipulation_eval_process_0.json.gz/{name}/agent1.json', 'w') as file:
            #     json.dump(data_final_1, file, indent=2)
            index_0 = 0
            index_1 = 0
            max_0 = len(data_final_0)
            max_1 = len(data_final_1)
            data_final = []
            num_step = 0
            while(index_0<max_0 or index_1<max_1):
                if(index_0<max_0 and index_1<max_1):
                    if(data_final_0[index_0]["step"] == data_final_1[index_1]["step"]):
                        result = {
                            "step":num_step,
                            "agent_0":data_final_0[index_0]["action"],
                            "agent_1":data_final_1[index_1]["action"],
                            "image_0":data_final_0[index_0]["image"],
                            "image_1":data_final_1[index_1]["image"]
                        }
                        data_final.append(result)
                        num_step+=1
                        index_0+=1
                        index_1+=1
                        continue
                    if(data_final_0[index_0]["step"] > data_final_1[index_1]["step"]):
                        result = {
                            "step":num_step,
                            "agent_0":{
                                "name":"continue",
                                "position":[0,0]
                            },
                            "agent_1":data_final_1[index_1]["action"],
                            "image_0":f"frame_{data_final_1[index_1]['step']}_agent_0_head_rgbFetchRobot_head_rgb.png",
                            "image_1":data_final_1[index_1]["image"]
                        }
                        data_final.append(result)
                        num_step+=1
                        index_1+=1
                        continue
                    if(data_final_0[index_0]["step"] < data_final_1[index_1]["step"]):
                        result = {
                            "step":num_step,
                            "agent_0":data_final_0[index_0]["action"],
                            "agent_1":{
                                "name":"continue",
                                "position":[0,0]
                            },
                            "image_0":data_final_0[index_0]["image"],
                            "image_1":f"frame_{data_final_0[index_0]['step']}_agent_1_head_rgbFetchRobot_head_rgb.png"
                        }
                        data_final.append(result)
                        num_step+=1
                        index_0+=1
                        continue
                if(index_0>=max_0 and index_1<max_1):
                    result = {
                        "step":num_step,
                        "agent_0":{
                            "name":"none",
                            "position":[0,0]
                        },
                        "agent_1":data_final_1[index_1]["action"],
                        "image_0":data_final_0[max_0-1]["image"],
                        "image_1":data_final_1[index_1]["image"]
                    }
                    data_final.append(result)
                    num_step+=1
                    index_1+=1
                    continue
                if(index_0<max_0 and index_1>=max_1):
                    result = {
                        "step":num_step,
                        "agent_0":data_final_0[index_0]["action"],
                        "agent_1":{
                            "name":"none",
                            "position":[0,0]
                        },
                        "image_0":data_final_0[index_0]["image"],
                        "image_1":data_final_1[max_1-1]["image"]
                    }
                    data_final.append(result)
                    num_step+=1
                    index_1+=1
                    continue
            temp_info = {
                    "episode_id":int(name.replace('episode_', '')),
                    "sample_frame":[],
                }
            for i in range(len(data_final)):
                match = re.search(r"frame_(\d+)_agent_(\d+)", data_final[i]["image_0"])
                if match:
                    frame_number = match.group(1)
                    agent_number = match.group(2)
                result_0 = [int(frame_number), int(agent_number)]
                temp_info["sample_frame"].append(result_0)
                match = re.search(r"frame_(\d+)_agent_(\d+)", data_final[i]["image_1"])
                if match:
                    frame_number = match.group(1)
                    agent_number = match.group(2)
                result_1 = [int(frame_number), int(agent_number)]
                temp_info["sample_frame"].append(result_1)
            sample_info.append(temp_info)
            with open(os.path.join(process_dir_path,name,f"{name}.json"), 'w') as file:
                json.dump(data_final, file, indent=2)
        except:
            print("name",name)
    return sample_info


    
if __name__ == "__main__":
    print(datatrans_2_end_single_agent(process_dir="image_dir"))