import os,json,re
import shutil
import random,pdb

video_dir = './video_dir_TRUE/manipulation_new'  # Path to the directory with mp4 files
episode_dir = './video_dir_TRUE/image_dir'  # Path to the directory with episode folders
find_episode = []
episode_id_sum = []
from camera_3dto2d import _3d_to_2d

def check_if_in_range(t):
    if t>=0 and t<256:
        return True
def check_bounding_box(data):
    if len(data) != 1:
        return False
    x,y,w,h = data[0]
    if check_if_in_range(x) and check_if_in_range(y) and check_if_in_range(x+w) and check_if_in_range(y+h):
        return True
    return False

def limit_to_range(num):
    if num<0:
        return 0
    elif num>255:
        return 255
    else:
        return num
def datatrans_2_end_single_agent_objectcentric(process_dir:str,skip_len:int,pick_place_sample_num=3) -> list:
    find_episode = []
    skip_len_start = skip_len
    process_dir_path = os.path.join('./video_dir',process_dir)
    for folder_name in os.listdir(process_dir_path):
        json_path = os.path.join(process_dir_path,folder_name,"sum_data.json")
        if os.path.exists(json_path):
            with open(json_path,'r',encoding='utf-8') as f:
                data = json.load(f)
                if 'entities' in data:
                    entity_count = len(data['entities'])
                    if 5< entity_count < 747:
                        find_episode.append(folder_name)
    temp_q = 0
    sample_info = []
    for name in find_episode:
        try:
            with open(os.path.join(process_dir_path,name,"data/data_trans.json"), 'r') as file:
                data = json.load(file)

            data_final_0 = []
            action = ["turn","nav_to_point","pick","turn","nav_to_point","place"]
            action_point_index = []
            i = 1
            result_agent_0 = []
            flag = 0
            late_action = data[0]["agent_0_action"]
            while i < len(data):
                if (data[i]["agent_0_action"] != late_action):
                    action_point_index.append(i)
                    late_action = data[i]["agent_0_action"]
                i+=1
            assert len(action_point_index) == 5,"Wrong episode"
            if check_bounding_box(data[0]["agent_0_rec"]):
                nav_ = {
                    "step":1,
                    "action":{
                        "name":"nav_to_point",
                        "position":data[0]["agent_0_rec"]
                    },
                    "image":f"frame_1"+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                }
                data_final_0.append(nav_)
            else:
                turn1 = {
                    "step":1,
                    "action":{
                        "name":"turn",
                        "position":None
                    },
                    "image":f"frame_1"+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                }
                data_final_0.append(turn1)
            nav_1_point = [action_point_index[0]]
            i = action_point_index[0]
            skip_len = skip_len_start +random.randint(-10,10)
            while i +skip_len < action_point_index[1]:
                i +=skip_len
                skip_len = skip_len_start +random.randint(-10,10)
                nav_1_point.append(i)
            nav_1_point.append(action_point_index[1])
            for i in range(len(nav_1_point)):
                if i+1< len(nav_1_point):
                    if check_bounding_box(data[nav_1_point[i]]["agent_0_rec"]):
                        nav_temp = {
                            "step":data[nav_1_point[i]]["step"],
                            "action":{
                                "name":"nav_to_point",
                                "position": data[nav_1_point[i]]["agent_0_rec"]
                            },
                            "image":f"frame_"+str(data[nav_1_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                        }
                        data_final_0.append(nav_temp)
            for i in range(0,pick_place_sample_num):
                pick_skip = i*20
                if pick_skip+action_point_index[1] < len(data):
                    pick_temp = {
                        "step":data[pick_skip+action_point_index[1]]["step"],
                        "action":{
                            "name":"pick",
                            "position":data[pick_skip+action_point_index[1]]["agent_0_obj"]
                        },
                        "image":f"frame_"+str(data[pick_skip+action_point_index[1]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                    }
                    data_final_0.append(pick_temp)
            if check_bounding_box(data[action_point_index[2]]["agent_0_target"]):
                nav_ = {
                    "step":action_point_index[2],
                    "action":{
                        "name":"nav_to_point",
                        "position":data[action_point_index[2]]["agent_0_target"]
                    },
                    "image":f"frame_"+str(action_point_index[2])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                }
                data_final_0.append(nav_)
            else:
                turn2 = {
                    "step":action_point_index[2],
                    "action":{
                        "name":"turn",
                        "position":None
                    },
                    "image":f"frame_"+str(action_point_index[2])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                }
                data_final_0.append(turn2)
            nav_2_point = [action_point_index[3]]
            i = action_point_index[3]
            skip_len = skip_len_start +random.randint(-10,10)
            while i +skip_len < action_point_index[4]:
                i +=skip_len
                skip_len = skip_len_start +random.randint(-10,10)
                nav_2_point.append(i)
            nav_2_point.append(action_point_index[4])
            for i in range(len(nav_2_point)):
                if i+1< len(nav_2_point):
                    if check_bounding_box(data[nav_2_point[i]]["agent_0_target"]):
                        nav_temp = {
                            "step":data[nav_2_point[i]]["step"],
                            "action":{
                                "name":"nav_to_point",
                                "position":data[nav_2_point[i]]["agent_0_target"]
                            },
                            "image":f"frame_"+str(data[nav_2_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                        }
                        data_final_0.append(nav_temp)
            for i in range(0,pick_place_sample_num):
                place_skip = i*20
                if place_skip+action_point_index[4] < len(data):
                    place_temp = {
                        "step":data[place_skip+action_point_index[4]]["step"],
                        "action":{
                            "name":"place",
                            "position":data[place_skip+action_point_index[4]]["agent_0_target"]
                        },
                        "image":f"frame_"+str(data[place_skip+action_point_index[4]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
                    }
                    data_final_0.append(place_temp)
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
        except:
            continue
    return sample_info
def datatrans_2_end_single_agent_waypoint(process_dir:str,skip_len:int,pick_place_sample_num=3,sample_clip=800) -> list:
    find_episode = []
    
    skip_len_start = skip_len
    process_dir_path = process_dir
    for folder_name in os.listdir(process_dir_path):
        json_path = os.path.join(process_dir_path,folder_name,"sum_data.json")
        if os.path.exists(json_path):
            with open(json_path,'r',encoding='utf-8') as f:
                data = json.load(f)
                if 'entities' in data:
                    entity_count = len(data['entities'])
                    if 5< entity_count < sample_clip-5:
                        find_episode.append(folder_name)
    temp_q = 0
    sample_info = []
    for name in find_episode:
        try:
            with open(os.path.join(process_dir_path,name,"data/data_trans.json"), 'r') as file:
                data = json.load(file)

            data_final_0 = []
            action = ["turn","nav_to_point","pick","turn","nav_to_point","place"]
            action_point_index = []
            i = 1
            result_agent_0 = []
            flag = 0
            late_action = data[0]["agent_0_action"]
            while i < len(data):
                if (data[i]["agent_0_action"] != late_action):
                    action_point_index.append(i)
                    late_action = data[i]["agent_0_action"]
                i+=1
            # print("action_point_index:",action_point_index)
            assert len(action_point_index) == 5,"Wrong episode"
            turn1 = {
                "step":1,
                "action":{
                    "name":"search_for_object_rec",
                    "position":None
                },
                "image":f"frame_1"+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                "step_info":{}
            }
            data_final_0.append(turn1)
            nav_1_point = [action_point_index[0]]
            i = action_point_index[0]
            skip_len = skip_len_start +random.randint(-3,3)
            while i +skip_len < action_point_index[1]:
                now_step = i
                if i+skip_len+14 >= action_point_index[1]:
                    i = action_point_index[1]
                else:
                    i +=skip_len
                skip_len = skip_len_start +random.randint(-3,3)
                test_step = i
                for a in range(now_step,test_step):
                    test_point = _3d_to_2d(matrix=data[now_step]["agent_0_martix"],
                                                 point_3d=data[test_step]["agent_0_now_worldloc"][:3])
                    x,y = test_point
                    if not (0 <= x < 256 and 0 <= y < 256):
                        test_step -=1
                    else:
                        break
                nav_1_point.append(test_step)
                i = test_step
            if nav_1_point[-1] != action_point_index[1]:
                nav_1_point.append(action_point_index[1])
            for i in range(len(nav_1_point)):
                if i+1< len(nav_1_point):
                    x,y = _3d_to_2d(matrix=data[nav_1_point[i]]["agent_0_martix"],
                                                 point_3d=data[nav_1_point[i+1]]["agent_0_now_worldloc"][:3])
                    x = limit_to_range(x)
                    y = limit_to_range(y)
                    nav_temp = {
                        "step":data[nav_1_point[i]]["step"],
                        "action":{
                            "name":"nav_to_point",
                            "position":[x,y]
                        },
                        "image":f"frame_"+str(data[nav_1_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                        "step_info":{
                            "now_loc":data[nav_1_point[i]]["agent_0_now_worldloc"][:3],
                            "obj_bbox":data[nav_1_point[i]]["agent_0_obj"],
                            "object_rec_bbox":data[nav_1_point[i]]["agent_0_rec"],
                        }
                    }
                    data_final_0.append(nav_temp)
            pick_step_num = (action_point_index[2]-action_point_index[1])*0.4
            pick_step_jump = float(pick_step_num/pick_place_sample_num)
            for i in range(0,pick_place_sample_num):
                pick_skip = int(i*pick_step_jump)
                if pick_skip+action_point_index[1] < len(data):
                    pick_temp = {
                        "step":data[pick_skip+action_point_index[1]]["step"],
                        "action":{
                            "name":"pick",
                            "position":data[pick_skip+action_point_index[1]]["agent_0_obj"]
                        },
                        "image":f"frame_"+str(data[pick_skip+action_point_index[1]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                        "step_info":{
                            "now_loc":data[pick_skip+action_point_index[1]]["agent_0_now_worldloc"][:3],
                            "obj_bbox":data[pick_skip+action_point_index[1]]["agent_0_obj"],
                            "object_rec_bbox":data[pick_skip+action_point_index[1]]["agent_0_rec"],
                        }
                    }
                    data_final_0.append(pick_temp)
            turn2 = {
                "step":action_point_index[2],
                "action":{
                    "name":"search_for_goal_rec",
                    "position":None
                },
                "image":f"frame_"+str(action_point_index[2])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                "step_info":{}
            }
            data_final_0.append(turn2)
            nav_2_point = [action_point_index[3]]
            i = action_point_index[3]
            skip_len = skip_len_start +random.randint(-3,3)
            while i +skip_len < action_point_index[4]:
                now_step = i
                if i+skip_len+14 >= action_point_index[4]:
                    i = action_point_index[4]
                else:
                    i +=skip_len
                skip_len = skip_len_start +random.randint(-3,3)
                test_step = i
                for a in range(now_step,test_step):
                    test_point = _3d_to_2d(matrix=data[now_step]["agent_0_martix"],
                                                 point_3d=data[test_step]["agent_0_now_worldloc"][:3])
                    x,y = test_point
                    if not (0 <= x < 256 and 0 <= y < 256):
                        test_step -=1
                    else:
                        break
                nav_2_point.append(test_step)
                i = test_step
            if nav_2_point[-1] != action_point_index[4]:
                nav_2_point.append(action_point_index[4])
            for i in range(len(nav_2_point)):
                if i+1< len(nav_2_point):
                    x,y = _3d_to_2d(matrix=data[nav_2_point[i]]["agent_0_martix"],
                                                 point_3d=data[nav_2_point[i+1]]["agent_0_now_worldloc"][:3])
                    x = limit_to_range(x)
                    y = limit_to_range(y)
                    nav_temp = {
                        "step":data[nav_2_point[i]]["step"],
                        "action":{
                            "name":"nav_to_point",
                            "position": [x,y]
                            },
                        "image":f"frame_"+str(data[nav_2_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                        "step_info":{
                            "now_loc":data[nav_2_point[i]]["agent_0_now_worldloc"][:3],
                            "target_rec_bbox":data[nav_2_point[i]]["agent_0_target"]
                        }
                    }
                    data_final_0.append(nav_temp)
            for i in range(0,pick_place_sample_num):
                place_skip = i*13
                if place_skip+action_point_index[4] < len(data)-4:
                    place_temp = {
                        "step":data[place_skip+action_point_index[4]]["step"],
                        "action":{
                            "name":"place",
                            "position":data[place_skip+action_point_index[4]]["agent_0_target"]
                        },
                        "image":f"frame_"+str(data[place_skip+action_point_index[4]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                        "step_info":{
                            "now_loc":data[place_skip+action_point_index[4]]["agent_0_now_worldloc"][:3],
                            "target_rec_bbox":data[place_skip+action_point_index[4]]["agent_0_target"]
                        }
                    }
                    data_final_0.append(place_temp)
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
        except:
            continue
    return sample_info
    #     while i < len(data):
    #         if(flag == 1):
    #             break
    #         if action_index == 0:
    #             if data[i+skip_len]["agent_0_action"] == "pick":
    #                 result = {
    #                     "step":data[i]["step"],
    #                     "action":{
    #                         "name":"nav_to_point",
    #                         "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
    #                     },
    #                     "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
    #                 }
    #                 j = 0
    #                 for j in range(i,i+skip_len+1):
    #                     if(data[j]["agent_0_action"] == "pick"):
    #                         break
    #                 i = j
    #                 action_index = 1
    #             else:
    #                 result = {
    #                         "step":data[i]["step"],
    #                         "action":{
    #                             "name":"nav_to_point",
    #                             "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
    #                         },
    #                         "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
    #                     }
    #                 i+= skip_len
    #             data_final_0.append(result)
    #             continue
    #         if action_index == 1:
    #             result = {
    #                 "step":data[i]["step"],
    #                 "action":{
    #                     "name":"pick",
    #                     "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_objpos"])
    #                 },
    #                 "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
    #             }
    #             data_final_0.append(result)
    #             j = i
    #             while(True):
    #                 if(data[j]["agent_0_action"]== "nav_to_point"):
    #                     break
    #                 j+=1
    #             i = j
    #             # print("pick_j:",i)
    #             action_index = 2
    #             continue
    #         if action_index == 2:
    #             if i +skip_len < len(data):
    #                 if data[i+skip_len]["agent_0_action"] == "place":
    #                     result = {
    #                         "step":data[i]["step"],
    #                         "action":{
    #                             "name":"nav_to_point",
    #                             "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
    #                         },
    #                         "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
    #                     }
    #                     j = 0
    #                     for j in range(i,i+skip_len+1):
    #                         if(data[j]["agent_0_action"] == "place"):
    #                             break
    #                     i = j
    #                     action_index = 3
    #                 else:
    #                     result = {
    #                             "step":data[i]["step"],
    #                             "action":{
    #                                 "name":"nav_to_point",
    #                                 "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
    #                             },
    #                             "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
    #                         }
    #                     i+= skip_len
    #             else:
    #                 result = {
    #                             "step":data[i]["step"],
    #                             "action":{
    #                                 "name":"nav_to_point",
    #                                 "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_pre_worldloc"])
    #                             },
    #                             "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
    #                         }
    #                 j = i
    #                 while(True):
    #                     j+=1
    #                     if(data[j]["agent_0_action"] == "place"):
    #                         break
    #                 i = j
    #                 action_index = 3

    #             data_final_0.append(result)
    #             continue
    #         if action_index == 3:
    #             result = {
    #                 "step":data[i]["step"],
    #                 "action":{
    #                     "name":"place",
    #                     "position":_3d_to_2d(matrix=data[i]["agent_0_martix"],point_3d=data[i]["agent_0_target"])
    #                 },
    #                 "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png"
    #             }
    #             data_final_0.append(result)
    #             flag = 1

    #     action_index = 0
    #     temp_info = {
    #             "episode_id":int(name.replace('episode_', '')),
    #             "sample_frame":[],
    #         }
    #     for i in range(len(data_final_0)):
    #         match = re.search(r"frame_(\d+)_agent_(\d+)", data_final_0[i]["image"])
    #         if match:
    #             frame_number = match.group(1)
    #             agent_number = match.group(2)
    #         result_0 = [int(frame_number), int(agent_number)]
    #         temp_info["sample_frame"].append(result_0)
            
    #     sample_info.append(temp_info)
    #     with open(os.path.join(process_dir_path,name,f"{name}.json"), 'w') as file:
    #         json.dump(data_final_0, file, indent=2)
    # return sample_info
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

def datatrans_2_end_sat_waypoint_closer(process_dir:str,skip_len:int,pick_place_sample_num=3,sample_clip=800) -> list:
    find_episode = []
    
    skip_len_start = skip_len
    process_dir_path = process_dir
    for folder_name in os.listdir(process_dir_path):
        json_path = os.path.join(process_dir_path,folder_name,"sum_data.json")
        if os.path.exists(json_path):
            with open(json_path,'r',encoding='utf-8') as f:
                data = json.load(f)
                if 'entities' in data:
                    entity_count = len(data['entities'])
                    if 5< entity_count < sample_clip-5:
                        find_episode.append(folder_name)
    temp_q = 0
    sample_info = []
    for name in find_episode:
        try:
            with open(os.path.join(process_dir_path,name,"data/data_trans.json"), 'r') as file:
                data = json.load(file)

            data_final_0 = []
            action = ["turn","nav_to_point","pick","turn","nav_to_point","place"]
            action_point_index = []
            i = 1
            result_agent_0 = []
            flag = 0
            late_action = data[0]["agent_0_action"]
            while i < len(data):
                if (data[i]["agent_0_action"] != late_action):
                    action_point_index.append(i)
                    late_action = data[i]["agent_0_action"]
                i+=1
            # print("action_point_index:",action_point_index)
            assert len(action_point_index) == 5,"Wrong episode"
            turn1 = {
                "step":1,
                "action":{
                    "name":"search_for_object_rec",
                    "position":None
                },
                "image":f"frame_1"+"_agent_0_head_rgbFetchRobot_head_rgb.png",
            }
            data_final_0.append(turn1)
            nav_1_point = [action_point_index[0]]
            i = action_point_index[0]
            skip_len = skip_len_start +random.randint(-3,3)
            while i +skip_len < action_point_index[1]:
                now_step = i
                if i+skip_len+14 >= action_point_index[1]:
                    i = action_point_index[1]
                else:
                    i +=skip_len
                skip_len = skip_len_start +random.randint(-3,3)
                test_step = i
                for a in range(now_step,test_step):
                    test_point = _3d_to_2d(matrix=data[now_step]["agent_0_martix"],
                                                 point_3d=data[test_step]["agent_0_now_worldloc"][:3])
                    x,y = test_point
                    if not (0 <= x < 256 and 0 <= y < 256):
                        test_step -=1
                    else:
                        break
                nav_1_point.append(test_step)
                i = test_step
            if nav_1_point[-1] != action_point_index[1]:
                nav_1_point.append(action_point_index[1])
            for i in range(len(nav_1_point)):
                if i+1< len(nav_1_point):
                    x,y = _3d_to_2d(matrix=data[nav_1_point[i]]["agent_0_martix"],
                                                 point_3d=data[nav_1_point[i+1]]["agent_0_now_worldloc"][:3])
                    x = limit_to_range(x)
                    y = limit_to_range(y)
                    nav_temp = {
                        "step":data[nav_1_point[i]]["step"],
                        "action":{
                            "name":"nav_to_point",
                            "position":[x,y]
                        },
                        "image":f"frame_"+str(data[nav_1_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                    }
                    data_final_0.append(nav_temp)
            for i in range(action_point_index[1]-2,action_point_index[1]+1):
                pick_temp = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"pick",
                        "position":data[i]["agent_0_obj"]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                }
                data_final_0.append(pick_temp)
            turn2 = {
                "step":action_point_index[2],
                "action":{
                    "name":"search_for_goal_rec",
                    "position":None
                },
                "image":f"frame_"+str(action_point_index[2])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                "step_info":{}
            }
            data_final_0.append(turn2)
            nav_2_point = [action_point_index[3]]
            i = action_point_index[3]
            skip_len = skip_len_start +random.randint(-3,3)
            while i +skip_len < action_point_index[4]:
                now_step = i
                if i+skip_len+14 >= action_point_index[4]:
                    i = action_point_index[4]
                else:
                    i +=skip_len
                skip_len = skip_len_start +random.randint(-3,3)
                test_step = i
                for a in range(now_step,test_step):
                    test_point = _3d_to_2d(matrix=data[now_step]["agent_0_martix"],
                                                 point_3d=data[test_step]["agent_0_now_worldloc"][:3])
                    x,y = test_point
                    if not (0 <= x < 256 and 0 <= y < 256):
                        test_step -=1
                    else:
                        break
                nav_2_point.append(test_step)
                i = test_step
            if nav_2_point[-1] != action_point_index[4]:
                nav_2_point.append(action_point_index[4])
            for i in range(len(nav_2_point)):
                if i+1< len(nav_2_point):
                    x,y = _3d_to_2d(matrix=data[nav_2_point[i]]["agent_0_martix"],
                                                 point_3d=data[nav_2_point[i+1]]["agent_0_now_worldloc"][:3])
                    x = limit_to_range(x)
                    y = limit_to_range(y)
                    nav_temp = {
                        "step":data[nav_2_point[i]]["step"],
                        "action":{
                            "name":"nav_to_point",
                            "position": [x,y]
                            },
                        "image":f"frame_"+str(data[nav_2_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                    }
                    data_final_0.append(nav_temp)
            for i in range(action_point_index[4]-2,action_point_index[4]+1):
                place_temp = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"place",
                        "position":data[i]["agent_0_target"]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                }
                data_final_0.append(place_temp)
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
        except:
            continue
    return sample_info
def datatrans_2_end_sat_waypoint_closer_jumper(process_dir:str,skip_len:int,pick_place_sample_num=3,sample_clip=800,pick_place_skip = 3)  -> list:
    find_episode = []
    
    skip_len_start = skip_len
    process_dir_path = process_dir
    for folder_name in os.listdir(process_dir_path):
        json_path = os.path.join(process_dir_path,folder_name,"sum_data.json")
        if os.path.exists(json_path):
            with open(json_path,'r',encoding='utf-8') as f:
                data = json.load(f)
                if 'entities' in data:
                    entity_count = len(data['entities'])
                    if 5< entity_count < sample_clip-5:
                        find_episode.append(folder_name)
    temp_q = 0
    sample_info = []
    for name in find_episode:
        try:
            with open(os.path.join(process_dir_path,name,"data/data_trans.json"), 'r') as file:
                data = json.load(file)

            data_final_0 = []
            action = ["turn","nav_to_point","pick","turn","nav_to_point","place"]
            action_point_index = []
            i = 1
            result_agent_0 = []
            flag = 0
            late_action = data[0]["agent_0_action"]
            while i < len(data):
                if (data[i]["agent_0_action"] != late_action):
                    action_point_index.append(i)
                    late_action = data[i]["agent_0_action"]
                i+=1
            # print("action_point_index:",action_point_index)
            assert len(action_point_index) == 5,"Wrong episode"
            turn1 = {
                "step":1,
                "action":{
                    "name":"search_for_object_rec",
                    "position":None
                },
                "image":f"frame_1"+"_agent_0_head_rgbFetchRobot_head_rgb.png",
            }
            data_final_0.append(turn1)
            nav_1_point = [action_point_index[0]]
            i = action_point_index[0]
            skip_len = skip_len_start +random.randint(-3,3)
            while i +skip_len < action_point_index[1]:
                now_step = i
                if i+skip_len+14 >= action_point_index[1]:
                    i = action_point_index[1]
                else:
                    i +=skip_len
                skip_len = skip_len_start +random.randint(-3,3)
                test_step = i
                for a in range(now_step,test_step):
                    test_point = _3d_to_2d(matrix=data[now_step]["agent_0_martix"],
                                                 point_3d=data[test_step]["agent_0_now_worldloc"][:3])
                    x,y = test_point
                    if not (0 <= x < 256 and 0 <= y < 256):
                        test_step -=1
                    else:
                        break
                nav_1_point.append(test_step)
                i = test_step
            if nav_1_point[-1] != action_point_index[1]:
                nav_1_point.append(action_point_index[1])
            for i in range(len(nav_1_point)):
                if i+1< len(nav_1_point):
                    x,y = _3d_to_2d(matrix=data[nav_1_point[i]]["agent_0_martix"],
                                                 point_3d=data[nav_1_point[i+1]]["agent_0_now_worldloc"][:3])
                    x = limit_to_range(x)
                    y = limit_to_range(y)
                    nav_temp = {
                        "step":data[nav_1_point[i]]["step"],
                        "action":{
                            "name":"nav_to_point",
                            "position":[x,y]
                        },
                        "image":f"frame_"+str(data[nav_1_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                    }
                    data_final_0.append(nav_temp)
            for i in range(action_point_index[1]-(3*pick_place_skip),action_point_index[1]+1,pick_place_skip):
                pick_temp = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"pick",
                        "position":data[i]["agent_0_obj"]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                }
                data_final_0.append(pick_temp)
            turn2 = {
                "step":action_point_index[2],
                "action":{
                    "name":"search_for_goal_rec",
                    "position":None
                },
                "image":f"frame_"+str(action_point_index[2])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                "step_info":{}
            }
            data_final_0.append(turn2)
            nav_2_point = [action_point_index[3]]
            i = action_point_index[3]
            skip_len = skip_len_start +random.randint(-3,3)
            while i +skip_len < action_point_index[4]:
                now_step = i
                if i+skip_len+14 >= action_point_index[4]:
                    i = action_point_index[4]
                else:
                    i +=skip_len
                skip_len = skip_len_start +random.randint(-3,3)
                test_step = i
                for a in range(now_step,test_step):
                    test_point = _3d_to_2d(matrix=data[now_step]["agent_0_martix"],
                                                 point_3d=data[test_step]["agent_0_now_worldloc"][:3])
                    x,y = test_point
                    if not (0 <= x < 256 and 0 <= y < 256):
                        test_step -=1
                    else:
                        break
                nav_2_point.append(test_step)
                i = test_step
            if nav_2_point[-1] != action_point_index[4]:
                nav_2_point.append(action_point_index[4])
            for i in range(len(nav_2_point)):
                if i+1< len(nav_2_point):
                    x,y = _3d_to_2d(matrix=data[nav_2_point[i]]["agent_0_martix"],
                                                 point_3d=data[nav_2_point[i+1]]["agent_0_now_worldloc"][:3])
                    x = limit_to_range(x)
                    y = limit_to_range(y)
                    nav_temp = {
                        "step":data[nav_2_point[i]]["step"],
                        "action":{
                            "name":"nav_to_point",
                            "position": [x,y]
                            },
                        "image":f"frame_"+str(data[nav_2_point[i]]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                    }
                    data_final_0.append(nav_temp)
            for i in range(action_point_index[4]-(3*pick_place_skip),action_point_index[4]+1,pick_place_skip):
                place_temp = {
                    "step":data[i]["step"],
                    "action":{
                        "name":"place",
                        "position":data[i]["agent_0_target"]
                    },
                    "image":f"frame_"+str(data[i]["step"])+"_agent_0_head_rgbFetchRobot_head_rgb.png",
                }
                data_final_0.append(place_temp)
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
        except:
            continue
    return sample_info
    
if __name__ == "__main__":
    print(datatrans_2_end_single_agent_waypoint(process_dir="test_vlm_agent",skip_len=28))