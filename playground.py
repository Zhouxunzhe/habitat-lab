import json
import re

# # 文件路径
# file_path = '/home/lht/habitat-lab/data/robots/json/manipulation.json'

# # 读取 JSON 文件
# with open(file_path, 'r') as file:
#     data = file.read()

# # 使用正则表达式提取所有的数字键
# matches = re.findall(r'"(\d+)":', data)

# # 打印所有匹配到的数字键
# # print(matches)
# file_path = '/home/lht/habitat-lab/manipulation_eval_new.json'
# item = 0
# # 读取 JSON 文件
# with open(file_path, 'r') as file:
#     data = json.load(file)

# # 假设你想匹配的值大于当前 "episode_id"，并替换它
# for episode in data['episodes']:
#     episode_id = int(episode["episode_id"])
    
#     # 假设 match 是大于当前 episode_id 的值（这里假设加 10 ）
#     new_episode_id = str(episode_id + 10)
    
#     # 替换 episode_id 为 match 到的值
#     episode["episode_id"] = matches[item]
#     item+=1

# # 将修改后的数据保存回 JSON 文件
# output_file_path = '/home/lht/habitat-lab/manipulation_modified.json'
# with open(output_file_path, 'w') as outfile:
#     json.dump(data, outfile, indent=4)

# import os
# import json

# # 设置JSON文件所在的目录和文件夹所在的目录
# json_dir = '/home/lht/habitat-lab/video_dir/process_0.json.gz/'  # JSON文件夹路径
# folder_dir = '/home/lht/habitat-lab/video_dir/process_0.json.gz/'  # 对应的图片文件夹路径
# # 遍历JSON文件
# for json_file in os.listdir(json_dir):
#     if json_file.endswith('.json'):
#         json_path = os.path.join(json_dir, json_file)
        
#         # 加载JSON文件
#         with open(json_path, 'r') as f:
#             data = json.load(f)
        
#         # 如果data是列表，遍历其中的每个元素
#         if isinstance(data, list):
#             for item in data:
#                 # 获取JSON文件对应的文件夹名称
#                 folder_name = os.path.splitext(json_file)[0]
#                 folder_path = os.path.join(folder_dir, folder_name)

#                 # 检查文件夹是否存在
#                 if os.path.exists(folder_path) and os.path.isdir(folder_path):
#                     # 获取图片名称
#                     image_0 = item.get("image_0")
#                     image_1 = item.get("image_1")
                    
#                     # 检查图片是否存在于对应的文件夹中
#                     image_0_exists = os.path.exists(os.path.join(folder_path, image_0)) if image_0 else False
#                     image_1_exists = os.path.exists(os.path.join(folder_path, image_1)) if image_1 else False
#                     if image_0_exists and image_1_exists:
#                         print(folder_name)
#                     # 输出结果
#                     if image_0_exists!=1 or image_1_exists!=1:
                        
#                         print(f"Missing image(s) for {json_file}:")
#                         if not image_0_exists and image_0:
#                             print(f" - {image_0} not found")
#                         if not image_1_exists and image_1:
#                             print(f" - {image_1} not found")
#                 else:
#                     print(f"Folder {folder_name} does not exist")
#         else:
#             print(f"Unexpected data structure in {json_file}")


#这是用来看item是什么类型的代码
import json
import csv
json_path = '/home/lht/habitat-lab/data/scene_datasets/hssd-hab/scenes/108294465_176709960.scene_instance.json'
csv_path = '/home/lht/habitat-lab/data/scene_datasets/hssd-hab/semantics/objects.csv'

with open(json_path,'r') as f:
    json_data = json.load(f)

csv_data = {}
with open(csv_path,'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_data[row['id']] = row['name']
result = []
for obj in json_data["object_instances"]:
    template_name = obj["template_name"]
    if template_name in csv_data:
        name = csv_data[template_name]
        result.append({"template_name": template_name, "name": name})

# 输出匹配结果
for item in result:
    print(item)

#这是用来看episode_id能匹配啥manipulation的代码
# import json

# # 读取 JSON 文件
# file_path = '/home/lht/habitat-lab/data/robots/json/manipulation.json'

# with open(file_path, 'r') as file:
#     data = json.load(file)
# result = []
# # 提取所有数字键对应的数据
# for key, value in data.items():
#     if key.isdigit():
#         result.append(int(key))
# result.sort()
# print(result)