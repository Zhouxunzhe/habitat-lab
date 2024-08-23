import json
import gzip,os

with open('/home/lht/manipulation_eval_new.json','r') as file:
    data = json.load(file)
data_final = {"config": None, "episodes": []}
slect = [157, 134, 9, 52, 123, 32, 28, 145, 71, 54, 88, 111, 50, 18, 90, 34, 86, 60, 81, 159, 156, 13, 147, 139, 39, 151, 146, 25, 153, 161, 144, 115]
# for item in data["episodes"]:
#     if int(item["episode_id"]) in slect:
#         data_final["episodes"].append(item)

# with open('/home/lht/manipulation_eval_new.json','w') as file:
#     json.dump(data_final,file)
output_dir = '/home/lht/data_demo'
os.makedirs(output_dir, exist_ok=True)

# 分组并生成文件
for i in range(0, len(slect), 4):
    # 创建每个新文件的数据结构
    data_final = {"config": None, "episodes": []}
    selected_ids = slect[i:i+4]

    for item in data["episodes"]:
        if int(item["episode_id"]) in selected_ids:
            data_final["episodes"].append(item)

    # 文件名根据组号
    filename = f'{output_dir}/episodes_part_{i//4}.json.gz'
    
    # 写入 .gz 文件
    with gzip.open(filename, 'wt', encoding='utf-8') as gzfile:
        json.dump(data_final, gzfile)
