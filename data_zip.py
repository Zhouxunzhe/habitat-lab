import os
import gzip
import json
from pathlib import Path

# 输入和输出文件夹路径
input_folder = './data/datasets/sample_old'
output_folder = './data/datasets/sample'

# 创建输出文件夹（如果不存在）
Path(output_folder).mkdir(parents=True, exist_ok=True)
output_index = 0
# 遍历输入文件夹中的所有文件
for i in range(5000):
    input_file_path = os.path.join(input_folder, f'process_{i}.json.gz')

    # 打开并读取压缩的 JSON 文件
    with gzip.open(input_file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    # 拆分 "episodes" 为两组
    episodes = data['episodes']
    split_1 = episodes[0]
    split_2 = episodes[1]
    split_3 = episodes[2]
    split_3 = episodes[3]
    # 创建两个新 JSON 对象
    data_1 = {'config': None, 'episodes': split_1}
    data_2 = {'config': None, 'episodes': split_2}
    data_3 = {'config': None, 'episodes': split_3}
    data_4 = {'config': None, 'episodes': split_4}
    # 写入新的压缩文件
    output_file_1_path = os.path.join(output_folder, f'process_{output_index}.json.gz')
    output_index +=1
    output_file_2_path = os.path.join(output_folder, f'process_{output_index}.json.gz')
    output_index +=1
    output_file_3_path = os.path.join(output_folder, f'process_{output_index}.json.gz')
    output_index +=1
    output_file_4_path = os.path.join(output_folder, f'process_{output_index}.json.gz')
    output_index +=1
    with gzip.open(output_file_1_path, 'wt', encoding='utf-8') as f:
        json.dump(data_1, f)

    with gzip.open(output_file_2_path, 'wt', encoding='utf-8') as f:
        json.dump(data_2, f)

print("拆分完成！")