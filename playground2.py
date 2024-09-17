import os
import gzip
import shutil

def unzip_gz_files_in_directory(base_directory):
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.gz'):
                gz_file_path = os.path.join(root, file)
                output_file_path = os.path.join(root, file[:-3])  # 去掉 .gz 后缀

                # 解压缩 .gz 文件
                with gzip.open(gz_file_path, 'rb') as gz_file:
                    with open(output_file_path, 'wb') as out_file:
                        shutil.copyfileobj(gz_file, out_file)

# 示例用法
base_directory = './data/datasets/hssd_scene'  # 替换为包含文件夹的基目录
unzip_gz_files_in_directory(base_directory)
print("finish!")