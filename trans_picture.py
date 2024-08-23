import os
import shutil

def extract_images_with_keyword(src_path, keyword="agent_0", dest_folder="image"):
    # 创建目标文件夹
    dest_path = os.path.join(src_path, dest_folder)
    os.makedirs(dest_path, exist_ok=True)
    
    # 支持的图片扩展名
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
    
    # 遍历源路径下的所有文件
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if keyword in file and file.lower().endswith(image_extensions):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, file)
                
                # 复制文件到目标文件夹
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} to {dest_file}")

# 示例用法
src_path = "/home/lht/habitat-lab/video_dir/image_dir/episode_88/"  # 替换为实际的源路径
extract_images_with_keyword(src_path)

