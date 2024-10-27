#!/bin/bash

# 检查是否提供了目录参数
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# 指定要检查的目录
dir="$1"

# 检查目录是否存在
if [ ! -d "$dir" ]; then
  echo "Error: Directory '$dir' does not exist."
  exit 1
fi

# 支持的图片文件扩展名
extensions=("jpg" "png")

# 初始化计数器
image_count=0

# 遍历所有支持的扩展名
for ext in "${extensions[@]}"; do
  # 计算每种扩展名的文件数量
  count=$(find "$dir" -type f -iname "*.$ext" | wc -l)
  image_count=$((image_count + count))
done

# 输出结果
echo "Total number of images in '$dir': $image_count"