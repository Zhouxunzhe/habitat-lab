import os
import re

# 获取当前文件夹下所有文件名
import argparse
import os.path as osp

parser = argparse.ArgumentParser()
# necessary arguments
parser.add_argument(
    "--dir",
    type=str,
    default='',
    help="Relative path to video path.",
)
parser.add_argument(
    "--num",
    type=str,
    default='',
    help="Relative path to video path.",
)
parser.add_argument(
    "--test_num",
    type=int,
    default=34,
    help="number supposed to be tested.",
)
args, _ = parser.parse_known_args()
dir_name = args.dir
file_names = os.listdir(dir_name)

# 正则表达式匹配文件名中的 episode 和 pddl_success 部分
episode_pattern = r"episode=([0-9]+)"
success_pattern = r"pddl_success=([0-9]*\.?[0-9]+)"

# 初始化存储成功和失败的 episode 标号列表
success = []
fail = []

# 遍历文件名，提取成功率和 episode 标号
for file_name in file_names:
    episode_match = re.search(episode_pattern, file_name)
    success_match = re.search(success_pattern, file_name)
    if episode_match and success_match:
        episode = episode_match.group(1)
        success_rate = float(success_match.group(1))
        if success_rate > 0:
            success.append(episode)
        else:
            fail.append(episode)

# 输出成功和失败的 episode 标号
# if success:
#     print(f"{dir_name} 文件夹下成功的 episode 标号为: {success}")
# else:
#     print(f"{dir_name} 文件夹下没有找到成功的 episode。")
# if fail:
#     print(f"{dir_name} 文件夹下失败的 episode 标号为: {fail}")
# else:
#     print(f"{dir_name} 文件夹下没有找到失败的 episode。")

print(f"{dir_name} success rate: {len(success)} / {len(success)+len(fail)}")

