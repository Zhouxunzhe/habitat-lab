import os
import multiprocessing
import subprocess
import time
from tqdm import tqdm
def run_episode_generator(args):
    # 获取当前进程的名称
    process_name = multiprocessing.current_process().name
    
    data_name,gpu_id = args
    # 生成基于进程名称的输出文件名
    output_file = f"data/datasets/hssd_demo/{data_name}.json.gz"
    command = [
        "python",
        "./habitat-lab/habitat/datasets/rearrange/run_episode_generator.py",
        "--run",
        "--config", "data/hssd_dataset.yaml",
        "--gpu_id", f"{gpu_id}",
        "--num-episodes", f"{batch_num}",
        "--out", f"{output_file}",
        "--type", "manipulation",
        "--resume", "habitat-mas/habitat_mas/data/robot_resume/FetchRobot_default.json"
    ]
    log_file = f"./log/sample/{data_name}.log"
    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.9)

if __name__ == '__main__':
    sum_episode = 400
    process_num = 30
    batch_num = 4
    num = int(sum_episode / batch_num)
    zip_files = [f"data_{i}" for i in range(0,int(sum_episode/batch_num))]
    with multiprocessing.Pool(processes=process_num) as pool:
        args = [(f"data_{i}",int(i%2)) for i in range(0,int(sum_episode/batch_num))]
        for _ in tqdm(pool.imap_unordered(run_episode_generator,args), total=num, desc="Process Files"):
            pass
    # with tqdm(total=num,desc="Sample_Episode") as pbar:
    #     for j in range(0,num):
    #         start_time = time.time()
    #         processes = []
    #         for i in range(process_num):  
    #             process_name = f"process_{i + (process_num*j)}"
    #             p = multiprocessing.Process(target=run_episode_generator, name=process_name)
    #             p.start()
    #             processes.append(p)
            
    #         for p in processes:
    #             p.join()
            
    #         end_time = time.time()
    #         batch_time = end_time-start_time
    #         print(f"Batch {j} completed.")
    #         pbar.update(1)
    #         time.sleep(0.5) 

