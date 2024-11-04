import os
import multiprocessing
import subprocess
import time
from tqdm import tqdm
def run_episode_generator(args):
    # 获取当前进程的名称
    process_name = multiprocessing.current_process().name
    
    data_name,gpu_id,item,yaml_dir,output_dir = args
    # 生成基于进程名称的输出文件名
    output_file = f"./{output_dir}/{item}/{data_name}.json.gz"
    command = [
        "python",
        "./habitat-lab/habitat/datasets/rearrange/run_episode_generator.py",
        "--run",
        "--config", f"{yaml_dir}/{item}.yaml",
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
    sum_episode = 2000
    process_num = 30
    batch_num = 4
    gpu_num = 5
    num = int(sum_episode / batch_num)
    yaml_dir = "./new_scene_dir"
    output_dir = 'hssd_scene_13scene'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    zip_files = [f"data_{i}" for i in range(0,int(sum_episode/batch_num))]
    # scene_sample
    # 108294465_176709960
    # scene_sample = ["102344115","103997919_171031233","104348463_171513588","103997970_171031287",
    # "108736689_177263340","102344193","107733912_175999623"]
    scene_sample = ["104348463_171513588","103997970_171031287","108736689_177263340","102344193","107733912_175999623",
"102816786","103997643_171030747","105515211_173104185","102344115","103997919_171031233","102344457","102344529","104348202_171513150"]
    for item in scene_sample:
        if not os.path.exists(os.path.join(output_dir, item)):
            os.makedirs(os.path.join(output_dir, item))
        with multiprocessing.Pool(processes=process_num) as pool:
            args = [(f"data_{i}",int(i%gpu_num),item,yaml_dir,output_dir) for i in range(0,int(sum_episode/batch_num))]

            for _ in tqdm(pool.imap_unordered(run_episode_generator,args), total=num, desc=f"Processing {item} configs"):
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

