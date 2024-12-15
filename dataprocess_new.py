
import multiprocessing
import subprocess
import time
from tqdm import tqdm
import dp_config
from ruamel.yaml import YAML
from find_success import datatrans_2_end,datatrans_2_end_single_agent_waypoint,datatrans_2_end_single_agent_objectcentric,datatrans_2_end_sat_waypoint_closer
import os
from datatrans_batch import process_directory
import shutil,random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED,TimeoutError
yaml = YAML()
import zipfile
import concurrent.futures
from tqdm import tqdm
import time
import gzip
import shutil
import os,pdb
import time
from threading import Timer
import json
import argparse
def terminate_pool(pool):
    if pool is not None:
        pool.terminate()
        pool.close()
        pool.join()
        print("Terminated pool due to timeout.")
class SkipCurrentThread(Exception):
    pass
def run_with_timeout(func, *args, timeout=2, process_dir=None, skip_len=40, sample_clip=500, retries=1, data_name=None):
    for attempt in range(retries):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, process_dir=process_dir, skip_len=skip_len, sample_clip=sample_clip)
            try:
                result = future.result(timeout=timeout)
                return result
            # except TimeoutError:
            #     print(f"Timeout occurred: {data_name}")
            #     return -1
            # except Exception as e:
            #     print(f"Error occurred: {e}: {data_name}")
            except:

                return -1
    # print(f"Max retries--{data_name}.")
    
def unzip_gz_file(gz_file_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    output_file_path = os.path.join(extract_to, os.path.basename(gz_file_path).replace('.gz', ''))
    with gzip.open(gz_file_path, 'rb') as gz_file:
        with open(output_file_path, 'wb') as out_file:
            shutil.copyfileobj(gz_file, out_file)
def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def run_script(args):
    file_path,skip_len, base_directory,gpu_id,scene,scene_dataset_dir = args
    
    a = ["disk"]
    seed = random.randint(1, 1000000)
    cmd = [
        "python", "-u", "-m", "habitat_baselines.run",
        "--config-name=social_rearrange/llm_fetch_stretch_manipulation.yaml",
        "habitat_baselines.evaluate=True",
        "habitat_baselines.num_environments=1",
        f"habitat_baselines.eval.json_option={a}",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={gpu_id}",
        f"habitat_baselines.torch_gpu_id={gpu_id}",
        f"habitat_baselines.eval.video_option={a}",
        f"habitat_baselines.eval.video_option_new=False",
        f"habitat_baselines.video_dir={base_directory}/video_dir_",
        f"habitat_baselines.image_dir={base_directory}/{scene}/{file_path}",
        f"habitat.seed={seed}",
        f"habitat.environment.max_episode_steps={max_step}",
        f"habitat.dataset.data_path=data/datasets/{scene_dataset_dir}/{scene}/{file_path}"
    ]
    log_file = f"./log/{scene}_example_{file_path}.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.7)
    process_directory(os.path.join(base_directory,scene,file_path),skip_len=skip_len)
    print("finish_trans",flush=True)
    # sample_info = datatrans_2_end_single_agent_waypoint(process_dir=f"{scene}/{file_path}",skip_len = skip_len,sample_clip=max_step)
    # sample_info = run_with_timeout(datatrans_2_end_single_agent_waypoint, process_dir=f"{scene}/{file_path}", skip_len=skip_len, sample_clip=max_step,timeout=3,retries=2,data_name=file_path)
    flag = 0
    with ThreadPoolExecutor() as executor:
        future_set = executor.submit(datatrans_2_end_sat_waypoint_closer, process_dir=f"./{base_directory}/{scene}/{file_path}", skip_len=skip_len, sample_clip=max_step)
        try:
            result = future_set.result(timeout=10)
            sample_info = result
        except:
            # print(f"break:{file_path}",flush=True)
            return (False, -1)
    if len(sample_info)==0 or sample_info == -1:
        print(f"no sample_info,Exiting process,from{file_path}")
        return (False, 0)
    sample = str(sample_info)
    print(f"{file_path}'s sample:",sample)
    zxz_yaml_path = '/mnt/workspace/habitat-lab/habitat-lab/habitat/config/benchmark/single_agent/zxz_fetch_sample.yaml'
    with open(zxz_yaml_path,'r') as file:
        data = yaml.load(file)
    data['habitat']['dataset']['data_path'] = f"data/datasets/{scene_dataset_dir}/{scene}/{file_path}"
    data['habitat']['simulator']['habitat_sim_v0']['gpu_device_id'] = gpu_id
    relative_yaml_dir_path = os.path.join(f"sample_frame_dataprocess_{base_dir_name}",scene)
    scene_yaml_dir_path = os.path.join('./habitat-lab/habitat/config/benchmark/single_agent/override',relative_yaml_dir_path)
    os.makedirs(scene_yaml_dir_path,exist_ok=True)
    yaml_path = os.path.join(scene_yaml_dir_path,f"{file_path}.yaml")
    with open(yaml_path,'w') as file:
        yaml.dump(data,file)
    obs_key = "arm_workspace_rgb"
    sample_info_str = json.dumps(sample_info)
    command = [
        "python",
        "-u",
        "habitat-lab/habitat/datasets/rearrange/generate_episode_graph_images.py",
        "--config", f"benchmark/single_agent/override/sample_frame_dataprocess_{base_dir_name}/{scene}/{file_path}.yaml",
        "--gpu_id", f"{gpu_id}",
        "--output_dir", f"{base_directory}/{scene}/{file_path}",
        "--generate_type", "sample_frame",
        "--obs_keys",obs_key,
        "--sample_info", sample_info_str
    ]
    
    log_file = f"./log/sampleframe_{scene}_example_{file_path}.log"
    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.5)
    return (True, 1)

if __name__ == "__main__":
    # clear_directory('./habitat-baselines/habitat_baselines/config/override/')
    # clear_directory('./log')
    current_time = time.time()
    local_random = random.Random(current_time)
    random_number = local_random.randint(1, 1000000)
    max_step = dp_config.max_step
    scene_dataset_dir = dp_config.scene_dataset_dir
    base_directory = f"./{dp_config.base_directory}_{random_number}"
    base_dir_name = f"{dp_config.base_directory}_{random_number}"
    sum_episode = dp_config.sum_episode
    epnum_per_gz = dp_config.epnum_per_gz
    gz_sum = int(sum_episode/epnum_per_gz)
    # jump_gz = dp_config.jump_gz
    repeat_time = dp_config.repeat_time
    skip_len = dp_config.skip_len
    gpu_num = dp_config.gpu_num
    # base_directory = './video_dir'
    for it in range(gz_sum):
        # print("it:",it)
        for scene in dp_config.sample_scene:
            sum_episode = dp_config.sum_episode
            # epnum_per_gz = dp_config.epnum_per_gz
            # gz_start = dp_config.gz_start
            gpu_num = dp_config.gpu_num
            # num_gz = int((sum_episode/epnum_per_gz)-gz_start)
            gz_start_gz = dp_config.gz_start
            gz_end = gz_sum
            process_num = dp_config.process_num
            gz_start = process_num*it
            num_gz = int((sum_episode / epnum_per_gz) - gz_start_gz)
            zip_files = [(f"data_{i}.json.gz", int(i % gpu_num)) for i in range(gz_start_gz+gz_start,gz_start_gz+gz_start+process_num)]
            # print(f"Processing ",zip_files)
            print(f"START--{scene}--{it*process_num}/{num_gz}--dir:{dp_config.base_directory}_{random_number}")
            start_time = time.time()
            with multiprocessing.Pool(processes=process_num) as pool:
                args = [(file_path, skip_len, base_directory, gpu_id, scene, scene_dataset_dir) for file_path, gpu_id in zip_files]
                results = []
                async_results = [pool.apply_async(run_script, (arg,), error_callback=lambda e: print(f"Error: {e}")) for arg in args]

                # Set up a timer to terminate the pool after the timeout
                timer = Timer(dp_config.timeout, terminate_pool, [pool])
                timer.start()
                try:
        # Wait for all tasks to complete or timeout
                    pool.close()
                    pool.join()
                finally:
                    timer.cancel()
            end_time = time.time()
            print(f"FINISH--{scene}--{it*process_num}/{num_gz}----dir:{dp_config.base_directory}_{random_number}--usetime:{end_time-start_time}")
            