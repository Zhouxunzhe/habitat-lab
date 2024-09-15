import multiprocessing
import subprocess
import time
from tqdm import tqdm
import dp_config
from ruamel.yaml import YAML
from find_success import datatrans_2_end,datatrans_2_end_single_agent_waypoint,datatrans_2_end_single_agent_objectcentric
import os
from datatrans_batch import process_directory
import shutil
yaml = YAML()
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
    file_path,skip_len, base_directory,gpu_id = args
    a = ["disk"]
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
        f"habitat_baselines.image_dir=video_dir/{file_path}",
        f"habitat.dataset.data_path=data/datasets/hssd_demo/{file_path}"
    ]
    log_file = f"./log/example_{file_path}.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(1)
    process_directory(os.path.join(base_directory,file_path),skip_len=skip_len)
    print("finish_trans")
    sample_info = datatrans_2_end_single_agent_objectcentric(process_dir=file_path,skip_len = skip_len)
    if not sample_info:
        print("no sample_info,Exiting process.")
        return 
    sample = str(sample_info)
    print(f"{file_path}'s sample:",sample)
    llm_yaml_path = './habitat-baselines/habitat_baselines/config/social_rearrange/llm_fetch_stretch_manipulation.yaml'
    with open(llm_yaml_path,'r') as file:
        data = yaml.load(file)
    data['habitat_baselines']['eval']['episode_stored'] = sample_info
    with open(f'./habitat-baselines/habitat_baselines/config/override/{file_path}.yaml','w') as file:
        yaml.dump(data,file)
    cmd = [
        "python", "-u", "-m", "habitat_baselines.run",
        f"--config-name=override/{file_path}.yaml",
        "habitat_baselines.evaluate=True",
        "habitat_baselines.num_environments=1",
        f"habitat_baselines.eval.json_option=[]",
        f"habitat_baselines.eval.video_option={a}",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={gpu_id}",
        f"habitat_baselines.torch_gpu_id={gpu_id}",
        f"habitat_baselines.eval.video_option_new=False",
        f"habitat_baselines.image_dir=video_dir/{file_path}",
        f"habitat_baselines.eval.image_option={a}",
        # f"habitat_baselines.eval.episode_stored={sample_info}",
        f"habitat.dataset.data_path=data/datasets/hssd_demo/{file_path}"
    ]
    log_file = f"./log/example_new_{file_path}.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(1)

if __name__ == "__main__":
    clear_directory('./habitat-baselines/habitat_baselines/config/override/')
    sum_episode = dp_config.sum_episode
    epnum_per_gz = dp_config.epnum_per_gz
    gz_start = dp_config.gz_start
    num_gz = int((sum_episode/epnum_per_gz)-gz_start)
    skip_len = dp_config.skip_len
    base_directory = dp_config.base_directory
    zip_files = [(f"data_{i}.json.gz",int(i%2)) for i in range(gz_start,gz_start+num_gz)]
    process_num = dp_config.process_num
    total_batches = len(zip_files)//process_num
    with multiprocessing.Pool(processes=process_num) as pool:
        args = [(file_path, skip_len, base_directory,gpu_id) for file_path,gpu_id in zip_files]
        for _ in tqdm(pool.imap_unordered(run_script, args), total=len(zip_files), desc="Process Files"):
            pass
    # with tqdm(total=len(zip_files),desc = "Processing Files") as pbar: 
    #     for i in range(0, len(zip_files), batch_size):
    #         batch = zip_files[i:i + batch_size]  
    #         processes = []
    #         start_time = time.time()

    #         for file_path in batch:
    #             p = multiprocessing.Process(target=run_script, args=(file_path,))
    #             processes.append(p)
    #             p.start()
    #         for p in processes:
    #             p.join()
    #         end_time = time.time()
    #         batch_time = end_time-start_time
    #         print(f"Batch {i//batch_size + 1} completed.")
    #         pbar.update(len(batch))

    #         time.sleep(1.5)  


