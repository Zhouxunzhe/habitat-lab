import multiprocessing
import subprocess
import time
from tqdm import tqdm
def run_script(file_path):
    a = ["disk"]
    cmd = [
        "python", "-u", "-m", "habitat_baselines.run",
        "--config-name=social_rearrange/llm_fetch_stretch_manipulation.yaml",
        "habitat_baselines.evaluate=True",
        "habitat_baselines.num_environments=1",
        f"habitat_baselines.eval.json_option={a}",
        f"habitat_baselines.eval.video_option={a}",
        f"habitat_baselines.eval.video_option_new=False",
        f"habitat_baselines.image_dir=video_dir/{file_path}",
        f"habitat.dataset.data_path=data/datasets/sample/{file_path}"
    ]
    
    log_file = f"./log/example_{file_path}.log"

    
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    sum_episode = 20000
    batch_per_zip = 4
    start_gz = 180
    num_gz = int(sum_episode/batch_per_zip-start_gz)
    
    zip_files = [f"process_{i}.json.gz" for i in range(start_gz,start_gz+num_gz)]
    batch_size = 10
    total_batches = len(zip_files)//batch_size
    with tqdm(total=len(zip_files),desc = "Processing Files") as pbar: 
        for i in range(0, len(zip_files), batch_size):
            batch = zip_files[i:i + batch_size]  
            processes = []
            start_time = time.time()

            for file_path in batch:
                p = multiprocessing.Process(target=run_script, args=(file_path,))
                processes.append(p)
                p.start()


            for p in processes:
                p.join()
            end_time = time.time()
            batch_time = end_time-start_time
            print(f"Batch {i//batch_size + 1} completed.")
            pbar.update(len(batch))

            time.sleep(1.5)  