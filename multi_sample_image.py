import multiprocessing
import subprocess
import time,yaml
from tqdm import tqdm
from ruamel.yaml import YAML
from find_success import datatrans_2_end,datatrans_2_end_single_agent_waypoint
import os
from datatrans_batch import process_directory
yaml = YAML()
# sample_info = [{'episode_id': 27, 'sample_frame': [[1, 0], [1, 1], [21, 0], [21, 1], [41, 0], [41, 1], [43, 0], [43, 1], [55, 0], [55, 1], [145, 0], [145, 1], [157, 0], [157, 1], [165, 0], [165, 1], [177, 0], [177, 1], [185, 0], [185, 1], [197, 0], [197, 1], [205, 0], [205, 1], [217, 0], [217, 1], [225, 0], [225, 1], [237, 0], [237, 1], [245, 0], [245, 1], [257, 0], [257, 1], [265, 0], [265, 1], [277, 0], [277, 1], [280, 0], [280, 1]]}, {'episode_id': 64, 'sample_frame': [[1, 0], [1, 1], [21, 0], [21, 1], [41, 0], [41, 1], [61, 0], [61, 1], [81, 0], [81, 1], [87, 0], [87, 1], [101, 0], [101, 1], [121, 0], [121, 1], [140, 0], [140, 1], [189, 0], [189, 1], [209, 0], [209, 1], [229, 0], [229, 1], [242, 0], [242, 1], [249, 0], [249, 1], [262, 0], [262, 1], [269, 0], [269, 1], [282, 0], [282, 1], [289, 0], [289, 1], [302, 0], [302, 1], [309, 0], [309, 1], [322, 0], [322, 1], [329, 0], [329, 1], [342, 0], [342, 1], [349, 0], [349, 1], [362, 0], [362, 1], [360, 0], [360, 1]]}, {'episode_id': 102, 'sample_frame': [[1, 0], [1, 1], [21, 0], [21, 1], [38, 0], [38, 1], [41, 0], [41, 1], [61, 0], [61, 1], [81, 0], [81, 1], [101, 0], [101, 1], [121, 0], [121, 1], [140, 0], [140, 1], [141, 0], [141, 1], [160, 0], [160, 1], [161, 0], [161, 1], [180, 0], [180, 1], [181, 0], [181, 1], [200, 0], [200, 1], [201, 0], [201, 1], [209, 0], [209, 1], [220, 0], [220, 1], [240, 0], [240, 1], [260, 0], [260, 1], [280, 0], [280, 1], [300, 0], [300, 1], [311, 0], [311, 1], [320, 0], [320, 1], [331, 0], [331, 1], [340, 0], [340, 1], [351, 0], [351, 1], [360, 0], [360, 1], [371, 0], [371, 1], [380, 0], [380, 1], [391, 0], [391, 1], [391, 0], [391, 1]]}, {'episode_id': 136, 'sample_frame': [[1, 0], [1, 1], [21, 0], [21, 1], [41, 0], [41, 1], [61, 0], [61, 1], [81, 0], [81, 1], [86, 0], [86, 1], [97, 0], [97, 1], [188, 0], [188, 1], [199, 0], [199, 1], [208, 0], [208, 1], [219, 0], [219, 1], [228, 0], [228, 1], [239, 0], [239, 1], [248, 0], [248, 1], [259, 0], [259, 1], [268, 0], [268, 1], [279, 0], [279, 1], [288, 0], [288, 1], [299, 0], [299, 1], [308, 0], [308, 1], [319, 0], [319, 1], [328, 0], [328, 1], [334, 0], [334, 1]]}]
def run_script(args):
    file_path,skip_len,base_directory = args
    a = ["disk"]
    process_directory(os.path.join(base_directory,file_path),skip_len=skip_len)
    print("finish_trans")
    sample_info = datatrans_2_end_single_agent_waypoint(process_dir=file_path,skip_len = skip_len)
    if not sample_info:
        print("no sample_info,Exiting porcess.")
        return 
    sample = str(sample_info)
    print("sample:",sample)
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
        f"habitat_baselines.eval.video_option_new=False",
        f"habitat_baselines.image_dir=video_dir/{file_path}",
        f"habitat_baselines.eval.image_option={a}",
        # f"habitat_baselines.eval.episode_stored={sample_info}",
        f"habitat.dataset.data_path=data/datasets/sample/{file_path}"
    ]
    log_file = f"./log/example_new_{file_path}.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.5)

if __name__ == "__main__":
    sum_episode = 1500
    batch_per_zip = 2
    start_gz = 0
    num_gz = int(sum_episode/batch_per_zip-start_gz)
    skip_len = 40
    base_directory = './video_dir/'
    zip_files = [f"process_{i}.json.gz" for i in range(start_gz,start_gz+num_gz)]
    batch_size = 8
    total_batches = len(zip_files)//batch_size
    with multiprocessing.Pool(processes=batch_size) as pool:
        args = [(file_path, skip_len, base_directory) for file_path in zip_files]
        for _ in tqdm(pool.imap_unordered(run_script, args), total=len(zip_files), desc="Processing Files"):
            pass
    # with tqdm(total=len(zip_files),desc = "Processing Files") as pbar: 
    #     for i in range(0, len(zip_files), batch_size):
    #         batch = zip_files[i:i + batch_size]  
    #         processes = []
    #         start_time = time.time()

    #         for file_path in batch:
    #             p = multiprocessing.Process(target=run_script, args=(file_path,skip_len,))
    #             processes.append(p)
    #             p.start()


    #         for p in processes:
    #             p.join()
    #         end_time = time.time()
    #         batch_time = end_time-start_time
    #         print(f"Batch {i//batch_size + 1} completed.")
    #         pbar.update(len(batch))

    #         time.sleep(1.5)  