#python habitat-lab/habitat/datasets/rearrange/run_hssd_episode_generator.py --run --config /home/lgtm/zhouxunzhe/habitat-zxz/data/config/hssd/hssd_dataset.yaml --num-episodes 340 --out data/datasets/hssd_dist.json.gz --type distance
#
#python habitat-lab/habitat/datasets/rearrange/run_hssd_episode_generator.py --run --config /home/lgtm/zhouxunzhe/habitat-zxz/data/config/hssd/hssd_dataset.yaml --num-episodes 340 --out data/datasets/hssd_height.json.gz --type height

#python habitat-baselines/habitat_baselines/run.py --config-name=multi_rearrange/zxz_llm_dist_man.yaml habitat_baselines.evaluate=True habitat_baselines.num_environments=1

#python habitat-baselines/habitat_baselines/run.py --config-name=multi_rearrange/zxz_llm_dist_man_reverse.yaml habitat_baselines.evaluate=True habitat_baselines.num_environments=1

#python habitat-baselines/habitat_baselines/run.py --config-name=multi_rearrange/zxz_llm_height_man.yaml habitat_baselines.evaluate=True habitat_baselines.num_environments=1

python habitat-baselines/habitat_baselines/run.py --config-name=multi_rearrange/zxz_llm_height_man_reverse.yaml habitat_baselines.evaluate=True habitat_baselines.num_environments=1

#python habitat-baselines/habitat_baselines/run.py --config-name=multi_rearrange/zxz_llm_height_per.yaml habitat_baselines.evaluate=True habitat_baselines.num_environments=1

#python habitat-baselines/habitat_baselines/run.py --config-name=multi_rearrange/zxz_llm_height_per_reverse.yaml habitat_baselines.evaluate=True habitat_baselines.num_environments=1
