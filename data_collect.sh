#!/bin/bash

python ./habitat-lab/habitat/datasets/rearrange/run_episode_generator.py --run \
--config data/hssd_dataset.yaml \
--num-episodes 2 \
--gpu_id 1 \
--out data/datasets/test/policy_hssd.json.gz \
--type manipulation \
--resume habitat-mas/habitat_mas/data/robot_resume/FetchRobot_default.json
