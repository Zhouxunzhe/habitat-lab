#!/bin/bash

python /home/lht/habitat-lab/habitat-lab/habitat/datasets/rearrange/run_episode_generator.py --run \
--config data/hssd_ycb_multi_goal.yaml \
--num-episodes 1 \
--out data/datasets/ppp/policy_hssd.json.gz \
--type manipulation \
--resume habitat-mas/habitat_mas/data/robot_resume/FetchRobot_default.json
