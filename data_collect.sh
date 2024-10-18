#!/bin/bash

python ./habitat-lab/habitat/datasets/rearrange/run_episode_generator.py --run \
--config data/hssd_dataset/scene_graph_sample/107733912_175999623.yaml \
--num-episodes 1 \
--gpu_id 1 \
--out data/datasets/test_scene_graph/107733912_175999623test.json.gz \
--type manipulation \
--resume habitat-mas/habitat_mas/data/robot_resume/FetchRobot_default.json
