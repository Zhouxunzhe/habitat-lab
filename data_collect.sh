#!/bin/bash

num_epochs=1
for (( i=1; i<=num_epochs; i++ )); do
    echo "Running $i"
    python ./habitat-lab/habitat/datasets/rearrange/run_episode_generator.py --run \
    --config habitat-lab/habitat/datasets/rearrange/configs/mp3d.yaml \
    --num-episodes 1 \
    --gpu_id 0 \
    --out data/datasets/test_mp3d/$i.json.gz \
    --type manipulation \
    --resume habitat-mas/habitat_mas/data/robot_resume/FetchRobot_default.json
done
