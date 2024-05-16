# Habitat-MAS

Habitat-MAS is a Python package for Multi-Agent Systems in Habitat virtual environments.

## Table of Contents
- [Habitat-MAS](#habitat-mas)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Download Data](#download-data)
    - [Run the demo](#run-the-demo)

## Installation

To install the package, you can use the following command under habitat-lab root directory:

```sh
pip install -e habitat-mas
```

[Not available] Besides, you need to install the crab for multi-agent system. 

## Usage

To run the demo, you need to first install [habitat-lab](../README.md) and [habitat-baselines](../habitat-baselines/) following the normal habitat-lab installation guide.

### Download Data
The dataset used in the demo is the same as [Habitat-3.0 Multi-Agent Training](../habitat-baselines/README.md#habitat-30-multi-agent-training). You can download the dataset by running the following command:

```sh
python -m habitat_sim.utils.datasets_download --uids hssd-hab hab3-episodes habitat_humanoids hab_spot_arm hab3-episodes ycb hssd-hab hab3_bench_assets rearrange_task_assets
```

Besides, you should:
- download a drone urdf to insert into the environment. You can download the urdf from [here](https://drive.google.com/file/d/1WO4yUQaZRvlYcDY-A0ukNjOWmVg2nWEd/view?usp=sharing). Please place this dji_drone folder in data/robots folder that can be found by habitat project.
- download the spot animation data from [here](https://drive.google.com/file/d/1E_mW3PEWwgYbt3rzSJcA4DbsI-WTQUmF/view?usp=sharing), and place it into the data/robots/spot_data folder. 
- download the modified spot robot model from [here](https://drive.google.com/file/d/10idYOK2CAKgwWu9g2M4WgyuZBRSdYER1/view?usp=sharing). You should decompress it and place it into the data/robots folder.

The folder should look like this:
```
habitat-lab
├── data
│   ├── robots
│       ├── dji_drone
│           ├── meshes
│           ├── urdf
│       ├── hab_fetch
│       ├── hab_spot_arm
|           ├── meshesDae
│           ├── urdf
|               ├── hab_spot_onlyarm_dae.urdf
│               ...
│       ├── hab_stretch
│       ├── spot_data
│           ├── spot_walking_trajectory.csv
...
```

### Run the demo

The demo is adapted from [Habitat-3.0 Social Rearrangement](../habitat-baselines/README.md#social-rearrangement). You can run the demo by running the following command:

```sh
# Under the habitat-lab root directory
python -u -m habitat_baselines.run \
  --config-name=social_rearrange/llm_spot_drone.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```