from habitat.datasets.rearrange.rearrange_dataset import Rearrange3DEpisode, RearrangeEpisode
import habitat_sim
import magnum as mn
import warnings
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import DictConfig
import numpy as np
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from omegaconf import OmegaConf
import gzip
import json
import git, os

def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    
    # Set up an example scene
    sim_cfg.scene = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json")
    sim_cfg.scene_dataset = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json")
    sim_cfg.additional_object_paths = [os.path.join(data_path, 'objects/ycb/configs/')]

    
    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg


def init_rearrange_sim(agent_dict):
    # Start the scene config
    sim_cfg = make_sim_cfg(agent_dict)    
    cfg = OmegaConf.create(sim_cfg)
    
    # Create the scene
    sim = RearrangeSim(cfg)

    # This is needed to initialize the agents
    sim.agents_mgr.on_new_scene()

    # For this tutorial, we will also add an extra camera that will be used for third person recording.
    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.uuid = "scene_camera_rgb"

    # TODO: this is a bit dirty but I think its nice as it shows how to modify a camera sensor...
    sim.add_sensor(camera_sensor_spec, 0)

    return sim


if __name__ == "__main__":
    repo = git.Repo(".", search_parent_directories=True)
    dir_path = repo.working_tree_dir
    data_path = os.path.join(dir_path, "data")
    os.chdir(dir_path)

    # Define the agent configuration
    main_agent_config = AgentConfig()
    urdf_path = os.path.join(data_path, "robots/hab_fetch/robots/hab_fetch.urdf")
    main_agent_config.articulated_agent_urdf = urdf_path
    main_agent_config.articulated_agent_type = "FetchRobot"

    # Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
    # We will later talk about why we are giving the sensors these names
    main_agent_config.sim_sensors = {
        "third_rgb": ThirdRGBSensorConfig(),
        "head_rgb": HeadRGBSensorConfig(),
    }

    # We create a dictionary with names of agents and their corresponding agent configuration
    agent_dict = {"main_agent": main_agent_config}
    sim = init_rearrange_sim(agent_dict)

    init_pos = mn.Vector3(-5.5,0,-1.5)
    art_agent = sim.articulated_agent
    # We will see later about this
    art_agent.sim_obj.motion_type = MotionType.KINEMATIC
    print("Current agent position:", art_agent.base_pos)
    art_agent.base_pos = init_pos 
    print("New agent position:", art_agent.base_pos)
    # We take a step to update agent position
    _ = sim.step({})

    # TODO(YCC)
    episode_file = "/home/yuchecheng/habitat-lab/data/datasets/mobility/mobility_episodes.json.gz"
    with gzip.open(episode_file, "rt") as f: 
        episode_files = json.loads(f.read())

    # Get the first episode
    episode = episode_files["episodes"][0]
    rearrange_episode = Rearrange3DEpisode(**episode)

    art_agent = sim.articulated_agent
    art_agent._fixed_base = True
    sim.agents_mgr.on_new_scene()


    sim.reconfigure(sim.habitat_config, ep_info=rearrange_episode)
    sim.reset()

    art_agent.sim_obj.motion_type = MotionType.KINEMATIC
    sim.articulated_agent.base_pos =  init_pos 
    _ = sim.step({})

    aom = sim.get_articulated_object_manager()
    rom = sim.get_rigid_object_manager()
