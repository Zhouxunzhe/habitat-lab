@registry.register_sensor
class ObjPointSensor(UsesArticulatedAgentInterface, Sensor):
    """
    ObjPoint
    """
    cls_uuid = "obj_pos"

    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._task = task
        self._targets = {}
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
    def _get_uuid(self, *args, **kwargs):
        return ObjPointSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

@registry.register_sensor
class CameraExtrinsicSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "camera_extrinsic"
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")

    def _get_uuid(self, *args, **kwargs):
        return CameraExtrinsicSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(low=0.0, high=np.finfo(np.float64).max, shape=(4, 4), dtype=np.float64)
    def _get_camera_extrinsic(self, camera_name)-> np.ndarray:
        """get camera extrinsic from habitat simulator config
        Assume the depth and color sensor are aligned and have the same extrinsic parameters
        Args:
            sim (haibtat_sim.Simulator): simulator config class
            camera_name: name of the camera sensor
        """
        camera_key = camera_name.replace("_depth", "")
        cur_articulated_agent = self._sim.get_agent_data(self.agent_id).articulated_agent
        cam_info = cur_articulated_agent.params.cameras[camera_key]
        cam_trans = get_articulated_agent_camera_transform_from_cam_info(
            cur_articulated_agent, cam_info)
        return cam_trans

    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
        else:
            depth_camera_name = self.depth_sensor_name
        camera_extrinsic = self._get_camera_extrinsic(depth_camera_name)
        return camera_extrinsic

@registry.register_sensor
class TransofrobotSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "robot_trans_martix"

    def _get_uuid(self, *args, **kwargs):
        return TransofrobotSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(4,4),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        # print(f"agent_{self.agent_id}_trans:{trans}")
        return np.array(trans, dtype=np.float32)
    
@registry.register_sensor
class TargetStartSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "ee_global_pos_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        return np.array(ee_pos, dtype=np.float32)
    
@registry.register_sensor
class EEPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "ee_pos"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EEPositionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        # print("trans:",trans.type)
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        # print("eesss:",ee_pos)
        # if self.agent_id == 0:print(str(self.agent_id)+"ee_pos_global:",ee_pos)
        local_ee_pos = trans.inverted().transform_point(ee_pos)
        
        # print("hello:",local_ee_pos)
        # print("goog",goob[:3])
        return np.array(local_ee_pos, dtype=np.float32)