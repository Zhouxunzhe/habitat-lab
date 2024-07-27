from gym import spaces
from habitat.core.registry import registry
import numpy as np
import habitat_sim
from habitat.config.default import CN, Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import PointGoalSensor
from habitat_sim.utils.common import quat_from_magnum
from habitat.tasks.rearrange.rearrange3D_sim import Rearrange3DSim
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

@registry.register_sensor
class GrippedObjectSensor(Sensor):
    cls_uuid = "gripped_object_id"

    def __init__(
        self, *args: Any, sim: Rearrange3DSim, config: Config, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):

        return spaces.Discrete(self._sim.get_existing_object_ids())

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: Episode,
        *args: Any,
        **kwargs: Any,
    ):
        obj_id = self._sim.sim_object_to_objid_mapping.get(
            self._sim.gripped_object_id, -1
        )
        return obj_id


@registry.register_sensor
class ObjectPosition(PointGoalSensor):
    cls_uuid: str = "object_position"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        object_id = self._sim.get_existing_object_ids()[0]
        object_position = self._sim.get_translation(object_id)
        pointgoal = self._compute_pointgoal(
            agent_position, rotation_world_agent, object_position
        )
        return pointgoal


@registry.register_sensor
class ObjectGoal(PointGoalSensor):
    cls_uuid: str = "object_goal"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        goal_position = np.array(episode.goals.position, dtype=np.float32)

        point_goal = self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )
        return point_goal


@registry.register_measure
class ObjectToGoalDist(Measure):
    """The measure calculates distance of object towards the goal.
    """

    cls_uuid: str = "object_to_goal_dist"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToGoalDist.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        sim_obj_id = self._sim.get_existing_object_ids()[0]

        previous_position = np.array(
            self._sim.get_translation(sim_obj_id)
        ).tolist()
        goal_position = episode.goals.position
        self._metric = self._euclidean_distance(
            previous_position, goal_position
        )


@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent
    """

    cls_uuid: str = "agent_to_object_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        sim_obj_id = self._sim.get_existing_object_ids()[0]
        previous_position = np.array(
            self._sim.get_translation(sim_obj_id)
        ).tolist()

        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position

        self._metric = self._euclidean_distance(
            previous_position, agent_position
        )