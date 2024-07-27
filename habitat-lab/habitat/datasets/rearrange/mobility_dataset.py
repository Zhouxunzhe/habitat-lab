import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

import attr
import numpy as np
import os
import habitat_sim.utils.datasets_download as data_downloader
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.core.dataset import Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1, CONTENT_SCENES_PATH_FIELD, DEFAULT_SCENE_PATH_PREFIX
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode, RearrangeDatasetV0


@attr.s(auto_attribs=True, kw_only=True)
class RearrangeSpec:
    r"""Specifications that capture the initial or final pose of the object."""
    bbox: List[List[float]] = attr.ib(default=None)
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    scale: List[float] = attr.ib(
        default=[1.0, 1.0, 1.0], validator=not_none_validator
    )


@attr.s(auto_attribs=True, kw_only=True)
class RearrangeObjectSpec(RearrangeSpec):
    r"""Object specifications that capture position of each object in the scene,
    the associated object template.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_handle: str = attr.ib(default=None, validator=not_none_validator)
    object_transform: np.ndarray = attr.ib(default=None, validator=not_none_validator)


# TODO(YCC): define the rearrangement episode for mp3d dataset
# @attr.s(auto_attribs=True, kw_only=True)
# class Rearrange3DEpisode(NavigationEpisode):
#     r"""Specifies additional objects for MP3D object rearrangement task."""
#     objects: List[RearrangeObjectSpec] = attr.ib(
#         default=None, validator=not_none_validator
#     )
#     goals: List[RearrangeSpec] = attr.ib(
#         default=None, validator=not_none_validator
#     )

@attr.s(auto_attribs=True, kw_only=True)
class Rearrange3DEpisode(RearrangeEpisode):
    goals: List[np.ndarray]
    
# TODO(YCC): Dataset for rearrange3D
# @registry.register_dataset(name="RearrangeDataset-v1")
# class RearrangeDatasetV1(PointNavDatasetV1):
#     episodes: List[Rearrange3DEpisode] = []
#     content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

#     def to_json(self) -> str:
#         result = DatasetFloatJSONEncoder().encode(self)
#         return result
    
#     def __init__(self, config: Optional["DictConfig"] = None) -> None:
#         self.config = config
#         super().__init__(config)

#     def from_json(
#         self, json_str: str, scenes_dir: Optional[str] = None
#     ) -> None:
#         deserialized = json.loads(json_str)
#         if CONTENT_SCENES_PATH_FIELD in deserialized:
#             self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

#         if "object_templates" in deserialized:
#             self.object_templates = deserialized["object_templates"]
#         # self.object_templates = deserialized['object_templates']
#         for i, episode in enumerate(deserialized["episodes"]):
#             episode_obj = Rearrange3DEpisode(**episode)
#             episode_obj.episode_id = str(i)

#             if scenes_dir is not None:
#                 if episode_obj.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
#                     episode_obj.scene_id = episode_obj.scene_id[
#                         len(DEFAULT_SCENE_PATH_PREFIX) :
#                     ]

#                 episode_obj.scene_id = os.path.join(
#                     scenes_dir, episode_obj.scene_id
#                 )

#             for i, obj in enumerate(episode_obj.objects):
#                 idx = obj["object_handle"]
#                 if type(idx) is not str:
#                     template = episode_obj.object_templates[idx]
#                     obj["object_handle"] = template["object_handle"]
#                 episode_obj.objects[i] = RearrangeObjectSpec(**obj)

#             for i, goal in enumerate(episode_obj.goals):
#                 episode_obj.goals[i] = RearrangeSpec(**goal)

#             self.episodes.append(episode_obj)

@registry.register_dataset(name="RearrangeDataset-v1")
class RearrangeDatasetV1(RearrangeDatasetV0):
    episodes: List[Rearrange3DEpisode] = []
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"


