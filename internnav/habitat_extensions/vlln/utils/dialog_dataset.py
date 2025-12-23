import gzip
import json
import os
from typing import TYPE_CHECKING, List, Optional

import attr
from habitat.core.dataset import Dataset
from habitat.core.registry import registry

from .dialog_episodes import (
    AgentPosition,
    DialogEpisode,
    DialogGoal,
    DialogViewLocation,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@attr.s(auto_attribs=True, kw_only=True)
class DialogInstructionData:
    task_type: str
    instruction_text: str
    instance_id: List[str]
    instruction_info: Optional[List[str]] = None


@registry.register_dataset(name="dialog")
class DialogDatasetV1(Dataset):
    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.data_path.format(split=config.split)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        self.episodes = list(filter(self.build_content_scenes_filter(config), self.episodes))

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized["category_to_task_category_id"]

        for episode in deserialized["episodes"]:
            episode = DialogEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.instruction = DialogInstructionData(**episode.instruction)
            for g_index, goal in enumerate(episode.goals):
                view_points = []
                for view_point in goal['view_points']:
                    view_point = DialogViewLocation(**{'agent_state': AgentPosition(**view_point['agent_state'])})
                    view_points.append(view_point)
                goal['view_points'] = view_points
                episode.goals[g_index] = DialogGoal(**goal)
            self.episodes.append(episode)

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(config.data_path.format(split=config.split)) and os.path.exists(config.scenes_dir)
