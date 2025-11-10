import json
import os
from typing import Any, Dict, List, Optional

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import base


@base.Env.register('habitat')
class HabitatEnv(base.Env):
    def __init__(self, env_config: EnvCfg, task_config: TaskCfg):
        """
        env_settings include:
            - config_path: str, path to habitat config yaml file
            - split: str, dataset split to use
        """
        try:
            from habitat import Env
        except ImportError as e:
            raise RuntimeError(
                "Habitat modules could not be imported. " "Make sure both repositories are installed and on PYTHONPATH."
            ) from e

        super().__init__(env_config, task_config)

        self.config = env_config.env_settings['habitat_config']
        self.env = Env(self.config)

        self.episodes = self.generate_episodes()
        self.sort_episodes_by_scene()

        self.index = env_config.env_settings.get('idx', 0)
        self.world_size = env_config.env_settings.get('world_size', 1)
        self._current_episode_index: int = 0
        self._last_obs: Optional[Dict[str, Any]] = None

        self.step_id = 0
        self.is_running = True

    def generate_episodes(self) -> List[Any]:
        """
        Generate list of episodes for the current split
        """
        episodes = []

        # sort episode by scene
        scene_episode_dict = {}
        for episode in self.env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        done_res = set()

        if os.path.exists(os.path.join(self.output_path, 'result.json')):
            with open(os.path.join(self.output_path, 'result.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.add((res["scene_id"], res["episode_id"], res["episode_instruction"]))

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            for episode in episodes[self.index :: self.world_size]:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if 'objectnav' not in self.config_path
                    else episode.object_category
                )
                episode_id = int(episode.episode_id)
                if (scene_id, episode_id, episode_instruction) in done_res:
                    continue
                episodes.append(episode)
        return episodes

    def reset(self):
        """
        load next episode and return first observation
        """
        # no more episodes
        if not (0 <= self._current_episode_index < len(self.episodes)):
            self.is_running = False
            return

        # Manually set to next episode in habitat
        self.env.current_episode = self.episodes[self._current_episode_index]
        self._current_episode_index += 1

        # Habitat reset
        self._last_obs = self.env.reset()
        self.step_id = 0

        return self._last_obs

    def step(self, action: List[Any]):
        """
        step the environment with given action

        Args: action: List[Any], action for each env in the batch

        Return: obs, terminated
        """
        self._last_obs = self.env.step(action)
        terminated = self.env.episode_over
        return self._last_obs, terminated

    def close(self):
        print('Vln Env close')
        self.env.close()

    def render(self):
        self.env.render()

    def get_observation(self) -> Dict[str, Any]:
        return self.env.get_observations()

    def get_metrics(self) -> Dict[str, Any]:
        return self.env.get_metrics()

    def sort_episodes_by_scene(self, key_list: List[str]):
        sorted_episodes = []
        episode_dict = {ep.episode_id: ep for ep in self.episodes}
        for key in key_list:
            if key in episode_dict:
                sorted_episodes.append(episode_dict[key])
        self.episodes = sorted_episodes
