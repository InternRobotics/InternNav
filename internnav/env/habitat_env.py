import json
import os
from typing import Any, Dict, List, Optional

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import base


@base.Env.register('habitat')
class HabitatEnv(base.Env):
    """A lightweight wrapper around `habitat.Env` that adapts Habitat to the project's `base.Env` interface.

    Args:
        env_config (EnvCfg): Environment configuration.
        task_config (TaskCfg): Optional task configuration passed to the base environment.
    """

    def __init__(self, env_config: EnvCfg, task_config: TaskCfg = None):
        try:
            from habitat import Env
        except ImportError as e:
            raise RuntimeError(
                "Habitat modules could not be imported. " "Make sure both repositories are installed and on PYTHONPATH."
            ) from e

        super().__init__(env_config, task_config)
        self.config = env_config.env_settings['habitat_config']
        self._env = Env(self.config)

        self.rank = env_config.env_settings.get('rank', 0)
        self.world_size = env_config.env_settings.get('world_size', 1)
        self._current_episode_index: int = 0
        self._last_obs: Optional[Dict[str, Any]] = None

        self.is_running = True
        self.output_path = env_config.env_settings.get('output_path', './output')

        # generate episodes
        self.episodes = self.generate_episodes()

    def generate_episodes(self) -> List[Any]:
        """
        Generate list of episodes for the current split.

        Returns:
            List[Any]: A list of episode objects for the current split.
        """
        all_episodes = []

        # group episodes by scene
        scene_episode_dict: Dict[str, List[Any]] = {}
        for episode in self._env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)

        # load done_res
        done_res = set()
        result_path = os.path.join(self.output_path, 'progress.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                for line in f:
                    res = json.loads(line)
                    # only skip if current format has scene_id
                    if "scene_id" in res:
                        done_res.add((res["scene_id"], res["episode_id"]))

        # iterate scenes in order, collect all episodes
        for scene in sorted(scene_episode_dict.keys()):
            per_scene_eps = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]

            # shard by rank index / world_size
            for episode in per_scene_eps[self.rank :: self.world_size]:
                episode_id = int(episode.episode_id)
                if (scene_id, episode_id) in done_res:
                    continue
                all_episodes.append(episode)

        return all_episodes

    def reset(self):
        # no more episodes
        if not (0 <= self._current_episode_index < len(self.episodes)):
            self.is_running = False
            return

        # Manually set to next episode in habitat
        self._env.current_episode = self.episodes[self._current_episode_index]
        self._current_episode_index += 1

        # Habitat reset
        self._last_obs = self._env.reset()
        return self._last_obs

    def step(self, action: List[Any]):
        obs = self._env.step(action)
        done = self._env.episode_over
        info = self._env.get_metrics()
        reward = info.get('reward', 0.0)
        return obs, reward, done, info

    def close(self):
        print('Habitat Env close')
        self._env.close()

    def render(self):
        self._env.render()

    def get_observation(self) -> Dict[str, Any]:
        return self._env.get_observations()

    def get_metrics(self) -> Dict[str, Any]:
        return self._env.get_metrics()

    def get_current_episode(self):
        return self._env.current_episode
