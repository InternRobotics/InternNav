import os
import sys
from typing import Any, Dict, List

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import base
from internnav.env.utils.episode_loader import (
    ResumablePathKeyEpisodeloader,
    generate_vln_episode,
)


@base.Env.register('internutopia')
class InternutopiaEnv(base.Env):
    def __init__(self, env_config: EnvCfg, task_config: TaskCfg):
        try:
            from internutopia.core.config import Config, SimConfig
            from internutopia.core.config.distribution import RayDistributionCfg
            from internutopia.core.vec_env import Env

            from internnav.env.utils.internutopia_extension import import_extensions
        except ImportError as e:
            raise RuntimeError(
                "InternUtopia modules could not be imported. "
                "Make sure both repositories are installed and on PYTHONPATH."
            ) from e

        super().__init__(env_config, task_config)
        env_settings = self.env_config.env_settings
        task_settings = self.task_config.task_settings

        # generate episodes
        self.episode_loader = ResumablePathKeyEpisodeloader(
            env_settings['dataset'].dataset_type,
            **env_settings['dataset'].dataset_settings,
            rank=env_settings['rank'],
            world_size=env_settings['world_size']
        )
        self.episodes = generate_vln_episode(self.episode_loader, task_config)
        if len(self.episodes) == 0:
            print("No episodes found for the given configuration.")
            sys.exit(0)
        task_settings.update({'episodes': self.episodes})

        # set visible device for isaac sim
        os.environ["CUDA_VISIBLE_DEVICES"] = str(env_settings.get('local_rank', 0))

        config = Config(
            simulator=SimConfig(**env_settings),
            env_num=task_settings['env_num'],
            env_offset_size=task_settings['offset_size'],
            task_configs=task_settings['episodes'],
        )
        if 'distribution_config' in env_settings:
            distribution_config = RayDistributionCfg(**env_settings['distribution_config'])
            config = config.distribute(distribution_config)

        # register all extensions
        import_extensions()

        self.env = Env(config)

    def reset(self, reset_index=None):
        return self.env.reset(reset_index)

    def step(self, action: List[Any]):
        return self.env.step(action)

    def is_running(self):
        return True

    def close(self):
        print('Vln Env close')
        self.env.close()

    def render(self):
        self.env.render()

    def get_observation(self) -> Dict[str, Any]:
        return self.env.get_observations()

    def get_info(self) -> Dict[str, Any]:
        pass
