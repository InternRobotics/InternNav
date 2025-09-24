from typing import Any, Dict, List

from internutopia.core.config import Config, SimConfig
from internutopia.core.config.distribution import RayDistributionCfg
from internutopia.core.vec_env import Env

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import base
from internnav_benchmarks.internutopia.internutopia_vln_extension import (
    import_extensions,
)


@base.Env.register('vln_multi')
class VlnMultiEnv(base.Env):
    def __init__(self, env_config: EnvCfg, task_config: TaskCfg):
        super().__init__(env_config, task_config)
        env_settings = self.env_config.env_settings
        task_settings = self.task_config.task_settings
        config = Config(
            simulator=SimConfig(**env_settings),
            env_num=task_settings['env_num'],
            env_offset_size=task_settings['offset_size'],
            task_configs=task_settings['episodes'],
        )
        if 'distribution_config' in env_settings:
            distribution_config = RayDistributionCfg(**env_settings['distribution_config'])
            config = config.distribute(distribution_config)
        import_extensions()

        self.env = Env(config)
        self.env_num = task_settings['env_num']
        self.proc_num = env_settings['distribution_config']['proc_num'] if 'distribution_config' in env_settings else 1

    def reset(self, reset_index=None):
        # print('Vln env reset')
        return self.env.reset(reset_index)

    def step(self, action: List[Dict]):
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

    def transform_action_batch(self, actions: List[Dict], flash=False):
        transformed_actions = []
        for action in actions:
            if 'ideal_flag' in action.keys():
                ideal_flag = action['ideal_flag']
                if flash:
                    assert ideal_flag is True
            else:
                ideal_flag = False
            if not ideal_flag:
                transformed_actions.append({'h1': {'vln_dp_move_by_speed': action['action'][0]}})
                continue
            a = action['action']
            if a == 0 or a == [0] or a == [[0]]:
                transformed_actions.append({'h1': {'stop': []}})
            elif a == -1 or a == [-1] or a == [[-1]]:
                transformed_actions.append({'h1': {'stand_still': []}})
            else:
                move = f"move_by_{'discrete' if not flash else 'flash'}"
                transformed_actions.append({'h1': {move: a}})  # discrete e.g. [3]
        return transformed_actions

    def warm_up(self):
        while True:
            obs, _, _, _, _ = self.step(
                action=[{self.task_config.robot_name: {'stand_still': []}} for _ in range(self.env_num * self.proc_num)]
            )
            if obs[0][self.task_config.robot_name]['finish_action']:
                print('get_obs')
                break
        return obs
