from enum import Enum
from pathlib import Path
from time import time
from typing import Dict, List

import numpy as np

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.evaluator.utils.common import set_seed_model
from internnav.evaluator.utils.config import get_lmdb_path
from internnav.evaluator.utils.data_collector import DataCollector
from internnav.evaluator.utils.result_logger import ResultLogger
from internnav.evaluator.utils.visualize_util import VisualizeUtil
from internnav.utils import common_log_util, progress_log_multi_util
from internnav.utils.common_log_util import common_logger as log


class runner_status_code(Enum):
    NORMAL = 0
    WARM_UP = 1
    NOT_RESET = 3
    TERMINATED = 2
    STOP = 4


@Evaluator.register('vln_distributed')
class VLNDistributedEvaluator(DistributedEvaluator):
    def __init__(self, config: EvalCfg):
        start_time = time()

        self.task_name = config.task.task_name
        self.result_logger = ResultLogger(config.dataset)
        self.dataset_name = Path(config.dataset.dataset_settings['base_data_dir']).name
        config.env.env_settings['dataset'] = config.dataset

        # vec env settings
        self.env_num = config.task.task_settings['env_num']
        self.proc_num = (
            config.env.env_settings['distribution_config']['proc_num']
            if 'distribution_config' in config.env.env_settings
            else 1
        )

        # update config
        config.task.task_settings['env_num'] = self.env_num
        if 'distribution_config' in config.env.env_settings:
            config.env.env_settings['distribution_config']['proc_num'] = self.proc_num

        config.agent.model_settings.update({'env_num': self.env_num, 'proc_num': self.proc_num})
        self.robot_name = config.task.robot_name

        super().__init__(config)
        set_seed_model(0)

        common_log_util.init(self.task_name)
        self.total_path_num = len(self.env.episodes)
        progress_log_multi_util.init(self.task_name, self.total_path_num)
        progress_log_multi_util.progress_logger_multi.info(
            f'start eval dataset: {self.task_name}, total_path: {self.total_path_num}'  # noqa: E501
        )
        self.data_collector = DataCollector(get_lmdb_path(self.task_name), rank=self.rank, world_size=self.world_size)
        self.robot_flash = config.task.robot_flash
        self.save_to_json = config.eval_settings['save_to_json']
        self.vis_output = config.eval_settings['vis_output']
        self.visualize_util = VisualizeUtil(self.task_name, fps=6)

        end_time = time()
        duration = round(end_time - start_time, 2)
        log.info(f'[TIME] Env Init time: {duration}s')

    @property
    def ignore_obs_attr(self):
        return [
            'finish_action',
            'current_pose',
            'render',
            'fail_reason',
            'metrics',
        ]

    def remove_obs_attr(self, obs):
        return [{k: v for k, v in ob.items() if k not in self.ignore_obs_attr} for ob in obs]

    def warm_up(self):
        while True:
            obs, _, _, _, _ = self.env.step(
                action=[{self.robot_name: {'stand_still': []}} for _ in range(self.env_num * self.proc_num)]
            )
            if obs[0][self.robot_name]['finish_action']:
                break
        return obs

    def now_path_key(self, info):
        return info.data['path_key']

    def _obs_remove_robot_name(self, obs):
        obs = [
            *map(
                lambda ob: ob[self.robot_name] if ob is not None else self.fake_obs,
                obs,
            )
        ]
        return obs

    def _transform_action_batch(self, actions: List[Dict], flash=False):
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

    def get_action(self, obs, action):
        start_time = time()
        # process obs
        obs = np.array(obs)
        fake_obs_index = np.logical_or(
            self.runner_status == runner_status_code.WARM_UP,
            self.runner_status == runner_status_code.TERMINATED,
        )
        obs[fake_obs_index] = self.fake_obs
        obs = self.remove_obs_attr(obs)
        if not np.logical_and.reduce(self.runner_status == runner_status_code.WARM_UP):
            action = self.agent.step(obs)
            log.info(f'now action: {len(action)}, {action}, fake_obs_index: {fake_obs_index}')
            action = self._transform_action_batch(action, self.robot_flash)
        # change warm_up
        action = np.array(action)
        action[self.runner_status == runner_status_code.WARM_UP] = {'h1': {'stand_still': []}}
        end_time = time()
        duration = round(end_time - start_time, 2)
        log.info(f'[TIME] agent step time: {duration}s')
        return obs, action

    def _need_reset(self, terminated_ls):
        return np.logical_or.reduce(
            np.logical_and(
                terminated_ls,
                (self.runner_status != runner_status_code.TERMINATED),
            )
        )

    def env_step(self, action):
        start_time = time()

        while True:
            # stop action maybe also need 50 steps
            self.runner_status[
                np.logical_and(self.runner_status == runner_status_code.NORMAL, action == {'h1': {'stop': []}})
            ] = runner_status_code.STOP
            obs, reward, terminated, truncated, info = self.env.step(action=action.tolist())
            obs = self._obs_remove_robot_name(obs)
            finish_status = np.logical_or(
                np.array([ob['finish_action'] for ob in obs]),
                np.array(terminated),
            )  # strong condition

            if (
                np.logical_and.reduce(np.array(finish_status)[self.runner_status == runner_status_code.NORMAL])
                and runner_status_code.NORMAL in self.runner_status
            ) or np.logical_and.reduce(np.array(finish_status)):
                self.runner_status[self.runner_status == runner_status_code.STOP] = runner_status_code.NORMAL
                break
        end_time = time()
        duration = round(end_time - start_time, 2)
        log.info(f'[TIME] Env Step time: {duration}s')
        return obs, terminated

    def terminate_ops(self, obs_ls, reset_infos, terminated_ls):
        """
        1. reset agent if finished warm up
        2. reset envs that are terminated
        3. start new trace log and visualize log
        4. return whether all envs are terminated
        5. return updated reset_infos
        """
        start_time = time()

        finish_warmup_ls = (self.runner_status == runner_status_code.WARM_UP) & [ob['finish_action'] for ob in obs_ls]
        if np.logical_or.reduce(finish_warmup_ls):
            self.agent.reset(np.where(finish_warmup_ls)[0].tolist())
            self.runner_status[finish_warmup_ls] = runner_status_code.NORMAL
            log.info(f'env{np.where(finish_warmup_ls)[0].tolist()}: states switch to NORMAL.')
        # if no need reset, return False
        if not self._need_reset(terminated_ls):
            return False, reset_infos
        import json

        for env_id, terminated in enumerate(terminated_ls):
            if terminated and self.runner_status[env_id] != runner_status_code.TERMINATED:
                obs = obs_ls[env_id]
                reset_info = reset_infos[env_id]
                log.info(f"{self.now_path_key(reset_info)}: {json.dumps(obs['metrics'], indent=4)}")
                self.data_collector.save_eval_result(
                    key=self.now_path_key(reset_info),
                    result=obs['metrics'][list(obs['metrics'].keys())[0]][0]['fail_reason'],
                    info=obs['metrics'][list(obs['metrics'].keys())[0]][0],
                )  # save data to dataset
                # log data
                progress_log_multi_util.trace_end(
                    trajectory_id=self.now_path_key(reset_info),
                    step_count=obs['metrics'][list(obs['metrics'].keys())[0]][0]['steps'],
                    result=obs['metrics'][list(obs['metrics'].keys())[0]][0]['fail_reason'],
                )
                # visualize
                if self.vis_output:
                    self.visualize_util.trace_end(
                        trajectory_id=self.now_path_key(reset_info),
                        result=obs['metrics'][list(obs['metrics'].keys())[0]][0]['fail_reason'],
                    )
                # json format result
                self.result_logger.finalize_all_results(self.rank, self.world_size)
                self.runner_status[env_id] = runner_status_code.NOT_RESET
                log.info(f'env{env_id}: states switch to NOT_RESET.')
        # need this status to reset
        reset_env_ids = np.where(self.runner_status == runner_status_code.NOT_RESET)[0].tolist()
        if len(reset_env_ids) > 0:
            log.info(f'env{reset_env_ids}: start new episode!')
            obs, new_reset_infos = self.env.reset(reset_env_ids)
            self.runner_status[reset_env_ids] = runner_status_code.WARM_UP
            log.info(f'env{reset_env_ids}: states switch to WARM UP.')

            # modify original reset_info
            reset_infos = np.array(reset_infos)
            # If there is only one reset and no new_deset_infos, return an empty array
            reset_infos[reset_env_ids] = new_reset_infos if len(new_reset_infos) > 0 else None
            self.runner_status[
                np.vectorize(lambda x: x)(reset_infos) == None  # noqa: E711
            ] = runner_status_code.TERMINATED
            log.info(f'env{np.vectorize(lambda x: x)(reset_infos) == None}: states switch to TERMINATED.')
            reset_infos = reset_infos.tolist()

        if np.logical_and.reduce(self.runner_status == runner_status_code.TERMINATED):
            return True, reset_infos
        for reset_info in new_reset_infos:
            if reset_info is None:
                continue
            # start new trace log
            progress_log_multi_util.trace_start(
                trajectory_id=self.now_path_key(reset_info),
            )
            # start new visualize log
            if self.vis_output:
                self.visualize_util.trace_start(
                    trajectory_id=self.now_path_key(reset_info), reference_path=reset_info.data['reference_path']
                )

        end_time = time()
        duration = round(end_time - start_time, 2)
        log.info(f'[TIME] Env Reset time: {duration}s')
        return False, reset_infos

    def eval(self):
        print('--- VlnMultiEvaluator start ---')
        obs, reset_info = self.env.reset()
        for info in reset_info:
            if info is None:
                continue
            progress_log_multi_util.trace_start(
                trajectory_id=self.now_path_key(info),
            )
            if self.vis_output:
                self.visualize_util.trace_start(
                    trajectory_id=self.now_path_key(info), reference_path=info.data['reference_path']
                )
        log.info('start new episode!')

        obs = self.warm_up()
        self.fake_obs = obs[0][self.robot_name]
        action = [{self.robot_name: {'stand_still': []}} for _ in range(self.env_num * self.proc_num)]
        obs = self._obs_remove_robot_name(obs)
        self.runner_status = np.full(
            (self.env_num * self.proc_num),
            runner_status_code.NORMAL,
            runner_status_code,
        )
        self.runner_status[[info is None for info in reset_info]] = runner_status_code.TERMINATED

        while self.env.is_running():
            # get action from agent
            obs, action = self.get_action(obs, action)
            # step env
            obs, terminated = self.env_step(action)
            # terminate ops
            env_terminate, reset_info = self.terminate_ops(obs, reset_info, terminated)

            if env_terminate:
                break

            # save step obs
            if self.vis_output:
                for ob, info, act in zip(obs, reset_info, action):
                    if info is None or 'rgb' not in ob or ob['fail_reason']:
                        continue
                    self.visualize_util.save_observation(
                        trajectory_id=self.now_path_key(info), obs=ob, action=act[self.robot_name]
                    )

        self.env.close()
        progress_log_multi_util.report()

        print('--- VlnMultiEvaluator end ---')
