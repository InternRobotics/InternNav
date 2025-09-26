import sys
from enum import Enum
from pathlib import Path
from time import time

import numpy as np

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator.base import Evaluator
from internnav.utils import common_log_util, progress_log_multi_util
from internnav.utils.common_log_util import common_logger as log
from internnav.utils.visualize_util import VisualizeUtil
from internnav_benchmarks.internutopia.episode_loader.resumable import (
    ResumableEpisodeIterator,
)
from internnav_benchmarks.internutopia.utils.common import set_seed_model
from internnav_benchmarks.internutopia.utils.config import get_lmdb_path
from internnav_benchmarks.internutopia.utils.data_collector import DataCollector
from internnav_benchmarks.internutopia.utils.dataset import ResultLogger, split_data
from internnav_benchmarks.internutopia.utils.eval import generate_episode


class runner_status_code(Enum):
    NORMAL = 0
    WARM_UP = 1
    NOT_RESET = 3
    TERMINATED = 2
    STOP = 4


@Evaluator.register('utopia')
class UtopiaEvaluator(Evaluator):
    """
    UtopiaEvaluator
    ----------------
    Evaluator for InternUtopia VLN episodes. It orchestrates end-to-end evaluation by:
    (1) materializing episodes, (2) managing environment/agent interaction, (3) logging
    per-trajectory progress and aggregate results, and (4) optionally saving visualization
    frames and JSON outputs.

    This evaluator also defines the **communication protocol** between the agent and the
    environment and adapts it to the simulator’s schema. Concretely, the agent consumes
    observation batches in a task-level VLN format and produces action batches in a
    task-level VLN format; the evaluator then converts those actions to the simulator/
    robot-specific format via `env.transform_action_batch(...)`.

    Registration
    ------------
    Registered as ``'utopia'`` via ``@Evaluator.register('utopia')``.

    Protocol (VLN task)
    -------------------
    The evaluator standardizes the agent↔env interface at the VLN task level:

    • Observation (per env slot):
        obs: Dict with at least:
            - 'rgb':   np.ndarray  # H,W,3
            - 'depth': np.ndarray  # H,W or H,W,1
        Additional keys produced by the environment (e.g., 'finish_action', 'metrics', etc.)
        are **stripped** before calling the agent, see ``ignore_obs_attr``.

    • Action (batch for all env slots):
        List[Dict[str, Any]], e.g.:
            [{'action': [2], 'ideal_flag': True}, ...]
        where:
            - 'action' is a length-1 list wrapping the discrete action id (int)
            - 'ideal_flag' (bool) may annotate oracle/ideal actions or other flags
        The evaluator will call:
            ``action = self.agent.step(obs_batch)``  # VLN format
            ``action = self.env.transform_action_batch(action, self.robot_flash)``  # sim format
        During warm-up or after termination, the evaluator may inject safe placeholders
        (e.g., stand-still) until all slots are ready.

    Lifecycle
    ---------
    1) Episode materialization:
        - Ensures LMDB split exists (``split_data``) and constructs a
          ``ResumableEpisodeIterator`` from dataset config.
        - Generates a finite episode list with ``generate_episode(...)`` and writes it
          into ``config.task.task_settings['episodes']``.

    2) Resource sizing:
        - Derives ``env_num`` / ``proc_num`` from config, then reduces them if
          ``env_num * proc_num`` exceeds available paths.
        - Propagates sizes to both task and agent model settings.

    3) Environment loop:
        - ``env.reset()`` → warm-up → repeated cycles of:
            a) prepare obs for the agent (remove ignored fields; fill warm-up slots),
            b) ``agent.step(obs_batch)`` → task-level actions,
            c) map actions to sim format via ``env.transform_action_batch(...)``,
            d) ``env.step(...)`` with barrier on per-slot finish/termination,
            e) per-slot termination handling, resets, and visualization dumps.

    4) Termination & reporting:
        - Saves per-trajectory metrics (``DataCollector``) and accumulates summary stats
          (``ResultLogger``). Final aggregate is emitted by
          ``progress_log_multi_util.report()``.

    Logging & Artifacts
    -------------------
    • Progress logs:
        - ``progress_log_multi_util`` tracks per-trajectory start/end, FPS, duration,
          and result. Log files are placed under
          ``{PROJECT_ROOT_PATH}/logs/{task_name}/progress/``.

    • Result logs:
        - ``ResultLogger`` writes running summaries; optional JSON output controlled by
          ``eval_settings['save_to_json']``.

    • Visualization:
        - If ``vis_output`` is True, ``VisualizeUtil`` stores per-step observations and
          end-of-trajectory markers (FPS=6 by default).

    Attributes (selected)
    ---------------------
    task_name : str
        Name of the benchmark task (used for paths, logging, visualization).
    episode_iterator : ResumableEpisodeIterator
        Iterator over concrete episodes; used to seed environment resets.
    env_num, proc_num : int
        Effective parallelism after auto-downscaling to dataset size.
    robot_name : str
        Simulator robot namespace used to peel robot-scoped observations.
    runner_status : np.ndarray[rnner_status_code]
        Per-slot status (NORMAL/WARM_UP/NOT_RESET/TERMINATED/STOP) controlling flow.
    data_collector : DataCollector
        Persists per-trajectory evaluation results (keyed by path_key).
    result_logger : ResultLogger
        Accumulates and periodically flushes aggregate metrics.
    vis_output : bool
        Enables visualization dumps.
    save_to_json : bool
        Enables JSON result dumps via ``ResultLogger``.

    Parameters
    ----------
    config : EvalCfg
        Full evaluator configuration:
        - dataset.* for episode sources and split materialization,
        - env.* for simulator settings (including optional distribution_config),
        - task.* for robot and task settings,
        - eval_settings.* for visualization and JSON output options.

    Notes
    -----
    - The evaluator strips keys listed in ``ignore_obs_attr`` before dispatching to
      the agent. Environment-only fields (e.g., 'finish_action', 'metrics', 'render')
      never reach the agent.
    - Warm-up and terminated slots receive safe placeholder observations and actions
      so batch shapes remain stable throughout evaluation.
    - Path identity is taken from ``reset_info.data['path_key']``; per-trajectory
      progress is emitted via ``trace_start/trace_end``.
    """

    def __init__(self, config: EvalCfg):
        self.task_name = config.task.task_name
        if not Path(get_lmdb_path(self.task_name)).exists():
            split_data(config.dataset)
        self.result_logger = ResultLogger(config.dataset)
        common_log_util.init(self.task_name)
        self.episode_iterator = ResumableEpisodeIterator(config.dataset.dataset_type, **config.dataset.dataset_settings)
        self.dataset_name = Path(config.dataset.dataset_settings['base_data_dir']).name
        progress_log_multi_util.init(self.task_name, self.episode_iterator.size)
        self.total_path_num = self.episode_iterator.size
        progress_log_multi_util.progress_logger_multi.info(
            f'start eval dataset: {self.task_name}, total_path:{self.episode_iterator.size}'  # noqa: E501
        )
        self.vis_output = config.eval_settings['vis_output']
        self.visualize_util = VisualizeUtil(self.task_name, fps=6)

        # generate episode
        episodes = generate_episode(self.episode_iterator, config)
        if len(episodes) == 0:
            log.info("No more episodes to evaluate")
            sys.exit(0)
        config.task.task_settings.update({'episodes': episodes})
        self.env_num = config.task.task_settings['env_num']
        self.proc_num = (
            config.env.env_settings['distribution_config']['proc_num']
            if 'distribution_config' in config.env.env_settings
            else 1
        )
        # check env_num and proc_num
        # priority: reduce env_num first then reduce proc_num
        while self.env_num > 1 and self.proc_num * self.env_num > self.total_path_num:
            self.env_num -= 1
            log.info(f'dataset size is too small! Change env_num to {self.env_num}.')
        while self.proc_num > 1 and self.proc_num * self.env_num > self.total_path_num:
            self.proc_num -= 1
            log.info(f'dataset size is too small! Change proc_num to {self.proc_num}.')
        # update
        config.task.task_settings['env_num'] = self.env_num
        if 'distribution_config' in config.env.env_settings:
            config.env.env_settings['distribution_config']['proc_num'] = self.proc_num

        config.agent.model_settings.update({'env_num': self.env_num, 'proc_num': self.proc_num})
        self.robot_name = config.task.robot_name
        super().__init__(config)
        set_seed_model(0)
        self.data_collector = DataCollector(self.episode_iterator.lmdb_path)
        self.robot_flash = config.task.robot_flash
        self.save_to_json = config.eval_settings['save_to_json']

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

    def get_action(self, obs, action):
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
            log.info(f'now action:{len(action)} ,{action}, fake_obs_index:{fake_obs_index}')
            action = self.env.transform_action_batch(action, self.robot_flash)
        # change warm_up
        action = np.array(action)
        action[self.runner_status == runner_status_code.WARM_UP] = {'h1': {'stand_still': []}}
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
        # stop_count = [0 for _ in range(self.env_num * self.sim_num)]
        while True:
            # stop action maybe also need 50 steps
            self.runner_status[
                np.logical_and(self.runner_status == runner_status_code.NORMAL, action == {'h1': {'stop': []}})
            ] = runner_status_code.STOP
            # print(action)
            # t0 = time()
            obs, reward, terminated, truncated, info = self.env.step(action=action.tolist())
            # print(f"inner one step time {time() - t0}")
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
            if __debug__ and np.logical_or.reduce(np.array(finish_status)):
                print(f'finish_status: {finish_status}')
        end_time = time()
        duration = round(end_time - start_time, 2)
        log.info(f'env step time: {duration}s')
        return obs, terminated

    def terminate_ops(self, obs_ls, reset_infos, terminated_ls):
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
                if not __debug__:
                    pass
                log.info(json.dumps(obs['metrics']))
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
                self.result_logger.write_now_result()
                if self.vis_output:
                    self.visualize_util.trace_end(
                        trajectory_id=self.now_path_key(reset_info),
                        result=obs['metrics'][list(obs['metrics'].keys())[0]][0]['fail_reason'],
                    )
                if self.save_to_json:
                    self.result_logger.write_now_result_json()
                self.runner_status[env_id] = runner_status_code.NOT_RESET
                log.info(f'env{env_id}: states switch to NOT_RESET.')
        reset_env_ids = np.where(self.runner_status == runner_status_code.NOT_RESET)[  # need this status to reset
            0
        ].tolist()
        if len(reset_env_ids) > 0:
            log.info(f'env{reset_env_ids}: start new episode!')
            obs, new_reset_infos = self.env.reset(reset_env_ids)
            self.runner_status[reset_env_ids] = runner_status_code.WARM_UP
            log.info(f'env{reset_env_ids}: states switch to WARM UP.')
            # modify original reset_info
            reset_infos = np.array(reset_infos)
            reset_infos[reset_env_ids] = new_reset_infos if len(new_reset_infos) > 0 else None
            self.runner_status[
                np.vectorize(lambda x: x)(reset_infos) == None  # noqa: E711
            ] = runner_status_code.TERMINATED
            log.info(f'env{np.vectorize(lambda x: x)(reset_infos) == None}: states switch to TERMINATED.')
            reset_infos = reset_infos.tolist()

        if np.logical_and.reduce(self.runner_status == runner_status_code.TERMINATED):
            print('finished')
            return True, reset_infos
        for reset_info in new_reset_infos:
            if reset_info is None:
                continue
            progress_log_multi_util.trace_start(
                trajectory_id=self.now_path_key(reset_info),
            )
            if self.vis_output:
                self.visualize_util.trace_start(
                    trajectory_id=self.now_path_key(reset_info), reference_path=reset_info.data['reference_path']
                )
        return False, reset_infos

    def eval(self):
        print('--- UtopiaEvaluator start ---')
        obs, reset_info = self.env.reset()
        print('obs:', obs)
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

        obs = self.env.warm_up()
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

            obs, action = self.get_action(obs, action)
            obs, terminated = self.env_step(action)
            env_term, reset_info = self.terminate_ops(obs, reset_info, terminated)
            if env_term:
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

        print('--- UtopiaEvaluator end ---')
