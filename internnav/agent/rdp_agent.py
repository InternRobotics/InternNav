import time

import numpy as np
import torch
from gym import spaces

from internnav.agent.base import Agent
from internnav.agent.utils.common import batch_obs, set_seed_model
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model import get_config, get_policy
from internnav.model.utils.feature_extract import (
    extract_image_features,
    extract_instruction_tokens,
)
from internnav.utils.common_log_util import common_logger as log
from internnav.utils.geometry_utils import (
    FixedLengthStack,
    compute_actions,
    get_delta,
    map_action_to_2d,
    normalize_data,
    quat_to_euler_angles,
    to_local_coords_batch,
)


@Agent.register('rdp')
class RdpAgent(Agent):
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 1),
        dtype=np.float32,
    )

    def __init__(self, config: AgentCfg):
        super().__init__(config)
        set_seed_model(0)
        self._model_settings = self.config.model_settings
        self._model_settings = ModelCfg(**self._model_settings)
        env_num = getattr(self._model_settings, 'env_num', 1)
        proc_num = getattr(self._model_settings, 'proc_num', 1)
        self.device = torch.device('cuda', 0)

        policy = get_policy(self._model_settings.policy_name)
        self.policy = policy.from_pretrained(
            config.ckpt_path,
            config=get_config(self._model_settings.policy_name)(model_cfg={'model': self._model_settings.model_dump()}),
        ).to(self.device)
        self.policy.eval()

        # step required
        self._env_nums = env_num * proc_num
        self._not_done_masks = torch.tensor([0 * self._env_nums], device=self.device).to(torch.bool)
        self._reset_ls = set()

        self._first_step = np.array([True] * self._env_nums)

        # instruction_encoder
        self.use_clip_encoders = True
        self.use_bert = False
        self.bert_tokenizer = None
        self.is_clip_long = False

        if self.use_clip_encoders:
            if self._model_settings.text_encoder.type == 'roberta':
                from internnav.model.utils.bert_token import BertTokenizer

                self.bert_tokenizer = BertTokenizer(
                    max_length=self._model_settings.instruction_encoder.max_length,
                    load_model=self._model_settings.instruction_encoder.load_model,
                    device=self.device,
                )
                self.use_bert = True
            elif self._model_settings.text_encoder.type == 'clip-long':
                from internnav.model.basemodel.LongCLIP.model import longclip

                self.bert_tokenizer = longclip.tokenize
                self.use_bert = True
                self.is_clip_long = True
        self._reset()

    def reset(self, reset_ls=None):
        if reset_ls is None:
            reset_ls = [i for i in range(self._env_nums)]
        self._reset_ls.update(reset_ls)
        log.debug(f'new reset_ls: {self._reset_ls}')

    def _reset(self):
        reset_ls = list(self._reset_ls)
        if len(reset_ls) == 0:
            self._rnn_states = torch.zeros(
                self._env_nums,
                self.policy.num_recurrent_layers,
                self._model_settings.state_encoder.hidden_size,
                device=self.device,
            )
            self.step_ = 1
            self.steps = np.array([1] * self._env_nums)
            self.action_cache = [[]] * self._env_nums

            self.prev_globalgps = [
                FixedLengthStack(self._model_settings.len_traj_act + 1) for _ in range(self._env_nums)
            ]

            self.prev_globalyaw = [
                FixedLengthStack(self._model_settings.len_traj_act + 1) for _ in range(self._env_nums)
            ]

            self.start_positions, self.start_yaws = torch.zeros(
                (self._env_nums, 2), device=self.device, dtype=torch.float64
            ), torch.zeros((self._env_nums), device=self.device, dtype=torch.float64)

            self.action_dim = 3
            self.action_stats = None
            if hasattr(self._model_settings, 'diffusion_policy'):
                self.action_stats = {
                    'min': torch.Tensor(np.asarray(self._model_settings.diffusion_policy.action_stats.min)).to(
                        self.device
                    ),
                    'max': torch.Tensor(np.asarray(self._model_settings.diffusion_policy.action_stats.max)).to(
                        self.device
                    ),
                }

            self._prev_actions = torch.zeros(
                self._env_nums, self._model_settings.len_traj_act, self.action_dim, device=self.device
            )

            self._not_done_masks = torch.zeros(self._env_nums, 1, device=self.device, dtype=torch.bool)
            self._reset_ls = set([i for i in range(self._env_nums)])
            self._first_step = np.array([True] * self._env_nums)
        else:
            self._rnn_states.index_fill_(dim=0, index=torch.tensor(reset_ls).to(self.device), value=0)
            self._prev_actions.index_fill_(dim=0, index=torch.tensor(reset_ls).to(self.device), value=0)
            self._not_done_masks.index_fill_(
                dim=0,
                index=torch.tensor(reset_ls).to(self.device),
                value=False,
            )
            self.steps[reset_ls] = 1
            for i in reset_ls:
                self.action_cache[i] = []
                self.prev_globalgps[i] = FixedLengthStack(self._model_settings.len_traj_act + 1)
                self.prev_globalyaw[i] = FixedLengthStack(self._model_settings.len_traj_act + 1)
            self._reset_ls = set(reset_ls)  # use for record start positions and rotations

    def _set_init_attrs(self, obs):
        start_positions = np.array([x['globalgps'][[0, 1]] for x in obs])
        new_start_positions = np.stack(start_positions[list(self._reset_ls)], axis=0)
        self.start_positions[list(self._reset_ls)] = torch.from_numpy(new_start_positions).to(self.device)

        start_yaws = np.array([x['globalyaw'] for x in obs])
        new_start_yaws = np.stack(start_yaws[list(self._reset_ls)], axis=0)
        self.start_yaws[list(self._reset_ls)] = torch.from_numpy(new_start_yaws).to(self.device)

    def _cal_prev_actions(self):
        for idx in range(self._env_nums):
            if idx in self._reset_ls:
                continue
            # reverse to make the latest frame to be 0 position
            prev_globalgps_numpy = np.array(self.prev_globalgps[idx].get_stack(reverse=True))
            prev_globalyaw_numpy = np.array(self.prev_globalyaw[idx].get_stack(reverse=True))
            prev_act = compute_actions(
                prev_globalgps_numpy,
                prev_globalyaw_numpy,
                curr_time=0,
                fill_mode='constant',
                len_traj_pred=self._model_settings.len_traj_act,
                waypoint_spacing=self._model_settings.diffusion_policy.waypoint_spacing,
                learn_angle=self._model_settings.learn_angle,
                metric_waypoint_spacing=self._model_settings.diffusion_policy.metric_waypoint_spacing,
                num_action_params=self.action_dim,
                normalize=False,
            )
            action_deltas = get_delta(prev_act)
            if self._model_settings.learn_angle:
                # [x,y,yaw]
                prev_act_delta = torch.from_numpy(action_deltas).to(self.device)
                prev_act_delta_norm = normalize_data(prev_act_delta, self.action_stats)
                self._prev_actions[idx] = prev_act_delta_norm
            else:
                # [forward, rotation]
                prev_act_delta = torch.from_numpy(map_action_to_2d(action_deltas)).to(self.device)
                self._prev_actions[idx] = prev_act_delta

    @property
    def _need_reset(self):
        return (len(self.action_cache[0]) == 0 and len(self._reset_ls) > 0) or (len(self._reset_ls) >= self._env_nums)

    def _process_obs(self, obs):
        # transfer globalyaw
        for _, observation in enumerate(obs):
            observation['globalyaw'] = quat_to_euler_angles(observation['globalrotation'])[-1]
        if self._need_reset:
            self._set_init_attrs(obs)
        # imu

        # instruction
        max_instr_len = 248
        if type(obs[0]['instruction']) == str:
            obs = extract_instruction_tokens(
                obs,
                bert_tokenizer=self.bert_tokenizer,
                is_clip_long=self.is_clip_long,
                max_instr_len=max_instr_len,
            )

        batch = batch_obs(obs, device=self.device)

        for env_idx in range(self._env_nums):
            self.prev_globalgps[env_idx].push(batch[env_idx]['globalgps'].detach().cpu().numpy())
            self.prev_globalyaw[env_idx].push(batch[env_idx]['globalyaw'].detach().cpu().item())

        self._cal_prev_actions()  # update prev_actions

        batch_stack_rgb, batch_stack_depth = None, None

        classifier_free_mask_depth = (
            self._model_settings.diffusion_policy.use_cls_free_guidance
            and self._model_settings.image_encoder.depth.update_depth_encoder
        )
        batch = extract_image_features(
            self.policy,
            batch,
            img_mod=self._model_settings.image_encoder.rgb.img_mod,
            len_traj_act=self._model_settings.image_encoder.img_stack_nums,
            world_size=1,
            depth_encoder_type=self._model_settings.image_encoder.depth.bottleneck,
            stack_rgb=batch_stack_rgb,
            stack_depth=batch_stack_depth,
            proj=self._model_settings.image_encoder.rgb.rgb_proj,
            need_rgb_extraction=True,
            classifier_free_mask_depth=classifier_free_mask_depth,
        )

        batch['steps'] = self.steps

        # imu
        if self._model_settings.imu_encoder.use:
            batch['imu'] = torch.zeros(
                self._env_nums,
                self._model_settings.imu_encoder.input_size,
                device=self.device,
                dtype=torch.float64,
            )
            if self._model_settings.imu_encoder.to_local_coords:
                reset_mask = torch.zeros(self._env_nums, dtype=torch.bool, device=self.device)
                reset_mask[list(self._reset_ls)] = True
                batch['imu'][~reset_mask, :2] = to_local_coords_batch(
                    batch['globalgps'][:, [0, 1]].float(),
                    self.start_positions,
                    self.start_yaws,
                )[~reset_mask]
            else:
                batch['imu'][~reset_mask, :2] = (batch['globalgps'][:, [0, 1]] - self.start_positions)[~reset_mask]
            if self._model_settings.imu_encoder.input_size == 3:
                batch['imu'][~reset_mask, 2] = batch['globalyaw'][~reset_mask] - self.start_yaws[~reset_mask]
            batch['imu'] = batch['imu'].float()

        env_idx = 0
        # record the current position before action

        return batch

    def inference(self, obs):
        if self._need_reset:
            self._reset()
            log.debug(f'model reset_ls: {self._reset_ls}')

        reset_mask = torch.zeros(self._env_nums, dtype=torch.bool, device=self.device)
        reset_mask[list(self._reset_ls)] = True

        obs = self._process_obs(obs)  # turn back self._reset_ls

        denoise_action = True
        batch = {
            'mode': 'update_rnn',
            'observations': obs,
            'rnn_states': self._rnn_states,
            'prev_actions': self._prev_actions,
            'masks': self._not_done_masks,
        }
        with torch.no_grad():
            _, rnn_states = self.policy(batch)
            # 在 no_grad 模式下进行赋值
            self._rnn_states[~reset_mask] = rnn_states[~reset_mask]

        self.step_ += 1
        self.steps += 1
        if len(self.action_cache[0]) == 0:  # may have problem
            self._reset_ls = set()
            reset_mask = torch.zeros(self._env_nums, dtype=torch.bool, device=self.device)
            batch = {
                'mode': 'act',
                'observations': obs,
                'rnn_states': self._rnn_states,
                'prev_actions': self._prev_actions,
                'masks': self._not_done_masks,
                'add_noise_to_action': True,
                'denoise_action': denoise_action,
                'num_sample': 1,
                'stop_mode': 'stop_progress',
                'step': self.step_,
                'steps': self.steps,
                'train_cls_free_guidance': False,
                'sample_classifier_free_guidance': self._model_settings.diffusion_policy.use_cls_free_guidance,
                'need_txt_extraction': True,
                'vis': False,
            }

            with torch.no_grad():
                (
                    actions,
                    self._rnn_states,
                    noise_pred,
                    dist_pred,
                    noise,
                    diffusion_output,
                    un_actions_nocumsum,
                    pm_pred,
                    stop_progress_pred,
                ) = self.policy(batch)

            self.step_ += 1
            self.steps += 1
            self.action_cache = list(map(lambda a: a[: self._model_settings.len_traj_act], actions))

        ret_actions = [action_ls.pop(0) for action_ls in self.action_cache]
        if [0] in ret_actions:
            print(f'ret_actions: {ret_actions}')
        self._not_done_masks = torch.ones(self._env_nums, 1, device=self.device, dtype=torch.bool)
        ret_actions = np.asarray(ret_actions)
        ret_actions[list(self._reset_ls)] = [-1]

        return ret_actions.tolist()

    def step(self, obs):
        print('RdpPolicyAgent step')
        start = time.time()
        action = self.inference(obs)
        end = time.time()
        print(f'总时间： {round(end-start, 4)}s')

        # convert from [[a1],[a2]] to [{'action': [a1],'ideal_flag':True}, {'action': [a2],'ideal_flag':True}]
        actions = []
        for a in action:
            if not isinstance(a, list):
                a = [a]
            actions.append({'action': a, 'ideal_flag': True})
        return actions
