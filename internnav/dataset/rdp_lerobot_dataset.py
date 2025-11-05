import copy
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from internnav.dataset.base import BaseDataset, ObservationsDict, _block_shuffle
from internnav.evaluator.utils.common import norm_depth
from internnav.model.basemodel.LongCLIP.model import longclip
from internnav.model.utils.feature_extract import extract_instruction_tokens
from internnav.utils.geometry_utils import get_delta, normalize_data, to_local_coords
from internnav.utils.lerobot_as_lmdb import LerobotAsLmdb


def _convert_image_to_rgb(image):
    return image.convert('RGB')


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def optimize_delta_action(action_deltas, gt_actions):
    # This function is used to optimize the action deltas to be more similar to the descrete actions
    # action_deltas: [T, 3]
    # gt_actions: [T]
    # return: [T, 3]
    turn_angle = 0
    for idx, a in enumerate(gt_actions):
        if a == 1:
            if abs(action_deltas[idx][0]) < 0.1 and abs(action_deltas[idx][1]) < 0.1:
                action_deltas[idx][0] = 0.25 * np.cos(turn_angle)
                action_deltas[idx][1] = 0.25 * np.sin(turn_angle)
                if action_deltas[idx][2] > 0.1:
                    action_deltas[idx][2] = 0.0
        elif a == 2:
            if abs(action_deltas[idx][0]) > 0.1:
                action_deltas[idx][0] = 0
            if abs(action_deltas[idx][1]) > 0.1:
                action_deltas[idx][1] = 0
            if abs(action_deltas[idx][2]) < 0.2:
                action_deltas[idx][2] = 0.27
            turn_angle += np.pi / 12
        elif a == 3:
            if abs(action_deltas[idx][0]) > 0.1:
                action_deltas[idx][0] = 0
            if abs(action_deltas[idx][1]) > 0.1:
                action_deltas[idx][1] = 0
            if abs(action_deltas[idx][2]) < 0.2:
                action_deltas[idx][2] = -0.27
            turn_angle -= np.pi / 12
        elif a == 0:
            action_deltas[idx] = torch.zeros_like(action_deltas[idx])
    return action_deltas


class RDP_LerobotDataset(BaseDataset):
    def __init__(
        self,
        config,
        lerobot_features_dir,
        dataset_data: dict,
        lmdb_map_size=1e9,
        batch_size=1,
    ):
        super().__init__(config, dataset_data, lerobot_features_dir, lmdb_map_size, batch_size)
        self.lerobot_features_dir = lerobot_features_dir
        self.dp_config = config.model.diffusion_policy

        self.action_stats = {}
        self.action_stats['min'] = np.array(self.config.model.diffusion_policy.action_stats.min)
        self.action_stats['max'] = np.array(self.config.model.diffusion_policy.action_stats.max)

        self.is_clip_long = self.config.model.text_encoder.type == 'clip-long'
        self.preload_size = batch_size * 10

        # preprocess images
        self.to_pil = ToPILImage()
        self.image_processor = _transform(n_px=224)  # copy from clip-long
        self.lerobot_as_lmdb = LerobotAsLmdb(self.lerobot_features_dir)
        self.lmdb_keys = self.lerobot_as_lmdb.get_all_keys()
        self.length = len(self.lmdb_keys)

        self.start = 0
        self.end = self.length
        self.bert_tokenizer = longclip.tokenize

        self.waypoint_spacing = self.dp_config.waypoint_spacing
        self.len_traj_pred = self.dp_config.len_traj_pred
        self.learn_angle = self.config.model.learn_angle

        if self.config.model.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

        # change the BRG -> RGB for 3dgs dataset
        if '3dgs' in self.lmdb_features_dir:
            self.BRG_to_RGB = True
        else:
            self.BRG_to_RGB = False

    def _load_next(self):  # noqa: C901
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            finish_status_list = []
            fail_reasons_list = []
            empty_data_nums = 0

            for _ in range(self.preload_size):
                if len(self.load_ordering) == 0:
                    break

                key = self.lmdb_keys[self.load_ordering.pop()]
                data_to_load = self.lerobot_as_lmdb.get_data_by_key(key)
                try:
                    data = data_to_load['episode_data']
                except KeyError:
                    # print(f'KeyError: {key}')
                    continue
                finish_status = data_to_load['finish_status']
                fail_reason = data_to_load['fail_reason']
                # Filter the empty data
                if len(data['camera_info']) == 0:
                    empty_data_nums += 1
                    continue
                if self.config.il.filter_failure.use:
                    # Use certain rules to utilize failed collection data
                    if finish_status != 'success':
                        if len(data['camera_info']) == 0:  # without any camera info
                            continue
                        if 'rgb' in data['camera_info'][self.camera_name].keys():
                            if (
                                len(data['camera_info'][self.camera_name]['rgb'])
                                < self.config.il.filter_failure.min_rgb_nums
                            ):
                                continue
                        else:
                            if len(data['camera_info']) == 0:
                                continue
                    if finish_status == 'stuck':
                        drop_last_frame_nums = 25
                        data['camera_info'][self.camera_name]['rgb'] = data['camera_info'][self.camera_name]['rgb'][
                            :-drop_last_frame_nums
                        ]
                        data['camera_info'][self.camera_name]['depth'] = data['camera_info'][self.camera_name]['depth'][
                            :-drop_last_frame_nums
                        ]
                        data['robot_info']['yaw'] = data['robot_info']['yaw'][:-drop_last_frame_nums]
                        data['robot_info']['position'] = data['robot_info']['position'][:-drop_last_frame_nums]
                        data['robot_info']['orientation'] = data['robot_info']['orientation'][:-drop_last_frame_nums]
                        data['progress'] = data['progress'][:-drop_last_frame_nums]
                        data['step'] = data['step'][:-drop_last_frame_nums]

                # convert yaw from [-2pi,2pi] to [-pi, pi]
                yaws = np.array(data['robot_info']['yaw']).copy()
                for yaw_i, yaw in enumerate(data['robot_info']['yaw']):
                    yaw = yaw % (2 * np.pi)
                    if yaw > np.pi:
                        yaw -= 2 * np.pi
                    yaws[yaw_i] = yaw

                episodes_in_json = data_to_load['episodes_in_json']

                instructions = [
                    episodes_in_json[ep_idx]['instruction_text'][: self.config.model.text_encoder.max_length]
                    for ep_idx in range(len(episodes_in_json))
                ]

                for instruction in instructions:
                    new_data = self._create_new_data(data, yaws, instruction)
                    # limit the max length
                    if self.BRG_to_RGB:
                        # This is for 3dgs dataset which is BRG format
                        new_data['rgb'] = new_data['rgb'][..., ::-1]
                        new_data['depth'] = new_data['depth'] * 100
                        new_data['depth'] = norm_depth(new_data['depth'])
                    for k, v in new_data.items():
                        if isinstance(v, np.ndarray):
                            new_data[k] = v[: self.config.model.max_step]
                    new_preload.append(new_data)
                    finish_status_list.append(finish_status)
                    fail_reasons_list.append(fail_reason)
                    lengths.append(len(new_data))

            if self.need_extract_instr_features:
                new_preload = extract_instruction_tokens(
                    new_preload, self.bert_tokenizer, is_clip_long=self.is_clip_long
                )

            # process the instruction
            # copy the instruction to each step
            if self.need_extract_instr_features:
                for i in range(len(new_preload)):
                    new_preload[i]['instruction'] = np.tile(
                        np.array(new_preload[i]['instruction']), (len(new_preload[i]['progress']), 1)
                    )
            else:
                for i in range(len(new_preload)):
                    new_preload[i]['instruction'] = np.expand_dims(new_preload[i]['instruction'], axis=0)
                    new_preload[i]['instruction'] = np.tile(
                        new_preload[i]['instruction'], (len(new_preload[i]['progress']), 1, 1)
                    )

            for item_idx in range(len(new_preload)):
                item_obs = new_preload[item_idx]
                total_steps = len(item_obs['progress'])

                # add stop_progress
                item_obs['stop_progress'] = (np.arange(total_steps) + 1) / total_steps

                for k, v in item_obs.items():
                    item_obs[k] = torch.from_numpy(np.array(item_obs[k]))

                item_obs['actions'] = torch.zeros((total_steps, self.len_traj_pred, 3))
                item_obs['prev_actions'] = torch.zeros((total_steps, self.config.model.len_traj_act, 3))

                if self.extract_img_features:
                    # extract image features from raw images
                    # process RGB images
                    process_images = []
                    for image in item_obs['rgb']:
                        image = image.permute(2, 0, 1)  # H,W,C -> C,H,W
                        process_images.append(self.image_processor(self.to_pil(image)))
                    item_obs['rgb'] = torch.stack(process_images)  # [T, 3, 224, 224]

                    depth_shape = item_obs['depth'][0].shape
                    if len(depth_shape) == 2:
                        # [256, 256] -> [256, 256, 1]
                        item_obs['depth'] = torch.unsqueeze(item_obs['depth'], dim=-1)

                if self.config.model.imu_encoder.use:
                    item_obs['imu'] = torch.zeros((total_steps, self.config.model.imu_encoder.input_size))

                start_pos = item_obs['globalgps'][0][[0, 1]]
                start_yaw = item_obs['globalyaw'][0]
                for step_idx in range(total_steps):
                    # compute imu
                    if self.config.model.imu_encoder.use:
                        current_pos = item_obs['globalgps'][step_idx][[0, 1]]
                        if self.config.model.imu_encoder.to_local_coords:
                            item_obs['imu'][step_idx][:2] = to_local_coords(current_pos, start_pos, start_yaw)
                        if self.config.model.imu_encoder.input_size == 3:
                            item_obs['imu'][step_idx][2] = item_obs['globalyaw'][step_idx] - start_yaw

                for step_idx in range(total_steps):
                    # compute actions
                    actions = self._compute_actions(
                        item_obs['globalgps'], item_obs['globalyaw'], step_idx, fill_mode='constant'
                    )
                    prev_actions = self._compute_actions(
                        torch.flip(item_obs['globalgps'], dims=[0]),
                        torch.flip(item_obs['globalyaw'], dims=[0]),
                        total_steps - step_idx - 1,
                        fill_mode='constant',
                    )[: self.config.model.len_traj_act]

                    action_deltas = get_delta(actions)

                    if self.config.il.use_discrete_dataset and 'gt_actions' in item_obs.keys():
                        end_step_idx = min(step_idx + self.len_traj_pred, len(item_obs['gt_actions']))
                        gt_actions = item_obs['gt_actions'][step_idx:end_step_idx]
                        if len(gt_actions) < self.len_traj_pred:
                            gt_actions = np.concatenate([gt_actions, np.zeros(self.len_traj_pred - len(gt_actions))])
                        action_deltas = optimize_delta_action(action_deltas, gt_actions)

                    item_obs['actions'][step_idx] = normalize_data(
                        action_deltas, self.action_stats
                    )  # convert actions to [-1, 1]

                    # update prev action
                    if step_idx > 0:
                        prev_action_deltas = get_delta(prev_actions)
                        item_obs['prev_actions'][step_idx] = normalize_data(prev_action_deltas, self.action_stats)

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        if len(self._preload) == 0:
            raise StopIteration
        return self._preload.pop()  # pop one item each time

    def __next__(self):
        obs = self._load_next()

        return obs

    def _compute_actions(self, globalgps, yaws, curr_time, fill_mode):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = yaws[start_index : end_index : self.waypoint_spacing]
        globalgps = globalgps[:, [0, 1]]
        positions = globalgps[start_index : end_index : self.waypoint_spacing]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            if fill_mode == 'constant':
                yaw = torch.cat([yaw, yaw[-1].repeat(const_len)])
                positions = torch.cat([positions, positions[-1].unsqueeze(0).repeat(const_len, 1)], dim=0)
            elif fill_mode == 'zero':
                yaw = torch.cat([yaw, torch.zeros(const_len)])
                positions = torch.cat([positions, torch.zeros((const_len, 2))], dim=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f'{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal'
        assert positions.shape == (
            self.len_traj_pred + 1,
            2,
        ), f'{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal'

        waypoints = to_local_coords(positions, positions[0], yaw[0])

        assert waypoints.shape == (
            self.len_traj_pred + 1,
            2,
        ), f'{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal'

        delta_yaw = yaw[1:] - yaw[0]
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))

        actions = torch.cat([waypoints[1:], delta_yaw[:, None]], dim=-1)
        if self.learn_angle:
            assert actions.shape == (
                self.len_traj_pred,
                self.num_action_params,
            ), f'{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal'

        return actions

    def __len__(self) -> int:
        return self.length * 3

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(reversed(_block_shuffle(list(range(start, end)), self.preload_size)))

        return self


def rdp_collate_fn(global_batch_size=None):
    def _rdp_collate_fn(batch):
        def _pad_helper(t, max_len, fill_val=0, return_masks=False):
            pad_amount = max_len - t.size(0)
            if pad_amount == 0:
                if return_masks:
                    mask = torch.ones(max_len, dtype=torch.int)
                    return t, mask
                return t

            pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])

            # Create the mask: 1 for original tokens, 0 for padding
            if return_masks:
                mask = torch.zeros(max_len, dtype=torch.int)
                mask[: t.size(0)] = 1  # Original tokens
                mask[t.size(0) :] = 0  # Padded tokens

                return torch.cat([t, pad], dim=0), mask
            return torch.cat([t, pad], dim=0)

        observations_batch = batch

        B = torch.tensor(len(observations_batch))
        if B < global_batch_size:
            while B < global_batch_size:
                sample_to_copy = random.choice(batch)
                observations_batch.append(copy.deepcopy(sample_to_copy))
                B = torch.tensor(len(observations_batch))
        B = torch.tensor(len(observations_batch))

        new_observations_batch = defaultdict(list)
        for sensor in observations_batch[0].keys():
            for bid in range(B):
                new_observations_batch[sensor].append(observations_batch[bid][sensor])

        observations_batch = new_observations_batch

        max_traj_len = max(ele.size(0) for ele in observations_batch['progress'])
        not_done_masks_batch = torch.ones(B, max_traj_len, dtype=torch.uint8)
        for bid in range(B):
            for sensor in observations_batch:
                if sensor == 'progress':
                    observations_batch[sensor][bid] = _pad_helper(
                        observations_batch[sensor][bid], max_traj_len, fill_val=1.0
                    )
                else:
                    observations_batch[sensor][bid] = _pad_helper(
                        observations_batch[sensor][bid], max_traj_len, fill_val=0.0
                    )

        for sensor in observations_batch:
            observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
            observations_batch[sensor] = observations_batch[sensor].view(-1, *observations_batch[sensor].size()[2:])

        observations_batch = ObservationsDict(observations_batch)
        # Expand B to match the flattened batch size
        B_expanded = B.repeat(observations_batch['prev_actions'].shape[0]).view(-1, 1)

        return (observations_batch, observations_batch['prev_actions'], not_done_masks_batch.view(-1, 1), B_expanded)

    return _rdp_collate_fn
