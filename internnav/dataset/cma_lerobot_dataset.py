import random
from collections import defaultdict

import numpy as np
import torch

# from internnav.modules.utils.feature_extract import extract_instruction_tokens
from internnav_baselines.modules.utils.feature_extract import extract_instruction_tokens

from internnav.dataset.base import BaseDataset, ObservationsDict, _block_shuffle
from internnav.utils.lerobot_as_lmdb import LerobotAsLmdb


class CMALerobotDataset(BaseDataset):
    def __init__(
        self,
        config,
        lerobot_features_dir,
        use_iw,
        dataset_data,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
        bert_tokenizer=None,
    ):
        super().__init__(config, dataset_data, lerobot_features_dir, lmdb_map_size, batch_size)
        self.config = config
        self.lerobot_features_dir = lerobot_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.dataset_data = dataset_data
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        self.camera_name = self.config.il.camera_name

        self.lerobot_as_lmdb = LerobotAsLmdb(self.lerobot_features_dir)
        self.lmdb_keys = self.lerobot_as_lmdb.get_all_keys()
        self.length = len(self.lmdb_keys)

        # For CMA-CLIP
        self.use_clip_encoders = False
        self.bert_tokenizer = None
        if self.config.model.policy_name == 'CMA_CLIP_Policy':
            self.is_clip_long = True
            self.bert_tokenizer = bert_tokenizer

        if hasattr(self.config.model, 'text_encoder'):
            self.bert_tokenizer = bert_tokenizer
            self.is_clip_long = True

        # change the BRG -> RGB for 3dgs dataset
        if '3dgs' in self.lmdb_features_dir:
            self.BRG_to_RGB = True
        else:
            self.BRG_to_RGB = False

    def _load_next(self):
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
                    continue
                finish_status = data_to_load['finish_status']
                fail_reason = data_to_load['fail_reason']
                # Filter the empty data
                if len(data['camera_info']) == 0:
                    empty_data_nums += 1
                    continue
                if self.config.il.filter_failure.use:
                    if finish_status != 'success':
                        if 'rgb' in data['camera_info'][self.camera_name].keys():
                            if (
                                len(data['camera_info']) == 0
                                or len(data['camera_info'][self.camera_name]['rgb'])
                                < self.config.il.filter_failure.min_rgb_nums
                            ):
                                continue
                        else:
                            if (
                                len(data['camera_info']) == 0
                                or len(data['rgb_features']) < self.config.il.Filter_failure.min_rgb_nums
                            ):
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
                        if 'rgb_features' in data.keys():
                            data['rgb_features'] = data['rgb_features'][:-drop_last_frame_nums]
                        if 'depth_features' in data.keys():
                            data['depth_features'] = data['depth_features'][:-drop_last_frame_nums]

                    # convert yaw from [-2pi,2pi] to [-pi, pi]
                    yaws = np.array(data['robot_info']['yaw']).copy()
                    for yaw_i, yaw in enumerate(data['robot_info']['yaw']):
                        yaw = yaw % (2 * np.pi)
                        if yaw > np.pi:
                            yaw -= 2 * np.pi
                        yaws[yaw_i] = yaw

                    episodes_in_json = data_to_load['episodes_in_json']

                    if self.bert_tokenizer is not None:
                        instructions = [
                            episodes_in_json[ep_idx]['instruction_text'] for ep_idx in range(len(episodes_in_json))
                        ]
                    else:
                        instructions = [
                            episodes_in_json[ep_idx]['instruction_tokens'] for ep_idx in range(len(episodes_in_json))
                        ]
                    for instruction in instructions:
                        new_data = self._create_new_data(data, yaws, instruction)
                        if self.BRG_to_RGB:
                            # This is for 3dgs dataset which is BRG format
                            new_data['rgb'] = new_data['rgb'][..., ::-1]
                            new_data['depth'] = new_data['depth'] * 100
                        for k, v in new_data.items():
                            if isinstance(v, np.ndarray):
                                new_data[k] = v[: self.config.model.max_step]
                        new_preload.append(new_data)
                        finish_status_list.append(finish_status)
                        fail_reasons_list.append(fail_reason)
                        lengths.append(len(new_data))

            if self.bert_tokenizer is not None:
                new_preload = extract_instruction_tokens(
                    new_preload, self.bert_tokenizer, is_clip_long=self.is_clip_long
                )

            # process the instruction
            # copy the instruction to each step
            for i in range(len(new_preload)):
                new_preload[i]['instruction'] = np.tile(
                    np.array(new_preload[i]['instruction']), (len(new_preload[i]['progress']), 1)
                )

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(obs['prev_actions']))
        oracle_actions = torch.from_numpy(np.copy(obs['gt_actions']))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __len__(self) -> int:
        return self.length * 3  # each trajectory corresponds to 3 instructions

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


def cma_collate_fn(batch):
    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount <= 0:
            if pad_amount < 0:
                t = t[:max_len]
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(observations_batch[bid][sensor])

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(observations_batch[sensor][bid], max_traj_len, fill_val=1.0)

        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        corrected_actions_batch[bid] = _pad_helper(corrected_actions_batch[bid], max_traj_len)
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
        observations_batch[sensor] = observations_batch[sensor].view(-1, *observations_batch[sensor].size()[2:])

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(corrected_actions_batch, dtype=torch.uint8)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )
