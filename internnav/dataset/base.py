import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class BaseDataset(IterableDataset):
    def __init__(self, config, dataset_data, lmdb_features_dir, lmdb_map_size=1e9, batch_size=1):
        super().__init__()
        self.config = config
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.dataset_data = dataset_data
        self.batch_size = batch_size
        self._preload = []
        self.camera_name = self.config.il.camera_name
        self.need_extract_instr_features = True
        self.extract_img_features = True
        self.global_batch_size = self.batch_size * len(self.config.torch_gpu_ids)

    def _create_new_data(self, data, yaws, instruction):
        """Helper function to create new data entry"""
        if isinstance(data['action'][-1], list):
            data['action'] = data['action'][:-1] + data['action'][-1]
            data['action'] = np.array(data['action'])

        new_data = {
            'instruction': instruction,
            'progress': data['progress'],
            'globalgps': data['robot_info']['position'],
            'global_rotation': data['robot_info']['orientation'],
            'globalyaw': yaws,
            'gt_actions': np.array(data['action']),
            'prev_actions': np.concatenate([np.array([0]), data['action'][:-1]]),
        }

        # Handle RGB and depth features/data
        new_data['rgb'] = data['camera_info'][self.camera_name]['rgb']
        depth_shape = data['camera_info'][self.camera_name]['depth'].shape  # empty data is filtered
        if len(depth_shape) == 3:
            # [N, 256, 256] -> [N, 256, 256, 1]
            new_data['depth'] = np.expand_dims(data['camera_info'][self.camera_name]['depth'], axis=-1)
        else:
            new_data['depth'] = data['camera_info'][self.camera_name]['depth']

        return new_data

    def _load_next(self):
        raise NotImplementedError

    def __next__(self):
        obs = self._load_next()
        return obs

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
