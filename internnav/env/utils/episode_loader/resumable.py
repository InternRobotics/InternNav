import os

import lmdb
import msgpack_numpy

from internnav.evaluator.utils.config import get_lmdb_path

from .base import BasePathKeyEpisodeloader


class ResumablePathKeyEpisodeloader(BasePathKeyEpisodeloader):
    def __init__(
        self,
        dataset_type,
        base_data_dir,
        split_data_types,
        robot_offset,
        filter_same_trajectory,
        task_name,
        run_type,
        retry_list,
        filter_stairs,
        rank=0,
        world_size=1,
    ):
        # 加载所有数据
        super().__init__(
            dataset_type=dataset_type,
            base_data_dir=base_data_dir,
            split_data_types=split_data_types,
            robot_offset=robot_offset,
            filter_same_trajectory=filter_same_trajectory,
            revise_data=True,
            filter_stairs=filter_stairs,
            rank=rank,
            world_size=world_size,
        )
        self.task_name = task_name
        self.run_type = run_type
        self.lmdb_path = get_lmdb_path(task_name)
        self.retry_list = retry_list

        if not os.path.exists(self.lmdb_path):
            os.makedirs(self.lmdb_path, exist_ok=True)

        database = lmdb.open(
            f'{self.lmdb_path}/sample_data{rank}.lmdb',
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            lock=True,
        )

        filtered_target_path_key_list = []
        for path_key in self.path_key_data.keys():
            # trajectory_id = int(path_key.split('_')[0])
            with database.begin() as txn:
                value = txn.get(path_key.encode())
                if value is None:
                    filtered_target_path_key_list.append(path_key)
                else:
                    value = msgpack_numpy.unpackb(value)
                    if value['finish_status'] == 'success':
                        if 'success' in self.retry_list:
                            filtered_target_path_key_list.append(path_key)
                        else:
                            continue
                    else:
                        fail_reason = value['fail_reason']
                        if fail_reason in retry_list:
                            filtered_target_path_key_list.append(path_key)

        filtered_target_path_key_list.reverse()
        self.resumed_path_key_list = filtered_target_path_key_list
        database.close()

    @property
    def size(self):
        return len(self.resumed_path_key_list)
