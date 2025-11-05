import lmdb
import msgpack_numpy

from internnav.evaluator.utils.config import get_lmdb_path

from .data_reviser import revise_one_data, skip_list

# from internnav.evaluator.utils.common import load_data
from internnav.evaluator.utils.common import get_load_func


class BasePathKeyEpisodeLoader:
    def __init__(
        self,
        dataset_type,
        base_data_dir,
        split_data_types,
        robot_offset,
        filter_same_trajectory,
        revise_data=True,
        filter_stairs=True,
    ):
        self.path_key_data = {}
        self.path_key_scan = {}
        self.path_key_split = {}

        for split_data_type in split_data_types:
            load_data_map = get_load_func(dataset_type)(
                base_data_dir,
                split_data_type,
                filter_same_trajectory=filter_same_trajectory,
                filter_stairs=filter_stairs,
            )
            for scan, path_list in load_data_map.items():
                for path in path_list:
                    trajectory_id = path['trajectory_id']
                    if revise_data:
                        if trajectory_id in skip_list:
                            continue
                        path = revise_one_data(path)
                    episode_id = path['episode_id']
                    path_key = f'{trajectory_id}_{episode_id}'
                    path['start_position'] += robot_offset
                    for i, _ in enumerate(path['reference_path']):
                        path['reference_path'][i] += robot_offset
                    self.path_key_data[path_key] = path
                    self.path_key_scan[path_key] = scan
                    self.path_key_split[path_key] = split_data_type


class ResumablePathKeyEpisodeLoader(BasePathKeyEpisodeLoader):
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
        )
        self.task_name = task_name
        self.run_type = run_type
        self.lmdb_path = get_lmdb_path(task_name)
        self.retry_list = retry_list
        database = lmdb.open(
            f'{self.lmdb_path}/sample_data.lmdb',
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            readonly=True,
            lock=False,
        )

        filtered_target_path_key_list = []
        for path_key in self.path_key_data.keys():
            trajectory_id = int(path_key.split('_')[0])
            if trajectory_id in skip_list:
                continue
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
