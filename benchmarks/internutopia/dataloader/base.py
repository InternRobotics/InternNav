from internnav.evaluator.utils.common import load_data

from .data_reviser import revise_one_data, skip_list


class BasePathKeyDataloader:
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
            load_data_map = load_data(
                base_data_dir,
                split_data_type,
                filter_same_trajectory=filter_same_trajectory,
                filter_stairs=filter_stairs,
                dataset_type=dataset_type,
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
