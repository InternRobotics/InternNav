from .dataset_utils import load_data, revise_one_data, skip_list


class BasePathKeyEpisodeloader:
    def __init__(
        self,
        dataset_type,
        base_data_dir,
        split_data_types,
        robot_offset,
        filter_same_trajectory,
        revise_data=True,
        filter_stairs=True,
        rank=0,
        world_size=1,
    ):
        # current supported dataset types in InternUtopia
        # only kujiale has special scene path
        # others type should be considered the same as mp3d in loading
        allowed = ('R2RVLN', 'mp3d', 'kujiale', 'grscene')
        assert dataset_type in allowed, f"Unsupported dataset type: {dataset_type}. Allowed: {allowed}"

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
                rank=rank,
                world_size=world_size,
            )
            for scan, path_list in load_data_map.items():
                for path in path_list:
                    trajectory_id = path['trajectory_id']

                    # tiny revision for R2R dataset in MP3D to fit vlnpe task
                    if dataset_type == 'mp3d' and revise_data:
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
