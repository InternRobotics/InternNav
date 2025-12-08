import os
import time

import lmdb
import msgpack_numpy
import numpy as np


class DataCollector:
    def __init__(self, lmdb_path, rank=0, world_size=1):
        if not os.path.exists(lmdb_path):
            os.makedirs(lmdb_path)
        self.lmdb_path = lmdb_path
        self.episode_total_data = []
        self.actions = []
        self.rank = rank
        self.world_size = world_size

    def collect_observation(self, rgb, depth, step, process, camera_pose, robot_pose):
        from omni.isaac.core.utils.rotations import quat_to_euler_angles

        episode_data = {
            'camera_info': {},
            'robot_info': {},
            'step': step,
            'progress': process,
        }
        c_pos, c_quat = camera_pose[0], camera_pose[1]
        _, _, c_yaw = quat_to_euler_angles(c_quat)
        episode_data['camera_info']['pano_camera_0'] = {
            'rgb': rgb,
            'depth': depth,
            'position': c_pos.tolist(),
            'orientation': c_quat.tolist(),
            'yaw': c_yaw,
        }
        r_pos, r_quat = robot_pose[0], robot_pose[1]
        _, _, r_yaw = quat_to_euler_angles(r_quat)
        episode_data['robot_info'] = {
            'position': r_pos.tolist(),
            'orientation': r_quat.tolist(),
            'yaw': r_yaw,
        }
        self.episode_total_data.append(episode_data)

    def collect_action(self, action):
        self.actions.append(action)

    def merge_data(self, episode_datas, actions):
        camera_info_dict = {}
        robot_info_list = {
            'position': [],
            'orientation': [],
            'yaw': [],
        }
        progress_list = []
        step_list = []

        for episode_data in episode_datas:
            for camera, info in episode_data['camera_info'].items():
                if camera not in camera_info_dict:
                    camera_info_dict[camera] = {
                        'rgb': [],
                        'depth': [],
                        'position': [],
                        'orientation': [],
                        'yaw': [],
                    }

                camera_info_dict[camera]['rgb'].append(info['rgb'])
                camera_info_dict[camera]['depth'].append(info['depth'])
                camera_info_dict[camera]['position'].append(info['position'])
                camera_info_dict[camera]['orientation'].append(info['orientation'])
                camera_info_dict[camera]['yaw'].append(info['yaw'])

            robot_info_list['position'].append(episode_data['robot_info']['position'])
            robot_info_list['orientation'].append(episode_data['robot_info']['orientation'])
            robot_info_list['yaw'].append(episode_data['robot_info']['yaw'])

            step_list.append(episode_data['step'])
            progress_list.append(episode_data['progress'])

        for camera, info in camera_info_dict.items():
            for key, values in info.items():
                camera_info_dict[camera][key] = np.array(values)

        for key, values in robot_info_list.items():
            robot_info_list[key] = np.array(values)

        collate_data = {
            'camera_info': camera_info_dict,
            'robot_info': robot_info_list,
            'progress': np.array(progress_list),
            'step': np.array(step_list),
            'action': actions,
        }

        return collate_data

    def get_timestamp(self):
        data = {'timestamp': time.time()}
        return msgpack_numpy.packb(data, use_bin_type=True)

    def save_sample_data(self, key, result, instruction):
        finish_flag = result
        if result != 'success':
            finish_flag = 'fail'
        lmdb_file = os.path.join(self.lmdb_path, 'sample_data.lmdb')
        database = lmdb.open(
            lmdb_file,
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            max_dbs=0,
            lock=True,
        )
        with database.begin(write=True) as txn:
            encode_key = key.encode()
            episode_datas = self.merge_data(self.episode_total_data, self.actions)
            data_to_store = {
                'episode_data': episode_datas,
                'finish_status': finish_flag,
                'fail_reason': result,
                'instruction': instruction,
            }
            serialized_data = msgpack_numpy.packb(data_to_store, use_bin_type=True)
            txn.put(encode_key, serialized_data)
            txn.put('timestamp_'.encode(), self.get_timestamp())
        database.close()
        self.episode_total_data = []
        self.actions = []

    def save_eval_result(self, key, result, info):
        finish_flag = result
        if result != 'success':
            finish_flag = 'fail'
        database_write = lmdb.open(
            f'{self.lmdb_path}/sample_data{self.rank}.lmdb',
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            max_dbs=0,
            lock=True,
        )
        with database_write.begin(write=True) as txn:
            key_write = key.encode()
            data_to_store = {
                'info': info,
                'finish_status': finish_flag,
                'fail_reason': result,
            }
            value_write = msgpack_numpy.packb(data_to_store, use_bin_type=True)
            txn.put(key_write, value_write)
            txn.put('timestamp_'.encode(), self.get_timestamp())
        database_write.close()
