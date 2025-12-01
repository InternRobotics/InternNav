import collections
import json
import os

import lmdb
import msgpack_numpy

from internnav import PROJECT_ROOT_PATH
from internnav.configs.evaluator import EvalDatasetCfg
from internnav.env.utils.episode_loader.dataset_utils import load_data
from internnav.evaluator.utils.config import get_lmdb_path

from .config import Config


class ResultLogger:
    def __init__(self, dataset_cfg: EvalDatasetCfg):
        if isinstance(dataset_cfg.dataset_settings, dict):
            config = Config(**dataset_cfg.dataset_settings)
        self.name = config.task_name
        self.lmdb_path = get_lmdb_path(self.name)
        self.dataset_type = dataset_cfg.dataset_type
        self.split_map = self.get_split_map(
            base_data_dir=config.base_data_dir,
            split_data_types=config.split_data_types,
            filter_stairs=config.filter_stairs,
            dataset_type=self.dataset_type,
        )

    def get_split_map(
        self,
        base_data_dir,
        split_data_types,
        filter_stairs,
        dataset_type='mp3d',
    ):
        split_map = {}
        for split_data_type in split_data_types:
            load_data_map = load_data(
                base_data_dir,
                split_data_type,
                filter_same_trajectory=False,
                filter_stairs=filter_stairs,
                dataset_type=dataset_type,
            )
            path_key_list = []
            for scan, path_list in load_data_map.items():
                for path in path_list:
                    trajectory_id = path['trajectory_id']
                    episode_id = path['episode_id']
                    path_key = f'{trajectory_id}_{episode_id}'
                    path_key_list.append(path_key)
            split_map[split_data_type] = path_key_list
        return split_map

    def write_now_result_json(self):
        self.database_read = lmdb.open(
            f'{self.lmdb_path}/sample_data.lmdb',
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            readonly=True,
            lock=False,
        )
        json_data = {}
        for split, path_key_list in self.split_map.items():
            data_list = []
            for path_key in path_key_list:
                data_key = path_key
                with self.database_read.begin() as txn:
                    value = txn.get(data_key.encode())
                    if value is None:
                        continue
                    value = msgpack_numpy.unpackb(value)
                value['path_key'] = path_key
                data_list.append(value)
            count = len(data_list)
            total_TL = 0
            total_NE = 0
            total_osr = 0
            total_success = 0
            total_spl = 0
            reason_map = {'reach_goal': 0}

            for data in data_list:
                # TL Trajectory Length (TL) - trajectory total length (0)
                TL = data['info']['TL']
                # NE Navigation Error (NE) - Euclidean distance from current position to goal (-1)
                NE = data['info']['NE']
                if NE < 0:
                    NE = 0
                # OS Oracle Success Rate (OSR) - whether there is a point in the trajectory reaching the goal (-1)
                osr = data['info']['osr']
                if osr < 0:
                    osr = 0
                # SR Success Rate (SR) - whether the goal is reached (0)
                success = data['info']['success']
                # SPL (Success weighted by Path Length)(0)
                spl = data['info']['spl']

                total_TL += TL
                total_NE += NE
                total_osr += osr
                total_success += success
                total_spl += spl

                ret_type = data['fail_reason']
                if ret_type == '':
                    ret_type = 'success'
                if ret_type not in reason_map:
                    reason_map[ret_type] = 1
                else:
                    reason_map[ret_type] = reason_map[ret_type] + 1
                if success > 0:
                    reason_map['reach_goal'] = reason_map['reach_goal'] + 1

            if count == 0:
                continue
            json_data[split] = {}
            json_data[split]['TL'] = round((total_TL / count), 4)
            json_data[split]['NE'] = round((total_NE / count), 4)
            if 'fall' not in reason_map:
                reason_map['fall'] = 0
            json_data[split]['FR'] = round((reason_map['fall'] / count), 4)
            if 'stuck' in reason_map:
                json_data[split]['StR'] = round((reason_map['stuck'] / count), 4)
            else:
                json_data[split]['StR'] = 0
            json_data[split]['OS'] = round((total_osr / count), 4)
            json_data[split]['SR'] = round((total_success / count), 4)
            json_data[split]['SPL'] = round((total_spl / count), 4)
            json_data[split]['Count'] = count

        # write log content to file
        with open(f'{self.dataset_type}_result.json', 'w') as f:
            json.dump(json_data, f)
        self.database_read.close()

    def write_now_result(self):

        # create log file
        log_content = []

        def log_print(content):
            log_content.append(str(content))

        self.database_read = lmdb.open(
            f'{self.lmdb_path}/sample_data.lmdb',
            map_size=1 * 1024 * 1024 * 1024 * 1024,
            readonly=True,
            lock=False,
        )

        for split, path_key_list in self.split_map.items():
            data_list = []
            for path_key in path_key_list:
                data_key = path_key
                with self.database_read.begin() as txn:
                    value = txn.get(data_key.encode())
                    if value is None:
                        continue
                    value = msgpack_numpy.unpackb(value)
                value['path_key'] = path_key
                data_list.append(value)
            count = len(data_list)
            log_print(f'[split:{split}] total {count} data')
            total_TL = 0
            total_NE = 0
            total_osr = 0
            total_success = 0
            total_spl = 0
            reason_map = {'reach_goal': 0}

            for data in data_list:
                # TL Trajectory Length (TL) - trajectory total length (0)
                TL = data['info']['TL']
                # NE Navigation Error (NE) - Euclidean distance from current position to goal (-1)
                NE = data['info']['NE']
                if NE < 0:
                    NE = 0
                # OS Oracle Success Rate (OSR) - whether there is a point in the trajectory reaching the goal (-1)
                osr = data['info']['osr']
                if osr < 0:
                    osr = 0
                # SR Success Rate (SR) - whether the goal is reached (0)
                success = data['info']['success']
                # SPL (Success weighted by Path Length)(0)
                spl = data['info']['spl']

                total_TL += TL
                total_NE += NE
                total_osr += osr
                total_success += success
                total_spl += spl

                ret_type = data['fail_reason']
                if ret_type == '':
                    ret_type = 'success'
                if ret_type not in reason_map:
                    reason_map[ret_type] = 1
                else:
                    reason_map[ret_type] = reason_map[ret_type] + 1
                if success > 0:
                    reason_map['reach_goal'] = reason_map['reach_goal'] + 1

            log_print(f'############[{split}]#############')
            if count == 0:
                log_print('############[count == 0,skip]#############')
                continue
            log_print(f'TL = {total_TL} / {count} = {round((total_TL / count),4)}')
            log_print(f'NE = {total_NE} / {count} = {round((total_NE / count),4)}')
            if 'fall' not in reason_map:
                reason_map['fall'] = 0
            log_print(f"FR = {reason_map['fall']} / {count} = {round((reason_map['fall'] / count),4) * 100}%")
            if 'stuck' in reason_map:
                log_print(f"StR = {reason_map['stuck']} / {count} = {round((reason_map['stuck'] / count),4) * 100}%")
            else:
                log_print(f'StR = 0 / {count} = 0%')
            log_print(f'OS = {total_osr} / {count} = {round((total_osr / count),4) * 100}%')
            log_print(f'SR = {total_success} / {count} = {round((total_success / count),4) * 100}%')
            log_print(f'SPL = {total_spl} / {count} = {round((total_spl / count),4) * 100}%')
            log_print('detail:')
            for k, v in reason_map.items():
                log_print(f'[{k}]:{v}')
            log_print('##########################')

        # write log content to file
        log_file_path = os.path.join(self.lmdb_path, 'eval.log')
        log_file_path1 = f'{PROJECT_ROOT_PATH}/logs/{self.name}/eval_result.log'
        with open(log_file_path, 'w') as f:
            f.write('\n'.join(log_content))
        with open(log_file_path1, 'w') as f:
            f.write('\n'.join(log_content))

        self.database_read.close()

    def finalize_all_results(self, rank, world_size):
        # accumulator for all splits across all ranks
        split_acc = {}
        for split in self.split_map.keys():
            split_acc[split] = {
                "total_TL": 0.0,
                "total_NE": 0.0,
                "total_osr": 0.0,
                "total_success": 0.0,
                "total_spl": 0.0,
                "reason_map": collections.Counter({"reach_goal": 0}),
                "count": 0,
            }

        # loop over all ranks' lmdbs
        for i in range(world_size):
            lmdb_dir = f"{self.lmdb_path}/sample_data{i}.lmdb"
            if not os.path.exists(lmdb_dir):
                # this rank might not have produced a db; skip
                continue

            env = lmdb.open(
                lmdb_dir,
                readonly=True,
                lock=False,
                max_readers=256,
            )

            for split, path_key_list in self.split_map.items():
                for path_key in path_key_list:
                    with env.begin() as txn:
                        value = txn.get(path_key.encode())
                    if value is None:
                        continue

                    data = msgpack_numpy.unpackb(value)
                    data["path_key"] = path_key

                    acc = split_acc[split]

                    TL = data["info"]["TL"]
                    NE = data["info"]["NE"]
                    if NE < 0:
                        NE = 0
                    osr = data["info"]["osr"]
                    if osr < 0:
                        osr = 0
                    success = data["info"]["success"]
                    spl = data["info"]["spl"]

                    acc["total_TL"] += TL
                    acc["total_NE"] += NE
                    acc["total_osr"] += osr
                    acc["total_success"] += success
                    acc["total_spl"] += spl
                    acc["count"] += 1

                    ret_type = data.get("fail_reason", "") or "success"
                    acc["reason_map"][ret_type] += 1
                    if success > 0:
                        acc["reason_map"]["reach_goal"] += 1

            env.close()

        # build final json
        json_data = {}
        for split, acc in split_acc.items():
            count = acc["count"]
            if count == 0:
                continue

            reason_map = acc["reason_map"]
            fall = reason_map.get("fall", 0)
            stuck = reason_map.get("stuck", 0)

            json_data[split] = {
                "TL": round(acc["total_TL"] / count, 4),
                "NE": round(acc["total_NE"] / count, 4),
                "FR": round(fall / count, 4),
                "StR": round(stuck / count, 4),
                "OS": round(acc["total_osr"] / count, 4),
                "SR": round(acc["total_success"] / count, 4),
                "SPL": round(acc["total_spl"] / count, 4),
                "Count": count,
            }

        # write log content to file
        with open(f"{self.name}_result.json", "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
