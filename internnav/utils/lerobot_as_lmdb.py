# Accessing the lerobot dataset using the LMDB interface
import json
import os

import numpy as np
import pandas as pd


class LerobotAsLmdb:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_all_keys(self):
        keys = []
        for scan in os.listdir(self.dataset_path):
            scan_path = os.path.join(self.dataset_path, scan)
            if not os.path.isdir(scan_path):
                continue
            for scene_index in os.listdir(scan_path):
                scene_path = os.path.join(scan_path, scene_index)
                if not os.path.isdir(scene_path):
                    continue

                data_dir = os.path.join(scene_path, "data")
                if os.path.exists(data_dir):
                    for chunk_dir in os.listdir(data_dir):
                        if chunk_dir.startswith("chunk-"):
                            chunk_path = os.path.join(data_dir, chunk_dir)
                            chunk_idx = int(chunk_dir.split("-")[1])

                            for file in os.listdir(chunk_path):
                                if file.startswith("episode_") and file.endswith(".parquet"):
                                    episode_idx = int(file.split("_")[1].split(".")[0])
                                    keys.append("{}_{:03d}_{:06d}".format(scene_index, chunk_idx, episode_idx))
                else:
                    for trajectory in os.listdir(scene_path):
                        trajectory_path = os.path.join(scene_path, trajectory)
                        if not os.path.isdir(trajectory_path):
                            continue
                        keys.append("{}_000_{:06d}".format(scene_index, trajectory))
        return keys

    def get_data_by_key(self, key):
        parts = key.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid key format: {key}")

        scene_index = "_".join(parts[:-2])
        chunk_idx = int(parts[-2])
        episode_idx = int(parts[-1])

        dataset_name = None
        for dn in os.listdir(self.dataset_path):
            dataset_dir = os.path.join(self.dataset_path, dn)
            if os.path.isdir(dataset_dir) and scene_index in os.listdir(dataset_dir):
                dataset_name = dn
                break

        if dataset_name is None:
            raise ValueError(f"Scene index {scene_index} not found")

        base_path = os.path.join(self.dataset_path, dataset_name, scene_index)

        chunk_str = "chunk-{:03d}".format(chunk_idx)
        parquet_path = os.path.join(base_path, "data", chunk_str, "episode_{:06d}.parquet".format(episode_idx))
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        stats_path = os.path.join(base_path, "meta", "episodes_stats.jsonl")
        task_min = 0
        task_max = 0

        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                for line in f:
                    try:
                        stats_data = json.loads(line.strip())
                        if stats_data.get("episode_index") == episode_idx:
                            task_info = stats_data.get("task_index", {})
                            task_min = task_info.get("min", 0)
                            task_max = task_info.get("max", 0)
                            break
                    except json.JSONDecodeError as e:
                        print(f"Error decoding stats JSON: {e}")

        tasks_path = os.path.join(base_path, "meta", "tasks.jsonl")
        episodes_in_json = []
        finish_status_in_json = None
        fail_reason_in_json = None

        with open(tasks_path, 'r') as f:
            for line in f:
                try:
                    json_data = json.loads(line.strip())
                    task_index = json_data.get("task_index")

                    if task_index is not None and task_min <= task_index <= task_max:
                        episodes_in_json.append(json_data)

                        finish_status_in_json = json_data.get('finish_status')
                        fail_reason_in_json = json_data.get('fail_reason')
                except json.JSONDecodeError as e:
                    print(f"Error decoding tasks JSON: {e}")

        rgb_path = os.path.join(
            base_path,
            "videos",
            chunk_str,
            "observation.images.rgb",
            "episode_{:06d}.npy".format(episode_idx),
        )
        depth_path = os.path.join(
            base_path, "videos", chunk_str, "observation.images.depth", "episode_{:06d}.npy".format(episode_idx)
        )

        data = {}
        data['episode_data'] = {}
        data['episode_data']['camera_info'] = {}
        data['episode_data']['camera_info']['pano_camera_0'] = {}

        data['episode_data']['camera_info']['pano_camera_0']['position'] = np.array(
            df['observation.camera_position'].tolist()
        )
        data['episode_data']['camera_info']['pano_camera_0']['orientation'] = np.array(
            df['observation.camera_orientation'].tolist()
        )
        data['episode_data']['camera_info']['pano_camera_0']['yaw'] = np.array(df['observation.camera_yaw'].tolist())

        data['episode_data']['robot_info'] = {}
        data['episode_data']['robot_info']['position'] = np.array(df['observation.robot_position'].tolist())
        data['episode_data']['robot_info']['orientation'] = np.array(df['observation.robot_orientation'].tolist())
        data['episode_data']['robot_info']['yaw'] = np.array(df['observation.robot_yaw'].tolist())

        data['episode_data']['progress'] = np.array(df['observation.progress'].tolist())
        data['episode_data']['step'] = np.array(df['observation.step'].tolist())
        data['episode_data']['action'] = df['observation.action'].tolist()

        data["finish_status"] = finish_status_in_json
        data["fail_reason"] = fail_reason_in_json
        data["episodes_in_json"] = episodes_in_json

        data['episode_data']['camera_info']['pano_camera_0']['rgb'] = np.load(rgb_path)
        data['episode_data']['camera_info']['pano_camera_0']['depth'] = np.load(depth_path)

        return data


if __name__ == '__main__':
    ds = LerobotAsLmdb('path_to_vln_pe')

    keys = ds.get_all_keys()
    print(f"total keys: {len(keys)}")
    for k in keys[:5]:
        try:
            o = ds.get_data_by_key(k)
            print(f"Key: {k}")
            print(f"  Finish status: {o.get('finish_status')}")
            print(f"  Tasks in JSON: {len(o.get('episodes_in_json', []))}")
            print(
                f"  RGB data: {'loaded' if o['episode_data']['camera_info']['pano_camera_0'].get('rgb') is not None else 'not found'}"
            )
            print(
                f"  Depth data: {'loaded' if o['episode_data']['camera_info']['pano_camera_0'].get('depth') is not None else 'not found'}"
            )
        except Exception as e:
            print(f"Error processing key {k}: {e}")
