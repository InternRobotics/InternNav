#!/usr/bin/env python

import glob
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import cv2
import datasets
import numpy as np
import torch
import torchvision
import tqdm
from datasets import concatenate_datasets
from lerobot.common.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    embed_images,
    get_episode_data_index,
    hf_transform_to_torch,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
)
from lerobot.common.datasets.video_utils import get_safe_default_codec
from loguru import logger

LEROBOT_HOME = Path(os.environ.get("LEROBOT_HOME", "/shared/smartbot_new/liuyu/"))


def sample_images(input):
    if type(input) is str:
        video_path = input
        reader = torchvision.io.VideoReader(video_path, stream="video")
        frames = [frame["data"] for frame in reader]
        frames_array = torch.stack(frames).numpy()  # Shape: [T, C, H, W]

        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

            images[i] = img
    elif type(input) is np.ndarray:
        frames_array = input[:, None, :, :]  # Shape: [T, C, H, W]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

            images[i] = img

    return images


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    """calculate episode statistics"""
    ep_stats = {}
    for key, data in episode_data.items():
        if key not in features:  # skip non-feature data
            continue

        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            if isinstance(data, (str, list)) and all(
                isinstance(item, str) for item in (data if isinstance(data, list) else [data])
            ):
                # string path, skip stats calculation
                continue
            # ensure data is in the correct shape
            ep_ft_array = np.array(data)
            if len(ep_ft_array.shape) == 3:  # [H, W, C]
                ep_ft_array = ep_ft_array[np.newaxis, ...]  # add time dimension [1, H, W, C]
            axes_to_reduce = (0,)  # calculate stats only on time dimension
            keepdims = True
        else:
            # for non-image/video data, ensure it's a 2D array [N, D]
            ep_ft_array = np.array(data)
            if ep_ft_array.ndim == 1:
                if key == "episode_index":
                    ep_ft_array = ep_ft_array.reshape(-1, 1)
                else:
                    feature_shape = features[key]["shape"]
                    if len(feature_shape) > 1:
                        ep_ft_array = ep_ft_array.reshape(-1, np.prod(feature_shape))
                    else:
                        ep_ft_array = ep_ft_array.reshape(-1, 1)

            axes_to_reduce = (0,)  # calculate stats on the first dimension
            keepdims = True

        try:
            ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

            if features[key]["dtype"] in ["image", "video"]:
                value_norm = 1.0 if "depth" in key else 255.0
                ep_stats[key] = {k: v if k == "count" else np.squeeze(v / value_norm) for k, v in ep_stats[key].items()}
        except Exception as e:
            logger.warning(f"Failed to calculate stats for feature {key}: {e}")
            continue

    return ep_stats


class NavDatasetMetadata(LeRobotDatasetMetadata):
    def get_data_file_path(self, ep_index: int) -> Path:
        chunk = self.get_episode_chunk(ep_index)
        return Path("data") / f"chunk-{chunk:03d}" / f"episode_{ep_index:06d}.parquet"

    def get_video_file_path(self, ep_index: int, key: str) -> Path:
        chunk = self.get_episode_chunk(ep_index)
        video_key = key.split(".")[-1]
        return Path("videos") / f"chunk-{chunk:03d}" / video_key

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        # action_config: list[dict],
    ) -> None:
        """extend the base class's save_episode method, add action_config support"""
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
            # "action_config": action_config,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


class NavDataset(LeRobotDataset):
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "NavDataset":
        obj = cls.__new__(cls)
        obj.meta = NavDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        obj.episode_buffer = obj.create_episode_buffer()
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:

        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        features = {key: value for key, value in self.features.items() if key in self.hf_features}
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key, value in frame.items():
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, files: dict) -> None:
        """extend the base class's save_episode method, add video file copying and image directory copying support"""
        if not self.episode_buffer:
            return

        episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        image_data = {}
        for key, ft in self.features.items():
            if ft["dtype"] in ["image"]:
                image_data[key] = episode_buffer[key]

        for key, ft in self.features.items():
            # index, episode_index, task_index already processed, image and video are handled separately
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video", "image"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        # handle video files - copy existing mp4 files
        for key in self.meta.video_keys:
            if key in files:
                video_path = self.root / self.meta.get_video_file_path(episode_index, key)
                video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(files[key], video_path)
                episode_buffer[key] = str(video_path)  # PosixPath -> str

        for key, source_path in files.items():
            if key.startswith("observation.images."):

                video_path = self.root / self.meta.get_video_file_path(episode_index, key)
                video_path.parent.mkdir(parents=True, exist_ok=True)

                source_dir = Path(source_path)
                if source_dir.exists():
                    for img_file in source_dir.glob("*"):
                        if img_file.is_file():
                            shutil.copy2(img_file, video_path.parent / img_file.name)

        for key, data in image_data.items():
            episode_buffer[key] = data

        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self._save_episode_table(episode_buffer, episode_index)

        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        """save episode data to parquet file"""
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)


def get_streamvln_features() -> Dict[str, Dict]:
    """
    define the feature structure of StreamVLN dataset

    Args:
        img_size: image size (height, width)

    Returns:
        feature definition dictionary
    """
    return {
        "observation.images.rgb": {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
        "action": {"dtype": "int64", "shape": (1,), "names": ["action_index"]},
    }


def load_streamvln_episode(
    ann: Dict[str, Any],
    dataset_name: str,
    data_dir: Path,
    # img_size: Tuple[int, int] = (224, 224)
) -> Iterator[Dict[str, Any]]:
    """
    load StreamVLN episode data, return an iterator in LeRobot format

    Args:
        ann: single annotation dictionary
        dataset_name: dataset name (EnvDrop/R2R/RxR)
        data_dir: data root directory
        img_size: output image size (height, width)

    Yields:
        a dictionary of LeRobot format data for each frame
    """
    try:
        ann_id = ann["id"]
        video_path = ann["video"]

        # parse scene ID and episode ID
        parts = video_path.split("/")[-1].split("_")
        scene_id = parts[0]
        ann_id = parts[-1]
        # fix path parsing logic
        # original format: "video": "images/17DRP5sb8fy_envdrop_111702"
        # actual path: images/17DRP5sb8fy/rgb

        # src_image_dir = data_dir / dataset_name / "images" / "rgb" /scene_id

        # build source image directory
        src_image_dir = data_dir / dataset_name / video_path / "rgb"

        # get all image files
        image_files = sorted(glob.glob(str(src_image_dir / "*.jpg")))
        if not image_files:
            logger.warning(f"No image files found in {src_image_dir}")
            return

        # get actions and instructions
        actions = np.array(ann.get("actions", []), dtype=np.int64)
        instructions = ann.get("instructions", [])
        instruction = json.dumps({"instruction": instructions[0]}) if instructions else "Navigation task"

        # build file path mapping
        files = {"observation.images.rgb": str(src_image_dir)}

        for frame_idx, img_path in enumerate(image_files):
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            action_value = -1
            if frame_idx < len(actions):
                action_value = actions[frame_idx]

            action = np.array([action_value], dtype=np.int64)

            yield {
                'observation': {
                    'images.rgb': img,
                },
                'action': action,
                'language_instruction': instruction,
                'files': files,
            }

    except Exception as e:
        logger.error(f"Failed to load episode {ann_id}: {str(e)}", exc_info=True)
        return


def process_episode(
    ann: Dict[str, Any],
    dataset_name: str,
    data_dir: Path,
    repo_name: str,
    push_to_hub: bool,
) -> Tuple[str, bool, str]:

    try:
        episode_id = f"{dataset_name}_{ann['id']}"
        video_path = ann["video"]
        parts = video_path.split("/")[-1].split("_")
        scene_id = parts[0]
        ep_id = parts[-1] if len(parts) > 2 else "000000"

        output_path = LEROBOT_HOME / repo_name / dataset_name.lower() / scene_id / ep_id

        if output_path.exists():
            return (episode_id, True, "Skipped, already exists")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = get_streamvln_features()

        dataset = NavDataset.create(
            repo_id=f"{repo_name}_{dataset_name.lower()}",
            root=output_path,
            robot_type="unknown",
            fps=30,
            use_videos=True,
            features=features,
        )

        episode_iterator = load_streamvln_episode(ann, dataset_name, data_dir)
        frame_count = 0
        files = {}

        for step_data in episode_iterator:
            if frame_count == 0:
                files = step_data.pop('files', {})
            else:
                step_data.pop('files', {})

            dataset.add_frame(
                frame={
                    "observation.images.rgb": step_data["observation"]["images.rgb"],
                    "action": step_data["action"],
                },
                task=step_data["language_instruction"],
            )
            frame_count += 1

        if frame_count > 0:
            dataset.save_episode(files=files)
            message = f"Successfully processed: {episode_id}, {frame_count} frames"
            return (episode_id, True, message)
        else:
            message = f"No frames were processed, skipping: {episode_id}"
            return (episode_id, False, message)

    except Exception as e:
        message = f"Failed to process episode: {str(e)}"
        logger.error(message, exc_info=True)
        return (ann.get('id', 'unknown'), False, message)


def process_dataset(
    dataset_name: str,
    data_dir: Path,
    repo_name: str,
    # img_size: Tuple[int, int],
    push_to_hub: bool,
    num_threads: int = 10,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> Tuple[int, int]:
    """
    process the entire dataset

    Args:
        dataset_name: dataset name
        data_dir: data root directory
        repo_name: output dataset name
        img_size: image size
        push_to_hub: whether to push to Hub
        num_threads: number of threads

    Returns:
        (total_episodes, success_episodes)
    """
    # load annotations
    ann_file = data_dir / dataset_name / "annotations.json"
    if not ann_file.exists():
        logger.error(f"Annotation file not found: {ann_file}")
        return 0, 0

    with open(ann_file, "r") as f:
        annotations = json.load(f)

    total = len(annotations)
    end_idx = end_idx if end_idx is not None else total
    selected_anns = annotations[start_idx:end_idx]
    selected_count = len(selected_anns)

    if selected_count == 0:
        logger.warning(f"No episodes found in the index range [{start_idx}, {end_idx})")
        return 0, 0

    logger.info(
        f"Start processing dataset: {dataset_name} "
        f"(Total episodes: {total}, processing range: [{start_idx}, {end_idx}), actual processing: {selected_count})"
    )
    success_count = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                process_episode,
                ann=ann,
                dataset_name=dataset_name,
                data_dir=data_dir,
                repo_name=repo_name,
                # img_size=img_size,
                push_to_hub=push_to_hub,
            ): ann['id']
            for ann in selected_anns
        }

        progress_bar = tqdm.tqdm(
            as_completed(futures), total=len(selected_anns), desc=f"处理 {dataset_name} [{start_idx}:{end_idx}]"
        )
        for future in progress_bar:
            _, success, message = future.result()
            if success:
                success_count += 1
            progress_bar.set_postfix_str(
                f"Success: {success_count}/{selected_count} " f"({success_count/selected_count:.1%})"
            )

    return selected_count, success_count


def main(
    data_dir: str,
    repo_name: str = "nav_S1",
    output_dir: str | None = None,
    push_to_hub: bool = False,
    # img_height: int = 224,
    # img_width: int = 224,
    num_threads: int = 10,
    start_index: int = None,
    end_index: int = None,
    datasets: str = None,
):
    """
    main function

    Args:
        data_dir: data root directory
        repo_name: output dataset name
        output_dir: output directory
        push_to_hub: whether to push to Hub
        img_height: image height
        img_width: image width
        num_threads: number of threads
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # datasets_to_process = ["R2R", "EnvDrop", "RxR"]

    total_episodes = 0
    success_episodes = 0
    dataset_name = datasets
    # for dataset_name in datasets:

    total, success = process_dataset(
        dataset_name=dataset_name,
        data_dir=data_path,
        repo_name=repo_name,
        # img_size=img_size,
        push_to_hub=push_to_hub,
        num_threads=num_threads,
        start_idx=start_index,
        end_idx=end_index,
    )
    total_episodes += total
    success_episodes += success

    logger.info("=" * 50)
    logger.info("Conversion completed!")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Success: {success_episodes}")
    logger.info(f"Failed: {total_episodes - success_episodes}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert StreamVLN dataset to LeRobot format")
    parser.add_argument("--data_dir", type=str, default="/path/to/streamvln", help="StreamVLN data root directory")
    parser.add_argument("--repo_name", type=str, default="vln_ce_lerobot", help="Output dataset name")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: LEROBOT_HOME)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of threads")
    parser.add_argument("--start_index", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end_index", type=int, default=2000, help="End index (exclusive)")
    parser.add_argument("--datasets", type=str, default="RxR", help="List of datasets to process")

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        repo_name=args.repo_name,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        num_threads=args.num_threads,
        start_index=args.start_index,
        end_index=args.end_index,
        datasets=args.datasets,
    )
