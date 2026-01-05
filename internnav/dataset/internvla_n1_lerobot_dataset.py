import copy
import itertools
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from transformers.image_utils import to_numpy_array

from .rope2d import get_rope_index_2, get_rope_index_25
from .vlln_lerobot_dataset import VLLNDataset

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",  # noqa: F541
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}


R2R_125CM_0_30 = {
    "data_path": "traj_data/r2r",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

R2R_125CM_0_45 = {
    "data_path": "traj_data/r2r",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

R2R_60CM_15_15 = {
    "data_path": "traj_data/r2r",
    "height": 60,
    "pitch_1": 15,
    "pitch_2": 15,
}

R2R_60CM_30_30 = {
    "data_path": "traj_data/r2r",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

RxR_125CM_0_30 = {
    "data_path": "traj_data/rxr",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

RxR_125CM_0_45 = {
    "data_path": "traj_data/rxr",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

RxR_60CM_15_15 = {
    "data_path": "traj_data/rxr",
    "height": 60,
    "pitch_1": 15,
    "pitch_2": 15,
}

RxR_60CM_30_30 = {
    "data_path": "traj_data/rxr",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

SCALEVLN_125CM_0_30 = {
    "data_path": "traj_data/scalevln",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

SCALEVLN_125CM_0_45 = {
    "data_path": "traj_data/scalevln",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

SCALEVLN_60CM_30_30 = {
    "data_path": "traj_data/scalevln",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "r2r_125cm_0_30": R2R_125CM_0_30,
    "r2r_125cm_0_45": R2R_125CM_0_45,
    "r2r_60cm_15_15": R2R_60CM_15_15,
    "r2r_60cm_30_30": R2R_60CM_30_30,
    "rxr_125cm_0_30": RxR_125CM_0_30,
    "rxr_125cm_0_45": RxR_125CM_0_45,
    "rxr_60cm_15_15": RxR_60CM_15_15,
    "rxr_60cm_30_30": RxR_60CM_30_30,
    "scalevln_125cm_0_30": SCALEVLN_125CM_0_30,
    "scalevln_125cm_0_45": SCALEVLN_125CM_0_45,
    "scalevln_60cm_30_30": SCALEVLN_60CM_30_30,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
TRAJ_TOKEN_INDEX = 151667
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_TRAJ_TOKEN = "<traj>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:  # noqa: E722
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:  # noqa: E722
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>" * grid_thw_image[visual_replicate_index_image]  # noqa: F541
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + "<|video_pad|>" * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(images=None, videos=video, return_tensors="pt")
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [self.data_args.image_processor.temporal_patch_size / fps] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3  # noqa: F841
        num_final_retries = 30  # noqa: F841

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [os.path.join(image_folder, file) for file in image_file]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
            ]
        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [os.path.join(video_folder, file) for file in video_file]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(torch.stack(video_grid_thw, dim=0) if video_grid_thw else None),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged)
            position_ids = torch.arange(0, data_dict["input_ids"].size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1)

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in grid_thw], dim=0)
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in video_grid_thw], dim=0)

        return data_dict


def interpolate_and_resample_trajectory(absolute_trajectories, predict_step_num=None):
    start_point = np.array([[0.0, 0.0]])  # Avoid creating arrays repeatedly

    traj = absolute_trajectories[..., :2]
    # Vectorized filtering of valid steps (distance squared > 0.05)
    steps = traj[1:] - traj[:-1]  # (T, 2)
    steps_sq = (steps**2).sum(axis=-1)  # (T,)
    mask = steps_sq > 0.05  # (T,)

    # Filter and concatenate starting point
    filtered_traj = traj[1:][mask]  # (T, 2), where M is the number of filtered steps
    filtered_traj = np.concatenate([start_point, filtered_traj], axis=0)  # (T+1, 2)

    resampled_trajectories = smooth_and_resample_trajectory(filtered_traj, sample_length=predict_step_num + 1)
    resampled_relative_poses = xy_to_delta_xyt(resampled_trajectories)

    resampled_relative_poses[:, 0:2] *= 4  # norm

    return resampled_trajectories, resampled_relative_poses


def get_trajectory_relative_to_frame(extrinsics, camera_deg=0):
    """
    Calculate trajectory poses (x, y, yaw) relative to a reference frame

    Args:
        extrinsics: Sequence of 4x4 extrinsic matrices [T_world2camera], shape: (n, 4, 4), numpy array
        camera_deg: Camera pitch angle

    Returns:
        relative_xyyaw: Pose sequence relative to the reference frame (x, y, yaw), shape: (n, 3), numpy array
    """
    # T_world2camera
    # Coordinate transformation matrices
    T_camera2robot = np.array(
        [[[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    T_robot2camera = np.array(
        [[[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    # Apply camera pitch angle transformation (30 degrees downward)
    if camera_deg is not None:
        camera_rad = np.radians(camera_deg)
        # Clockwise rotation around the x-axis of the level camera is downward view
        T_deg = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, np.cos(-camera_rad), -np.sin(-camera_rad), 0.0],
                    [0.0, np.sin(-camera_rad), np.cos(-camera_rad), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=np.float32,
        )
        T_robot2camera = np.matmul(T_robot2camera, T_deg)
        T_camera2robot = np.linalg.inv(T_robot2camera)

    # Transform to robot coordinate system
    extrinsics_robot = np.matmul(extrinsics, T_camera2robot)

    # Get the transformation matrix of the reference frame and compute its inverse
    T_ref = extrinsics_robot[0]
    T_ref_inv = np.linalg.inv(T_ref)

    # Calculate transformations of all frames relative to the reference frame
    # T_relative = T_ref^{-1} * T_current
    relative_to_ref = np.matmul(T_ref_inv[np.newaxis, :, :], extrinsics_robot)

    # Extract relative poses
    relative_translations = relative_to_ref[:, :2, 3]  # (x, y), only take the xy-plane
    relative_yaws = np.arctan2(relative_to_ref[:, 1, 0], relative_to_ref[:, 0, 0])

    relative_xyyaw = np.concatenate((relative_translations, relative_yaws.reshape(-1, 1)), axis=-1)

    return relative_xyyaw


from scipy.interpolate import CubicSpline


def smooth_and_resample_trajectory(points, sample_length=33, interval=0.1):
    total_distance = sample_length * interval  # Total sampling length

    if len(points) == 0:
        return np.zeros((sample_length, 2))

    if len(points) == 1:
        return np.tile(points[0], (sample_length, 1))

    # Calculate cumulative distance of the original trajectory
    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    cumulative_distances = np.cumsum(segment_lengths)
    cumulative_distances = np.insert(cumulative_distances, 0, 0)  # Starting point distance is 0

    # Use cubic spline interpolation for smoothing
    if len(points) > 3:  # At least 4 points are needed for cubic spline interpolation
        # Construct cubic splines using cumulative distance as the parameter
        cs_x = CubicSpline(cumulative_distances, points[:, 0])
        cs_y = CubicSpline(cumulative_distances, points[:, 1])

        # Perform dense sampling within the original cumulative distance range
        dense_distances = np.linspace(0, cumulative_distances[-1], max(50, len(points) * 2))
        x_smooth = cs_x(dense_distances)
        y_smooth = cs_y(dense_distances)
        smoothed_points = np.column_stack((x_smooth, y_smooth))

        # Recalculate cumulative distance of the smoothed trajectory
        smooth_diff = np.diff(smoothed_points, axis=0)
        smooth_segment_lengths = np.sqrt(np.sum(smooth_diff**2, axis=1))
        smooth_cumulative_distances = np.cumsum(smooth_segment_lengths)
        smooth_cumulative_distances = np.insert(smooth_cumulative_distances, 0, 0)
    else:
        # Too few points for cubic spline interpolation, use original points directly
        smoothed_points = points
        smooth_cumulative_distances = cumulative_distances

    # Target sampling point distances
    target_distances = np.linspace(0, total_distance, sample_length)

    # Initialize result array
    resampled = np.zeros((sample_length, 2))

    # Interpolate for each target distance
    for i, target_dist in enumerate(target_distances):
        # If target distance exceeds total trajectory length, use the last point
        if target_dist >= smooth_cumulative_distances[-1]:
            resampled[i] = smoothed_points[-1]
            continue

        # Find the line segment where the target distance is located
        segment_idx = np.searchsorted(smooth_cumulative_distances, target_dist, side='right') - 1

        # Calculate interpolation ratio
        start_dist = smooth_cumulative_distances[segment_idx]
        end_dist = smooth_cumulative_distances[segment_idx + 1]
        t = (target_dist - start_dist) / (end_dist - start_dist)

        # Linear interpolation
        resampled[i] = smoothed_points[segment_idx] + t * (
            smoothed_points[segment_idx + 1] - smoothed_points[segment_idx]
        )

    return resampled


def xy_to_delta_xyt(xy_actions):
    """
    Compute (dx, dy, delta_yaw) where dx, dy in global frame and delta_yaw is heading difference.

    Args:
        xy_actions: [N, 2] array of absolute positions

    Returns:
        delta_xyt: [N-1, 3] array
    """
    vectors = np.diff(xy_actions, axis=0)  # [N-1, 2]
    yaw = np.arctan2(vectors[:, 1], vectors[:, 0])  # [N-1] yaw angles w.r.t x-axis

    delta_yaw = np.diff(yaw)  # [N-2]
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

    # prepend first yaw (absolute angle of first segment) as delta_yaw[0]
    delta_yaw = np.concatenate([[yaw[0]], delta_yaw])  # now length = N-1

    delta_xyt = np.concatenate([vectors, delta_yaw[:, None]], axis=1)
    return delta_xyt


def clip_or_pad(arr, fixed_len):
    T, D = arr.shape
    if T >= fixed_len:
        return arr[:fixed_len]
    else:
        pad = np.zeros((fixed_len - T, D), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)


def get_annotations_from_lerobot_data(data_path, setting):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pyarrow.parquet as pq

    annotations = {
        "axis_align_matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "episodes": [],
    }
    scene_ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    def process_scene(scene_id):
        scene_path = os.path.join(data_path, scene_id)
        episodes = read_jsonl(os.path.join(scene_path, "meta", "episodes.jsonl"))
        scene_annotations = []

        for ep in episodes:
            ep_id = ep["episode_index"]
            ep_instructions = ep["tasks"][0].split("<INSTRUCTION_SEP>")
            ep_len = ep["length"]
            parquet_path = os.path.join(
                scene_path, "data", f"chunk-{ep_id // 1000:03d}", f"episode_{ep_id:06d}.parquet"
            )

            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            ep_actions = df["action"].tolist()

            pose_key = f"pose.{setting}"
            goal_key = f"goal.{setting}"
            relative_goal_frame_id_key = f"relative_goal_frame_id.{setting}"

            if pose_key in df.columns and goal_key in df.columns and relative_goal_frame_id_key in df.columns:
                ep_poses = df[pose_key].apply(lambda x: x.tolist()).tolist()
                ep_pixel_goals = [
                    [df[relative_goal_frame_id_key][idx].tolist(), df[goal_key][idx].tolist()] for idx in range(len(df))
                ]
            else:
                print(f"Warning: Missing data for setting {setting} in episode {ep_id}, filling with defaults.")

            assert len(ep_actions) == ep_len, f"Action length mismatch in episode {ep_id}"

            for ep_instruction in ep_instructions:
                episode = {
                    "id": ep_id,
                    "video": f"{data_path}/{scene_id}/videos/chunk-{ep_id // 1000:03d}",
                    "instructions": ep_instruction,
                    "actions": ep_actions,
                    "length": ep_len,
                    f"poses_{setting}": ep_poses,
                    "pixel_goals": ep_pixel_goals,
                }
                scene_annotations.append(episode)

        return scene_annotations

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_scene, scene_id): scene_id for scene_id in scene_ids}
        for future in as_completed(futures):
            scene_id = futures[future]
            try:
                scene_annotations = future.result()
                annotations["episodes"].extend(scene_annotations)
            except Exception as e:
                print(f"Error processing scene {scene_id}: {e}")

    return annotations


class NavPixelGoalDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(NavPixelGoalDataset, self).__init__()
        dataset = data_args.vln_dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        self.sample_step = data_args.sample_step
        self.predict_step_num = data_args.predict_step_num
        self.pixel_goal_only = data_args.pixel_goal_only
        self.num_future_steps = data_args.num_future_steps

        self.list_data_dict = []

        for data in dataset_list:
            sampling_rate = data.get("sampling_rate", 1.0)
            height = data.get("height", None)
            pitch_1 = data.get("pitch_1", None)
            pitch_2 = data.get("pitch_2", None)

            data_path = data['data_path']
            setting = f'{height}cm_{pitch_2}deg'
            annotations = get_annotations_from_lerobot_data(data_path, setting)

            pixel_goal_list = []
            turn_list = []
            stop_list = []
            list_data_dict = []
            for item in annotations['episodes']:
                ep_id = item['id']
                instruction = item['instructions']
                video = item['video']
                actions = item['actions'][1:] + [0]
                pixel_goals = item['pixel_goals']
                poses = item[f'poses_{height}cm_{pitch_2}deg']

                actions_len = len(actions)
                if actions_len < 4:
                    continue

                num_rounds = actions_len // self.sample_step
                for n in range(num_rounds + 1):
                    if n * self.sample_step == actions_len or n * self.sample_step == actions_len - 1:
                        continue
                    start_frame_id = n * self.sample_step
                    action_flag = actions[start_frame_id]
                    pixel_goal = pixel_goals[start_frame_id]
                    if pixel_goal[0] == -1:
                        if action_flag == 1:
                            continue
                        else:
                            end_frame_id = min(actions_len, start_frame_id + self.num_future_steps)
                            turn_actions = []
                            for id in range(start_frame_id, end_frame_id):
                                if actions[id] == 1:
                                    break
                                turn_actions.append(actions[id])
                            turn_list.append(
                                (
                                    ep_id,
                                    data_path,
                                    video,
                                    height,
                                    pitch_1,
                                    pitch_2,
                                    instruction,
                                    (start_frame_id, start_frame_id + 1),
                                    turn_actions,
                                    None,
                                )
                            )
                    else:
                        goal_len = pixel_goal[0]
                        if goal_len < 3:
                            continue
                        action = pixel_goal[1]
                        pose = poses[start_frame_id : start_frame_id + goal_len + 1]
                        pixel_goal_list.append(
                            (
                                ep_id,
                                data_path,
                                video,
                                height,
                                pitch_1,
                                pitch_2,
                                instruction,
                                (start_frame_id, start_frame_id + goal_len + 1),
                                action,
                                pose,
                            )
                        )

                stop_list.append(
                    (
                        ep_id,
                        data_path,
                        video,
                        height,
                        pitch_1,
                        pitch_2,
                        instruction,
                        (actions_len - 1, actions_len),
                        0,
                        None,
                    )
                )

            list_data_dict = pixel_goal_list
            rank0_print(len(turn_list), len(pixel_goal_list), len(stop_list))
            if not self.pixel_goal_only:
                list_data_dict += turn_list
                list_data_dict += stop_list * 5
            if sampling_rate < 1.0:
                list_data_dict = random.sample(list_data_dict, int(len(list_data_dict) * sampling_rate))
                print(f"sampling {len(list_data_dict)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")

            self.list_data_dict.extend(list_data_dict)

        self.num_history = data_args.num_history
        self.idx2actions = {0: 'STOP', 1: "↑", 2: "←", 3: "→", 5: "↓"}
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]
        self.data_args = data_args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.list_data_dict)

    def process_image_unified(self, image):
        processor = copy.deepcopy(self.data_args.image_processor)
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def preprocess_depth_image_v2(
        self, depth_image, do_depth_scale=True, depth_scale=1000, target_height=None, target_width=None
    ):
        if target_height is None:
            target_height = self.image_processor.crop_size['height']
            target_width = self.image_processor.crop_size['width']

        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)

        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
        img[img > 5.0] = 5.0
        return img, (target_width, target_height)

    def __getitem__(self, i):
        (
            ep_id,
            data_path,
            video,
            height,
            pitch_1,
            pitch_2,
            instruction,
            (start_frame_id, end_frame_id),
            action,
            pose,
        ) = self.list_data_dict[i]
        if start_frame_id != 0:
            history_id = np.unique(np.linspace(0, start_frame_id - 1, self.num_history, dtype=np.int32)).tolist()
        else:
            history_id = []

        images = []
        grid_thws = []
        traj_images = []
        traj_depths = []  # optional

        for id in range(0, end_frame_id):
            image_file = os.path.join(
                video, f"observation.images.rgb.{height}cm_{pitch_1}deg", f"episode_{ep_id:06d}_{id}.jpg"
            )
            image = Image.open(image_file).convert('RGB')
            lookdown_image = Image.open(image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg')).convert('RGB')

            depth_image = Image.open(
                image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg').replace('rgb', 'depth').replace('.jpg', '.png')
            )

            depth_image, resize_shape = self.preprocess_depth_image_v2(
                depth_image, do_depth_scale=True, depth_scale=1000, target_height=224, target_width=224
            )
            depth_image = torch.as_tensor(np.ascontiguousarray(depth_image)).float()  # [H, W]
            if id in history_id or id == start_frame_id:
                if self.data_args.transform_train is not None:
                    image = self.data_args.transform_train(image)
                image, grid_thw = self.process_image_unified(image)
                images.append(image)
                grid_thws.append(grid_thw)
                if id == start_frame_id and pose is not None:
                    image, grid_thw = self.process_image_unified(lookdown_image)
                    images.append(image)
                    grid_thws.append(grid_thw)
                    traj_images.append(lookdown_image)
                    traj_depths.append(depth_image)
            elif id > start_frame_id:
                traj_images.append(lookdown_image)
                traj_depths.append(depth_image)

        history_imgs = "<image>\n" * len(history_id)

        if start_frame_id != 0:
            chat_sources = [
                [
                    {
                        'from': 'human',
                        'value': f"You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task. These are your historical observations: <history>. {random.choice(self.conjunctions)}<image>.",
                    }
                ]
            ]
            chat_sources[0][0]['value'] = (
                chat_sources[0][0]['value'].replace('<instruction>', instruction).replace('<history>', history_imgs)
            )
        else:
            chat_sources = [
                [
                    {
                        'from': 'human',
                        'value': f"You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task. {random.choice(self.conjunctions)}<image>.",
                    }
                ]
            ]
            chat_sources[0][0]['value'] = chat_sources[0][0]['value'].replace('<instruction>', instruction)

        if pose is not None:
            chat_sources[0].extend(
                [
                    {'from': 'gpt', 'value': self.idx2actions[5]},
                    {'from': 'human', 'value': f'{random.choice(self.conjunctions)}<image>.'},
                    {'from': 'gpt', 'value': f'{action[0]} {action[1]}'},
                ]
            )
        elif action == 0:
            chat_sources[0].extend([{'from': 'gpt', 'value': self.idx2actions[action]}])
        else:
            turn_action_text = ''.join([self.idx2actions[idx] for idx in action])
            chat_sources[0].extend([{'from': 'gpt', 'value': turn_action_text}])

        grid_thw_merged = copy.deepcopy(grid_thws)

        if not isinstance(grid_thws, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thws = [grid_thws]

        grid_thw_merged = [
            merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
        ]

        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
        )

        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thws, dim=0) if grid_thws else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        data_dict["pixel_values"] = torch.cat(images, dim=0)
        data_dict["image_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in grid_thws], dim=0)

        if self.pixel_goal_only:
            goal_len = end_frame_id - start_frame_id - 1
            interval = 2
            frame_ids = np.arange(0, goal_len, interval)
            max_len = 12
            traj_images = torch.tensor(np.stack([np.asarray(timg.resize((224, 224))) for timg in traj_images])) / 255.0
            if len(frame_ids) > max_len:
                interval = int(np.ceil(goal_len / max_len))
                frame_ids = np.arange(0, goal_len, interval)

            traj_poses_gt = []
            for cid in frame_ids:
                discrete_traj_pose = get_trajectory_relative_to_frame(pose[cid:], camera_deg=pitch_2)
                rel_trajectory, rel_pose_resample = interpolate_and_resample_trajectory(
                    discrete_traj_pose, self.predict_step_num
                )
                rel_pose_resample = clip_or_pad(rel_pose_resample, self.predict_step_num)
                traj_poses_gt.append(torch.tensor(rel_pose_resample))

            data_dict["traj_images"] = traj_images[:goal_len][::interval]
            data_dict["traj_depths"] = torch.stack(traj_depths[:goal_len][::interval])
            data_dict["traj_poses"] = torch.stack(traj_poses_gt)
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def process_input_with_traj_tokens(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        max_seq_len: int = None,
        traj_token_length: int = 4,  # TODO hard-code
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        if max_seq_len is None:
            max_seq_len = self.tokenizer.model_max_length - traj_token_length

        batch_size = len(input_ids)
        multi_input_ids = [None] * batch_size
        multi_labels = [None] * batch_size
        t_s_pos = [0] * batch_size

        traj_token_template = torch.full(
            (traj_token_length,), TRAJ_TOKEN_INDEX, dtype=input_ids[0].dtype, device=input_ids[0].device
        )

        for i in range(batch_size):
            truncated_input = input_ids[i][:max_seq_len]
            truncated_label = labels[i][:max_seq_len]

            t_s_pos[i] = len(truncated_input)

            multi_input_ids[i] = torch.cat([truncated_input, traj_token_template])
            multi_labels[i] = torch.cat([truncated_label, traj_token_template.clone()])

        return multi_input_ids, multi_labels, t_s_pos

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]

        if "traj_images" in instances[0]:
            input_ids, labels, t_s_pos = self.process_input_with_traj_tokens(input_ids, labels)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(
                f"Warning input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}"
            )

        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        if "traj_images" in instances[0]:
            traj_images, traj_depths, traj_poses = tuple(
                [instance[key] for instance in instances] for key in ("traj_images", "traj_depths", "traj_poses")
            )
            video_frame_num = []
            max_len = max(img.shape[0] for img in traj_images)
            traj_image_batch = []
            traj_depth_batch = []
            traj_pose_batch = []
            # import pdb; pdb.set_trace()
            for idx in range(len(traj_images)):
                t_img = traj_images[idx]
                t_depth = traj_depths[idx]
                t_pose = traj_poses[idx]
                n_frames = t_img.shape[0]
                video_frame_num.append(n_frames)
                if n_frames < max_len:
                    pad_len = max_len - n_frames
                    last_img = t_img[-1]
                    last_depth = t_depth[-1]
                    last_pose = t_pose[-1]
                    padding_img = last_img.unsqueeze(0).repeat(pad_len, 1, 1, 1)
                    padding_depth = last_depth.unsqueeze(0).repeat(pad_len, 1, 1)
                    padding_pose = last_pose.unsqueeze(0).repeat(pad_len, 1, 1)
                    padded_img = torch.cat([t_img, padding_img], dim=0)
                    padded_depth = torch.cat([t_depth, padding_depth], dim=0)
                    padded_pose = torch.cat([t_pose, padding_pose], dim=0)
                else:
                    padded_img = t_img
                    padded_depth = t_depth
                    padded_pose = t_pose
                traj_image_batch.append(padded_img)
                traj_depth_batch.append(padded_depth)
                traj_pose_batch.append(padded_pose)
            batch['position_ids'] = None
            batch['t_s_pos'] = t_s_pos
            batch['traj_images'] = torch.stack(traj_image_batch)
            batch['traj_depths'] = torch.stack(traj_depth_batch)
            batch['traj_poses'] = torch.stack(traj_pose_batch)
            batch['video_frame_num'] = torch.tensor(video_frame_num)

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(*(instance["attention_mask"] for instance in instances if "attention_mask" in instance))
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


class CombinedDataset(Dataset):
    """
    Combine multiple datasets into a single dataset interface.

    This class is used to merge different datasets for joint training.
    It concatenates samples from all provided datasets and optionally shuffles
    the global index mapping (without changing the underlying datasets).
    """

    def __init__(self, datasets, shuffle=False):
        super(CombinedDataset, self).__init__()
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_length = self.cum_lengths[-1]
        self.shuffle_enabled = shuffle
        self.indices = np.arange(self.total_length)
        if self.shuffle_enabled:
            self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def _map_index(self, idx):
        return self.indices[idx]

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, i):
        real_idx = self._map_index(i)
        for idx, cum_len in enumerate(self.cum_lengths):
            if real_idx < cum_len:
                return self.datasets[idx][real_idx - cum_len + self.lengths[idx]]
        raise ValueError(f"Index {real_idx} out of bound")


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_datasets = []
    if data_args.iign_dataset_use:
        train_datasets.append(VLLNDataset(tokenizer=tokenizer, data_args=data_args))
    if data_args.vln_dataset_use:
        train_datasets.append(NavPixelGoalDataset(tokenizer=tokenizer, data_args=data_args))
    train_dataset = CombinedDataset(train_datasets, shuffle=False)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


if __name__ == "__main__":
    pass
