import copy
import json
import os
import random
import re
from bisect import bisect_left
from typing import Dict, List, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from .rope2d import get_rope_index_2, get_rope_index_25

# Define placeholders for dataset paths
IIGN_split1 = {
    "data_path": "projects/VL-LN-Bench/traj_data/mp3d_split1",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

IIGN_split2 = {
    "data_path": "projects/VL-LN-Bench/traj_data/mp3d_split2",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

IIGN_split3 = {
    "data_path": "projects/VL-LN-Bench/traj_data/mp3d_split3",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

data_dict = {
    "iign_split1": IIGN_split1,
    "iign_split2": IIGN_split2,
    "iign_split3": IIGN_split3,
}

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
TRAJ_TOKEN_INDEX = 151667
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
_ORACLE_BLOCK = re.compile(r'<\|oracle\|>.*?<\|dialog_end\|>', re.DOTALL)

local_rank = None


class VLLNDataset(Dataset):
    """
    Dataset for 'Vision-Language'-'Language-Navigation' (VL-LN) / IIGN-style training.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to encode
            the chat template and produce `input_ids` / `labels`.
        data_args: A config-like object that must provide at least:
            - iign_dataset_use (str): comma-separated dataset names, optionally
              with sampling rate suffix like `iign_split1%50`.
            - model_type (str): decides which rope-index function to use.
            - sample_step (int): stride for sampling start frames.
            - pixel_goal_only (bool): whether to keep only pixel-goal samples.
            - num_future_steps (int): horizon for turn-action extraction.
            - max_dialog_turns (int): max number of answers the agent can get from oracle.
            - num_history (int): number of history frames in prompt.

    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(VLLNDataset, self).__init__()
        dataset = data_args.iign_dataset_use.split(",")
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
        self.pixel_goal_only = data_args.pixel_goal_only
        self.num_future_steps = data_args.num_future_steps
        self.max_dialog_turns = data_args.max_dialog_turns

        self.list_data_dict = []

        for data in dataset_list:
            sampling_rate = data.get("sampling_rate", 1.0)
            height = data.get("height", None)
            pitch_1 = data.get("pitch_1", None)
            pitch_2 = data.get("pitch_2", None)

            data_path = data['data_path']

            annotations = get_annotations_from_lerobot_data(data_path, pitch_1, pitch_2, height)

            pixel_goal_list = []
            turn_list = []
            stop_list = []
            list_data_dict = []
            dialog_list = []

            for ep_id, item in enumerate(annotations['episodes']):
                ep_id = item['id']
                instruction = item['instructions']
                video = item['video']
                dialogs = item['dialogs']
                dialogs, dia_idx = sort_dialogs_by_true_idx(dialogs)
                actions = item['actions'][1:] + [0]
                pixel_goals = item['pixel_goals']
                poses = item[f'poses_{height}cm_{pitch_1}deg']

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
                    history_dialogs = get_history_dialogs(start_frame_id, dialogs, dia_idx)
                    if pixel_goal[0] == -1:
                        if action_flag == 1:
                            continue
                        else:
                            turn_actions = get_turn_actions(actions, start_frame_id, self.num_future_steps)
                            turn_list.append(
                                (
                                    ep_id,
                                    data_path,
                                    video,
                                    height,
                                    pitch_1,
                                    pitch_2,
                                    instruction,
                                    start_frame_id,
                                    turn_actions,
                                    None,
                                    history_dialogs,
                                    None,
                                )
                            )
                    else:
                        goal_len = pixel_goal[0]
                        action = pixel_goal[1]
                        pose = poses[start_frame_id : start_frame_id + goal_len]
                        pixel_goal_list.append(
                            (
                                ep_id,
                                data_path,
                                video,
                                height,
                                pitch_1,
                                pitch_2,
                                instruction,
                                start_frame_id,
                                action,
                                pose,
                                history_dialogs,
                                None,
                            )
                        )
                stop_frame = actions_len - 1
                stop_history = get_history_dialogs(stop_frame, dialogs, dia_idx)
                stop_list.append(
                    (
                        ep_id,
                        data_path,
                        video,
                        height,
                        pitch_1,
                        pitch_2,
                        instruction,
                        actions_len - 1,
                        0,
                        None,
                        stop_history,
                        None,
                    )
                )
                for n in range(len(dia_idx)):
                    start_frame_id = dia_idx[n]
                    action = actions[start_frame_id : start_frame_id + self.num_future_steps]
                    history_dialogs = get_history_dialogs(start_frame_id, dialogs, dia_idx)
                    current_dialog = [sentence for sentence in dialogs if sentence['true_idx'] == start_frame_id]
                    if action[0] == 1:
                        pixel_goal = pixel_goals[start_frame_id]
                        if pixel_goal[0] != -1:
                            goal_len = pixel_goal[0]
                            pose = poses[start_frame_id : start_frame_id + goal_len]
                            dialog_list.append(
                                (
                                    ep_id,
                                    data_path,
                                    video,
                                    height,
                                    pitch_1,
                                    pitch_2,
                                    instruction,
                                    start_frame_id,
                                    pixel_goal[1],
                                    pose,
                                    history_dialogs,
                                    current_dialog,
                                )
                            )
                        else:
                            continue
                    elif action[0] == 0:
                        dialog_list.append(
                            (
                                ep_id,
                                data_path,
                                video,
                                height,
                                pitch_1,
                                pitch_2,
                                instruction,
                                start_frame_id,
                                0,
                                None,
                                history_dialogs,
                                current_dialog,
                            )
                        )
                    else:
                        turn_actions = get_turn_actions(actions, start_frame_id, self.num_future_steps)
                        dialog_list.append(
                            (
                                ep_id,
                                data_path,
                                video,
                                height,
                                pitch_1,
                                pitch_2,
                                instruction,
                                start_frame_id,
                                turn_actions,
                                None,
                                history_dialogs,
                                current_dialog,
                            )
                        )

            list_data_dict = pixel_goal_list
            rank0_print(len(turn_list), len(pixel_goal_list), len(stop_list), len(dialog_list))
            if not self.pixel_goal_only:
                list_data_dict += turn_list
                list_data_dict += stop_list * 10
                list_data_dict += dialog_list * 10
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

    def __getitem__(self, i):
        (
            ep_id,
            data_path,
            video,
            height,
            pitch_1,
            pitch_2,
            instruction,
            start_frame_id,
            action,
            pose,
            history_dialogs,
            current_dialog,
        ) = self.list_data_dict[i]
        dialogs_id = np.array([dialog['true_idx'] for dialog in history_dialogs])[::2]
        if start_frame_id != 0:
            history_id = np.unique(
                np.concatenate([np.linspace(0, start_frame_id - 1, self.num_history, dtype=np.int32), dialogs_id])
            ).tolist()
        else:
            history_id = []

        images = []
        grid_thws = []

        for id in range(0, start_frame_id + 1):
            image_file = os.path.join(
                video, f"observation.images.rgb.{height}cm_{pitch_1}deg", f"episode_{ep_id:06d}_{id}.jpg"
            )
            if id in history_id or id == start_frame_id:
                image = Image.open(image_file).convert('RGB')
                lookdown_image = Image.open(image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg')).convert('RGB')
                if self.data_args.transform_train is not None:
                    image = self.data_args.transform_train(image)
                image, grid_thw = self.process_image_unified(image)
                images.append(image)
                grid_thws.append(grid_thw)
                if id == start_frame_id and pose is not None:
                    image, grid_thw = self.process_image_unified(lookdown_image)
                    images.append(image)
                    grid_thws.append(grid_thw)

        if history_dialogs:
            history_imgs = build_dialog_history(history_id, dialogs_id, history_dialogs)
        else:
            history_imgs = "<image>\n" * len(history_id)

        if start_frame_id != 0:
            chat_sources = [
                [
                    {
                        'from': 'human',
                        'value': f"You are an autonomous navigation assistant. Your task is to <instruction> There is an oracle can help you to complete the task in current environment, you can either choose to move or talk. If choosing to talk, please say something that can help you better to find the target object. If choosing to move, when you want to output a waypoint you need to TILT DOWN (↓) by 30 degrees then output the next waypoint\'s coordinates in the image. In case the next waypoint is out of view, utilize the turn actions: TURN LEFT (←) or TURN RIGHT (→) by 30 degrees. Please output STOP when you have successfully completed the task. These are your historical observations: <history>. {random.choice(self.conjunctions)}<image>.",
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
                        'value': f"You are an autonomous navigation assistant. Your task is to <instruction> There is an oracle can help you to complete the task in current environment, you can either choose to move or talk. If choosing to talk, please say something that can help you better to find the target object. If choosing to move, when you want to output a waypoint you need to TILT DOWN (↓) by 30 degrees then output the next waypoint\'s coordinates in the image. In case the next waypoint is out of view, utilize the turn actions: TURN LEFT (←) or TURN RIGHT (→) by 30 degrees. Please output STOP when you have successfully completed the task. {random.choice(self.conjunctions)}<image>.",
                    }
                ]
            ]
            chat_sources[0][0]['value'] = chat_sources[0][0]['value'].replace('<instruction>', instruction)

        if current_dialog is not None:
            for turn in range(len(current_dialog) // 2):
                chat_sources[0].extend([{'from': 'gpt', 'value': '<talk>' + current_dialog[2 * turn]['message']}])
                chat_sources[0].extend([{'from': 'human', 'value': current_dialog[2 * turn + 1]['message']}])

        if pose is not None:
            chat_sources[0].extend(
                [
                    {'from': 'gpt', 'value': '<move>' + self.idx2actions[5]},
                    {'from': 'human', 'value': f'{random.choice(self.conjunctions)}<image>.'},
                    {'from': 'gpt', 'value': '<move>' + f'{action[0]} {action[1]}'},
                ]
            )
        elif action == 0:
            chat_sources[0].extend([{'from': 'gpt', 'value': '<move>' + self.idx2actions[action]}])
        else:
            turn_action_text = ''.join([self.idx2actions[idx] for idx in action])
            chat_sources[0].extend([{'from': 'gpt', 'value': '<move>' + turn_action_text}])
        chat_sources = enforce_simple_limit(chat_sources, limit=random.randint(0, self.max_dialog_turns))

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

        return data_dict


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


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    """Tokenize multi-modal chat sources for Qwen2.5-VL style training.

    Args:
        sources (list): Conversation sources. Expected structure is a list of
            conversations, where each conversation is a list of dict messages.
            The dict keys may be either:
            - {"from": "human"/"gpt", "value": "..."}, or
            - {"role": "user"/"assistant", "value": "..."}
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
        grid_thw_image (List[int]): For each "<image>" placeholder, provides the
            number of visual tokens (after merge) to replicate `<|image_pad|>`.
            Here "thw" refers to the visual token grid shape:
            - t: temporal length in the visual grid
            - h: grid height (number of patch rows)
            - w: grid width (number of patch columns)
        grid_thw_video (List[int]): Same as above for "<video>".
    Returns:
        Dict[str, torch.Tensor]:
            - input_ids: LongTensor of shape [B, L]
            - labels: LongTensor of shape [B, L]
    """
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
        except Exception:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except Exception:
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
                            + "<|image_pad|>" * grid_thw_image[visual_replicate_index_image]
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


def get_annotations_from_lerobot_data(data_path, pitch_1, pitch_2, height):
    """Load LeRobot-format dataset and convert it into unified annotations.

    It scans scene directories under `data_path`, and for each scene:
    - Reads `meta/episodes.jsonl` to get episode metadata, instructions, dialogs.
    - Reads `data/chunk-xxx/episode_XXXXXX.parquet` to get actions, poses, goals.
    - Constructs a unified dict `annotations` with an `episodes` list.

    The output `annotations["episodes"]` items include:
    - id, video, instructions, actions, length
    - poses for both horizon and look-down settings
    - pixel_goals in `[relative_goal_frame_id, goal]` format
    - dialogs (list)

    Args:
        data_path (str): Root directory containing multiple scene folders.
        pitch_1 (int): Horizon camera pitch (e.g., 0).
        pitch_2 (int): Look-down camera pitch (e.g., 30).
        height (int): Camera height in centimeters (e.g., 125).

    Returns:
        dict: A dict with keys:
            - axis_align_matrix (List[List[float]]): identity by default
            - episodes (List[dict]): unified episode entries
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pyarrow.parquet as pq

    setting = f'{height}cm_{pitch_2}deg'
    setting_horizon = setting.replace(str(pitch_2), str(pitch_1))
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
            ep_instructions = ep["tasks"][0].split(";")
            ep_len = ep["length"]
            ep_dialogs = ep["dialogs"]
            parquet_path = os.path.join(
                scene_path, "data", f"chunk-{ep_id // 1000:03d}", f"episode_{ep_id:06d}.parquet"
            )

            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            ep_actions = df["action"].tolist()
            pose_key = f"pose.{setting}"
            goal_key = f"goal.{setting}"
            relative_goal_frame_id_key = f"relative_goal_frame_id.{setting}"

            ep_poses_horizon = df[f"pose.{setting_horizon}"].apply(lambda x: x.tolist()).tolist()
            if pose_key in df.columns and goal_key in df.columns and relative_goal_frame_id_key in df.columns:
                ep_poses = df[pose_key].apply(lambda x: x.tolist()).tolist()
                ep_pixel_goals = [
                    [df[relative_goal_frame_id_key][idx].tolist(), df[goal_key][idx].tolist()] for idx in range(len(df))
                ]
            else:
                print(f"Warning: Missing data for setting {setting} in episode {ep_id}, filling with defaults.")

            assert len(ep_actions) == ep_len, f"Action length mismatch in episode {ep_id}"

            episode = {
                "id": ep_id,
                "video": f"{data_path}/{scene_id}/videos/chunk-{ep_id // 1000:03d}",
                "instructions": ep_instructions[0],
                "actions": ep_actions,
                "length": ep_len,
                f"poses_{setting}": ep_poses,
                f"poses_{setting_horizon}": ep_poses_horizon,
                "pixel_goals": ep_pixel_goals,
                "dialogs": ep_dialogs,
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


def get_turn_actions(actions, start_frame_id, num_future_steps):
    if not (0 <= start_frame_id < len(actions)):
        return []
    s = actions[start_frame_id : start_frame_id + num_future_steps]
    first = s[0]
    i = next((k for k, x in enumerate(s) if x != first), len(s))
    return s[:i]


def sort_dialogs_by_true_idx(dialogs):
    groups = []
    i, n = 0, len(dialogs)
    while i < n:
        groups.append(dialogs[i : i + 2])
        i += 2

    def group_key(g):
        return max(d.get("true_idx", float("inf")) for d in g)

    keyed = [(g, group_key(g)) for g in groups]
    keyed.sort(key=lambda x: x[1])

    sorted_dialogs = []
    unique_true_idx = []
    seen = set()
    for g, k in keyed:
        sorted_dialogs.extend(g)
        if k not in seen:
            unique_true_idx.append(k)
            seen.add(k)

    return sorted_dialogs, unique_true_idx


def get_history_dialogs(start_frame_id, dialogs, dia_idx):
    i = bisect_left(dia_idx, start_frame_id)
    if i != 0:
        return dialogs[: 2 * i]
    else:
        return []


def build_dialog_history(history_id, dialog_id, dialogs):
    """
    Build a serialized string that interleaves visual placeholders (<image>) with
    dialog blocks (<|dialog_start|>...<|dialog_end|>) aligned to history frames.

    Args:
        history_id (List[int]): History frame ids (sorted/unique).
        dialog_id (Sequence[int]): Frame indices that have dialogs (true_idx).
        dialogs (List[dict]): Dialog messages.

    Returns:
        str: Serialized history string aligned to `history_id`.
    """
    placeholder = [''] * (len(history_id) + 1)
    for n in dialog_id:
        pos = history_id.index(n)
        output = ""
        for dialog in dialogs:
            if dialog['true_idx'] == n:
                output += f"<|{dialog['role']}|>{dialog['message']}"
        placeholder[pos + 1] = "<|dialog_start|>" + output + "<|dialog_end|>"
    placeholder = ('<image>\n').join(placeholder)
    return placeholder


def enforce_simple_limit(
    conv, limit, sorry_msg: str = "Sorry, you have reached the question limit. No further answers are available."
):
    """Limit the number of answer-like parts in a conversation.

    This function truncates answer-like content beyond a given `limit`.
    Extra units beyond `limit` are replaced by a fixed `sorry_msg`.

    Args:
        conv (list): A single conversation packed as `[conv0]`, where `conv0`
            is a list of message dicts using keys `from/value`.
        limit (int): Maximum number of answer-like units to keep.
        sorry_msg (str): Replacement message inserted for truncated content.

    Returns:
        list: The updated conversation in the same format as input, i.e. `[conv0]`.
    """
    conv = [dict(m) for m in conv[0]]
    answer_indices = []
    replaced_indices = []

    first_val = conv[0].get('value', '') if len(conv) >= 1 else ''
    blocks = list(_ORACLE_BLOCK.finditer(first_val))
    for i in range(len(blocks)):
        answer_indices.append(('oracle', (0, i)))

    talk_human_indices: List[int] = []
    for k in range(len(conv) - 1):
        if conv[k].get('from', '') == 'gpt' and conv[k].get('value', '').lstrip().startswith('<talk>'):
            if conv[k + 1].get('from', '') == 'human':
                talk_human_indices.append(k + 1)
                answer_indices.append(('more', k + 1))

    to_replace = {idx for idx, _ in enumerate(answer_indices) if idx >= limit}

    if blocks:
        block_idx = -1

        def _repl(m):
            nonlocal block_idx
            block_idx += 1
            if block_idx in to_replace:
                replaced_indices.append(('oracle', (0, block_idx)))
                return '<|oracle|>' + sorry_msg + '<|dialog_end|>'
            return m.group(0)

        new_first_val = _ORACLE_BLOCK.sub(_repl, first_val)
        if new_first_val != first_val:
            conv[0]['value'] = new_first_val

    for global_idx, (tag, loc) in enumerate(answer_indices):
        if tag == 'more' and global_idx in to_replace:
            human_idx = loc
            if 0 <= human_idx < len(conv):
                conv[human_idx]['value'] = sorry_msg
                replaced_indices.append(('more', human_idx))

    return [conv]
