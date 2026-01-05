from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    system1: Optional[str] = field(default='nextdit')
    n_query: int = field(default=4)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)

    vln_dataset_use: str = field(default="")
    iign_dataset_use: str = field(default="")
    sample_step: int = field(default=4)
    num_history: Optional[int] = field(default=8)
    predict_step_num: Optional[int] = field(default=32)
    pixel_goal_only: Optional[bool] = field(default=False)
    data_augmentation: Optional[bool] = field(default=False)
    transform_train: Optional[str] = field(default=None)
    resize_h: Optional[int] = field(default=384)
    resize_w: Optional[int] = field(default=384)
    num_future_steps: Optional[int] = field(default=4)
    max_dialog_turns: Optional[int] = field(default=6)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
