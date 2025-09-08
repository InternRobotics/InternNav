from typing import Optional

from pydantic import BaseModel


class Loss(BaseModel, extra='allow'):
    alpha: Optional[float]
    dist_scale: Optional[int]


class FilterFailure(BaseModel, extra='allow'):
    use: Optional[bool]
    min_rgb_nums: Optional[int]


class IlCfg(BaseModel, extra='allow'):
    epochs: Optional[int]
    batch_size: Optional[int] = None
    lr: Optional[float]
    num_workers: Optional[int]
    weight_decay: Optional[float]
    warmup_ratio: Optional[float]
    use_iw: Optional[bool] = None
    inflection_weight_coef: Optional[float] = None
    save_interval_steps: Optional[int] = None
    save_filter_frozen_weights: Optional[bool] = None
    load_from_ckpt: Optional[bool]
    ckpt_to_load: Optional[str]
    dataset_r2r_root_dir: Optional[str] = None
    dataset_3dgs_root_dir: Optional[str] = None
    dataset_grutopia10_root_dir: Optional[str] = None
    lmdb_features_dir: Optional[str] = None
    camera_name: Optional[str] = None
    filter_failure: Optional[FilterFailure] = None
    use_discrete_dataset: Optional[bool] = None
    loss: Optional[Loss] = None
    report_to: Optional[str] = None
