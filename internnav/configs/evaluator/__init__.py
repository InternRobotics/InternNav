from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from internnav.configs.agent import AgentCfg


class EnvCfg(BaseModel):
    env_type: str
    env_settings: Dict[str, Any]


class SensorCfg(BaseModel):
    sensor_settings: Dict[str, Any] = None


class ControllerCfg(BaseModel):
    controller_settings: Dict[str, Any] = None


class RobotCfg(BaseModel):
    robot_settings: Dict[str, Any] = None
    sensors: List[SensorCfg] = None
    controllers: List[ControllerCfg] = None


class SceneCfg(BaseModel):
    scene_type: Optional[str] = None
    scene_asset_path: str = None
    scene_scale: Optional[Any] = None
    scene_settings: Optional[Dict[str, Any]] = None
    scene_data_dir: Optional[str] = None


class MetricCfg(BaseModel):
    save_dir: str = None
    metric_setting: Dict[str, Any] = None


class TaskCfg(BaseModel):
    task_name: Optional[str] = None
    task_settings: Dict[str, Any]
    scene: SceneCfg
    robot_name: Optional[str] = None
    robot: Optional[RobotCfg] = None
    robot_flash: Optional[bool] = None
    robot_usd_path: Optional[str] = None
    camera_resolution: Optional[List[int]] = None
    metric: Optional[MetricCfg] = None
    camera_prim_path: Optional[str] = None


class EvalDatasetCfg(BaseModel):
    dataset_type: Optional[str] = None
    dataset_settings: Dict[str, Any]


class EvalCfg(BaseModel):
    eval_type: Optional[str] = None
    eval_settings: Optional[Dict[str, Any]] = {}
    agent: Optional[AgentCfg] = None
    env: EnvCfg = None
    task: TaskCfg = None
    dataset: EvalDatasetCfg = None


__all__ = [
    'AgentCfg',
    'EnvCfg',
    'SensorCfg',
    'ControllerCfg',
    'RobotCfg',
    'SceneCfg',
    'MetricCfg',
    'EvalDatasetCfg',
    'EvalCfg',
]
