import os
import sys

import numpy as np
from internutopia.core.config.distribution import RayDistributionCfg
from internutopia_extension.configs.controllers import H1MoveBySpeedControllerCfg
from internutopia_extension.configs.sensors import RepCameraCfg
from pydantic import BaseModel

from internnav.configs.evaluator import (
    ControllerCfg,
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    MetricCfg,
    RobotCfg,
    SceneCfg,
    SensorCfg,
    TaskCfg,
)
from internnav.configs.model import cma_cfg, internvla_n1_cfg, rdp_cfg, seq2seq_cfg
from internnav.env.utils.internutopia_extension.configs.controllers import (
    DiscreteControllerCfg,
    StandStillControllerCfg,
    VlnMoveByFlashControllerCfg,
)
from internnav.env.utils.internutopia_extension.configs.metrics.vln_pe_metrics import (
    VLNPEMetricCfg,
)
from internnav.env.utils.internutopia_extension.configs.sensors.vln_camera import (
    VLNCameraCfg,
)

h1_vln_move_by_speed_cfg = H1MoveBySpeedControllerCfg(
    name='vln_move_by_speed',
    type='VlnMoveBySpeedController',
    policy_weights_path='data/Embodiments/vln-pe/h1/policy/move_by_speed/h1_loco_jit_policy.pt',
    joint_names=[
        'left_hip_yaw_joint',
        'right_hip_yaw_joint',
        'torso_joint',
        'left_hip_roll_joint',
        'right_hip_roll_joint',
        'left_shoulder_pitch_joint',
        'right_shoulder_pitch_joint',
        'left_hip_pitch_joint',
        'right_hip_pitch_joint',
        'left_shoulder_roll_joint',
        'right_shoulder_roll_joint',
        'left_knee_joint',
        'right_knee_joint',
        'left_shoulder_yaw_joint',
        'right_shoulder_yaw_joint',
        'left_ankle_joint',
        'right_ankle_joint',
        'left_elbow_joint',
        'right_elbow_joint',
    ],
)
vln_move_by_flash_cfg = VlnMoveByFlashControllerCfg(name='move_by_flash')

cfg = EvalCfg(
    agent=None,
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'physics_dt': 1 / 200,
            'rendering_dt': 1 / 200,
            'rendering_interval': 5,
            'use_fabric': True,
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name=None,
        task_settings={
            'max_step': 25000,
            'check_fall_and_stuck': True,
            'robot_ankle_height': 0.0758,
            'warm_up_step': 100,
            'offset_size': 100,
        },
        scene=SceneCfg(scene_asset_path='', scene_scale=None, scene_settings={}),
        metric=MetricCfg(
            save_dir='',
            metric_setting={
                'type': 'VLNPEMetric',
                'name': 'VLNPEMetric',
                'metric_config': VLNPEMetricCfg(success_distance=3.0, shortest_to_goal_distance=999),
            },
        ),
    ),
    dataset=EvalDatasetCfg(
        dataset_type='mp3d',
        dataset_settings={
            'filter_same_trajectory': False,
            'run_type': 'eval',
            'retry_list': [],
        },
    ),
    eval_type='vln_multi',
    eval_settings={'save_to_json': True, 'vis_output': True},
)


def validate_eval_config(eval_cfg: BaseModel):
    """validate the evaluation config"""

    def check_nested_none(obj, path=''):
        none_paths = []

        # 不检查dict内部
        if isinstance(obj, BaseModel):
            obj = obj.model_dump()
            for key, value in obj.items():
                current_path = f'{path}.{key}' if path else key
                if value is None:
                    none_paths.append(current_path)
                elif isinstance(value, BaseModel):
                    none_paths.extend(check_nested_none(value, current_path))
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        item_path = f'{current_path}[{i}]'
                        if item is None:
                            none_paths.append(item_path)
                        elif isinstance(item, BaseModel):
                            none_paths.extend(check_nested_none(item, item_path))

        return none_paths

    none_fields = check_nested_none(eval_cfg)
    if none_fields:
        error_msg = 'Evaluation config validation failed!\nNone values found in:\n'
        for field in none_fields:
            error_msg += f' - {field}\n'
        raise ValueError(error_msg)

    print('Evaluation config validation passed!')
    return True


def merge_models(base_model, update_model):
    """
    smart merge two models:
    - dict type uses dict.update()
    - other types directly overwrite
    """
    base_dict = base_model.model_dump()
    update_dict = update_model.model_dump(exclude_none=True)

    def update(base_dict, update_dict):
        """
        deep update, handle dict and list
        """
        for key, value in update_dict.items():
            if key in base_dict:
                if isinstance(base_dict[key], dict) and isinstance(value, dict):
                    # recursively handle dict
                    update(base_dict[key], value)
                elif isinstance(base_dict[key], list) and isinstance(value, list):
                    # list directly replace (or choose to merge)
                    base_dict[key] = value
                    # if you want to merge lists: base_dict[key].extend(value)
                else:
                    # other types directly overwrite
                    base_dict[key] = value
            else:
                base_dict[key] = value

    update(base_dict, update_dict)
    base_dict = type(base_model)(**base_dict)
    try:
        validate_eval_config(base_dict)
    except ValueError as e:
        print(f'❌ Configuration Error: \n{e}')
        sys.exit(1)
    return base_dict


def get_config(evaluator_cfg: EvalCfg):
    # switch robots
    if evaluator_cfg.task.robot_name == 'h1':
        move_by_speed_cfg = h1_vln_move_by_speed_cfg.model_dump()
        robot_type = 'VLNH1Robot'
        robot_usd_path = evaluator_cfg.task.robot_usd_path
        move_by_speed_cfg["policy_weights_path"] = (
            os.path.dirname(robot_usd_path) + '/policy/move_by_speed/h1_loco_jit_policy.pt'
        )
        camera_resolution = evaluator_cfg.task.camera_resolution
        robot_offset = np.array([0.0, 0.0, 1.05])
        camera_prim_path = evaluator_cfg.task.camera_prim_path
        fall_height_threshold = 0.5
    else:
        raise RuntimeError(f"unknown robot_name: {evaluator_cfg.task.robot_name}")

    stand_still_cfg_ = StandStillControllerCfg(
        name='stand_still',
        type='StandStillController',
        sub_controllers=[move_by_speed_cfg],
    ).model_dump()

    discrete_controller_cfg_ = DiscreteControllerCfg(
        name='move_by_discrete',
        steps_per_action=50,
        forward_distance=0.25,
        rotation_angle=15.0,
        physics_frequency=200,
        sub_controllers=[move_by_speed_cfg],
    ).model_dump()

    robot = RobotCfg(
        robot_settings={
            'type': robot_type,
            'usd_path': robot_usd_path,
            'position': (0.0, 0.0, 1.05),
        },
        sensors=[
            SensorCfg(
                sensor_settings=VLNCameraCfg(
                    name='pano_camera_0',
                    prim_path=camera_prim_path,
                    enable=True,
                    resolution=camera_resolution,
                ).model_dump(),
            ),
        ],
        controllers=[
            ControllerCfg(
                controller_settings=move_by_speed_cfg,
            ),
            ControllerCfg(
                controller_settings=stand_still_cfg_,
            ),
            ControllerCfg(
                controller_settings=discrete_controller_cfg_,
            ),
        ],
    )  # may need to be modified to two files

    # add the flash controller in, by flash flag.
    if evaluator_cfg.task.robot_flash:
        robot.controllers.append(ControllerCfg(controller_settings=vln_move_by_flash_cfg.model_dump()))

    if evaluator_cfg.task.robot_flash or evaluator_cfg.eval_settings.get('vis_output', True):
        topdown_camera = SensorCfg(
            sensor_type='VLNCamera',
            sensor_name='topdown_camera_500',
            sensor_settings=VLNCameraCfg(
                name='topdown_camera_500',
                prim_path='topdown_camera_500',
                enable=True,
                resolution=[500, 500],
            ).model_dump(),
        )
        robot.sensors.append(topdown_camera)

    if evaluator_cfg.task.robot_name == 'h1':
        tp_pointcloud = SensorCfg(
            sensor_settings=RepCameraCfg(
                name='tp_pointcloud',
                prim_path='logo_link/Camera_pointcloud',
                enable=True,
                rgba=False,
                pointcloud=True,
                resolution=[64, 64],
            ).model_dump(),
        )
        robot.sensors.append(tp_pointcloud)
    evaluator_cfg.task.robot = robot
    # task_settings update
    evaluator_cfg.task.task_settings.update({'fall_height_threshold': fall_height_threshold})
    # dataset_settings update
    evaluator_cfg.dataset.dataset_settings.update(
        {'robot_offset': robot_offset, 'task_name': evaluator_cfg.task.task_name}
    )

    # switch scene
    if evaluator_cfg.task.scene.scene_type == 'mp3d':
        evaluator_cfg.task.scene = SceneCfg(
            scene_type='mp3d',
            scene_asset_path='',
            scene_scale=(1, 1, 1),
            scene_settings={},
            scene_data_dir=evaluator_cfg.task.scene.scene_data_dir,
        )
    elif evaluator_cfg.task.scene.scene_type == 'grscene':
        evaluator_cfg.task.scene = SceneCfg(
            scene_type='grscene',
            scene_asset_path='',
            scene_scale=(0.01, 0.01, 0.01),
            scene_settings={},
            scene_data_dir=evaluator_cfg.task.scene.scene_data_dir,
        )
    elif evaluator_cfg.task.scene.scene_type == 'kujiale':
        evaluator_cfg.task.scene = SceneCfg(
            scene_type='kujiale',
            scene_asset_path='',
            scene_scale=(0.01, 0.01, 0.01),
            scene_settings={},
            scene_data_dir=evaluator_cfg.task.scene.scene_data_dir,
        )

    # switch model
    if evaluator_cfg.agent.model_name == 'cma':
        model_settings = cma_cfg.model_dump()
    elif evaluator_cfg.agent.model_name == 'rdp':
        model_settings = rdp_cfg.model_dump()
    elif evaluator_cfg.agent.model_name == 'seq2seq':
        model_settings = seq2seq_cfg.model_dump()
    elif evaluator_cfg.agent.model_name == 'internvla_n1':
        model_settings = internvla_n1_cfg.model_dump()

    model_settings.update(evaluator_cfg.agent.model_settings)
    evaluator_cfg.agent.model_settings = model_settings

    final_cfg = merge_models(cfg, evaluator_cfg)

    if evaluator_cfg.task.task_settings['use_distributed']:
        distribution_config = RayDistributionCfg(
            proc_num=evaluator_cfg.task.task_settings['proc_num'],
            head_address=None,
        ).model_dump()
        final_cfg.env.env_settings.update({'distribution_config': distribution_config})

    return final_cfg
