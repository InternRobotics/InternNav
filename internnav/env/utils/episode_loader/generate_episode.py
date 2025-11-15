import os

from internnav.configs.evaluator import TaskCfg
from internnav.utils.common_log_util import common_logger as log

from .resumable import ResumablePathKeyEpisodeloader


def load_scene_usd(mp3d_data_dir, scan):
    """Load scene USD based on the scan"""
    from internutopia.core.util import is_in_container

    find_flag = False
    for root, dirs, files in os.walk(os.path.join(mp3d_data_dir, scan)):
        target_file_name = 'fixed_docker.usd' if is_in_container() else 'fixed.usd'
        for file in files:
            if file == target_file_name:
                scene_usd_path = os.path.join(root, file)
                find_flag = True
                break
        if find_flag:
            break
    if not find_flag:
        log.error('Scene USD not found for scan %s', scan)
        return None
    return scene_usd_path


def load_kujiale_scene_usd(kujiale_iros_data_dir, scan):
    """Load scene USD based on the scan"""
    scene_usd_path = os.path.join(kujiale_iros_data_dir, scan, f'{scan}.usda')
    if not os.path.exists(scene_usd_path):
        log.error('Scene USD not found for scan %s', scan)
        return None
    return scene_usd_path


def generate_vln_episode(dataloader: ResumablePathKeyEpisodeloader, task: TaskCfg):
    scene_data_dir = task.scene.scene_data_dir
    scene_asset_path = task.scene.scene_asset_path
    eval_path_key_list = dataloader.resumed_path_key_list
    path_key_data = dataloader.path_key_data
    episodes = []

    # lazy import
    from internutopia.core.config.robot import ControllerCfg
    from internutopia_extension.configs.robots.h1 import H1RobotCfg
    from internutopia_extension.configs.sensors import RepCameraCfg

    from internnav.env.utils.internutopia_extension.configs.metrics import (
        VLNPEMetricCfg,
    )
    from internnav.env.utils.internutopia_extension.configs.tasks import VLNEvalTaskCfg

    robot = H1RobotCfg(
        **task.robot.robot_settings,
        controllers=[ControllerCfg(**cfg.controller_settings) for cfg in task.robot.controllers],
        sensors=[RepCameraCfg(**cfg.sensor_settings) for cfg in task.robot.sensors],
    )

    for path_key in eval_path_key_list:
        data = path_key_data[path_key]
        start_position = data['start_position']
        start_rotation = data['start_rotation']
        data['path_key'] = path_key
        data['name'] = dataloader.task_name

        if task.scene.scene_type == 'kujiale':
            load_scene_func = load_kujiale_scene_usd
            scene_scale = (1, 1, 1)
        else:
            load_scene_func = load_scene_usd
            scene_scale = (1, 1, 1)

        robot_flash = getattr(task, "robot_flash", False)
        one_step_stand_still = getattr(task, "one_step_stand_still", False)
        if task.metric.metric_setting['metric_config'].get('name', None) is None:
            task.metric.metric_setting['metric_config']['name'] = 'default_eval_name'
        episodes.append(
            VLNEvalTaskCfg(
                **task.task_settings,
                robot_flash=robot_flash,
                one_step_stand_still=one_step_stand_still,
                metrics=[VLNPEMetricCfg(**task.metric.metric_setting['metric_config'])],
                scene_asset_path=load_scene_func(scene_data_dir, dataloader.path_key_scan[path_key])
                if scene_asset_path == ''
                else scene_asset_path,
                scene_scale=scene_scale,
                robots=[
                    robot.update(
                        position=(
                            start_position[0],
                            start_position[1],
                            start_position[2],
                        ),
                        orientation=(
                            start_rotation[0],
                            start_rotation[1],
                            start_rotation[2],
                            start_rotation[3],
                        ),
                    )
                ],
                data=data,
            )
        )
    return episodes
