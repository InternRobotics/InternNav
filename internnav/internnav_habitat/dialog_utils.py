import inspect
import os
from typing import Optional

import cv2
import habitat_sim
import numpy as np
import quaternion
from habitat_baselines.config.default import get_config as get_habitat_config
from npc.utils.get_description import (
    get_path_description,
    get_path_description_without_additional_info,
)
from omegaconf import DictConfig, OmegaConf, open_dict

DEFAULT_IMAGE_TOKEN = "<image>"


def get_config(
    habitat_config_path: str,
    baseline_config_path: str,
    opts: Optional[list] = None,
    configs_dir: str = os.path.dirname(inspect.getabsfile(inspect.currentframe())),
) -> DictConfig:
    """
    Returns habitat_baselines config object composed of configs from yaml file (config_path) and overrides.

    :param config_path: path to the yaml config file.
    :param overrides: list of config overrides. For example, :py:`overrides=["habitat_baselines.trainer_name=ddppo"]`.
    :param configs_dir: path to the config files root directory (defaults to :ref:`_BASELINES_CFG_DIR`).
    :return: composed config object.
    """
    habitat_config = get_habitat_config(habitat_config_path, overrides=opts, configs_dir=configs_dir)
    baseline_config = OmegaConf.load(baseline_config_path)

    with open_dict(habitat_config):
        config = OmegaConf.merge(habitat_config, baseline_config)

    return config


def calculate_path_length(path):
    accumulated_length = [0]
    for i, p in enumerate(path[1:]):
        accumulated_length.append(accumulated_length[i] + np.linalg.norm(np.array(p) - np.array(path[i])))
    return accumulated_length


def get_shortest_path(env, start_position, target_position):
    """
    在habitat环境中找到从当前位置到目标位置的最短路径

    参数:
        env: habitat环境实例
        start_position: 起点位置坐标, numpy数组 [x, y, z]
        target_position: 目标位置坐标, numpy数组 [x, y, z]

    返回:
        path: 路径点列表
        success: 是否找到有效路径
    """
    # 创建路径规划器
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start_position
    shortest_path.requested_end = target_position

    # 计算最短路径
    success = env.sim.pathfinder.find_path(shortest_path)
    return shortest_path.points, success


def get_navigable_path(env, start_position, target_positions: list, object_info: dict):
    start_position = [float(i) for i in start_position]
    target_positions = sorted(
        target_positions,
        key=lambda x: np.linalg.norm(np.array(x['agent_state']['position']) - np.array(object_info['position'])),
    )
    success = False
    while not success and len(target_positions) > 0:
        target_position = target_positions.pop(0)
        shortest_path, success = get_shortest_path(env, start_position, target_position['agent_state']['position'])
    if success:
        return shortest_path, True
    else:
        return [], False


def get_path_description_(env, object_dict, region_dict):
    goal_path, success = get_navigable_path(
        env,
        env.sim.get_agent_state().position,
        [{'agent_state': {'position': vp.agent_state.position}} for vp in env.current_episode.goals[0].view_points],
        {'position': env.current_episode.goals[0].position},
    )
    if not success or len(np.unique(goal_path, axis=0)) == 1:
        print('no shortest path')
        return None, 0
    path_length = calculate_path_length(goal_path)
    pl = path_length[-1]
    goal_index = max([i for i, c in enumerate(path_length) if c < 4])
    # goal_index = len(goal_path)-1
    if goal_index == 0:
        goal_index = len(goal_path) - 1
    questioned_path = goal_path[: goal_index + 1]
    current_yaw = 2 * np.arctan2(env.sim.get_agent_state().rotation.y, env.sim.get_agent_state().rotation.w)
    _, idx = np.unique(questioned_path, axis=0, return_index=True)
    idx_sorted = np.sort(idx)
    questioned_path = list(np.array(questioned_path)[idx_sorted])
    try:
        path_description, _ = get_path_description(
            quaternion.from_euler_angles([0, current_yaw, 0]),
            questioned_path,
            object_dict,
            region_dict,
            return_finish=False,
            height_list=[env.sim.get_agent_state().position[1]] * len(questioned_path),
        )
    except Exception as e:
        print(e)
        path_description, _ = get_path_description_without_additional_info(
            quaternion.from_euler_angles([0, current_yaw, 0]),
            questioned_path,
            height_list=[env.sim.get_agent_state().position[1]] * len(questioned_path),
        )
    return path_description, pl


def unify_to_first(
    vis_frames,
    method: str = "resize",  # "resize" 或 "letterbox"
    pad_color=(0, 0, 0),  # letterbox 的填充色 (B,G,R)
    assume_rgb: bool = True,  # 如果后续用 OpenCV 写视频，通常 True 表示当前是 RGB，需要转 BGR
):
    assert len(vis_frames) > 0, "vis_frames 为空"
    h0, w0 = vis_frames[0].shape[:2]
    out = []

    for i, f in enumerate(vis_frames):
        f = np.asarray(f)

        # 保障三通道
        if f.ndim == 2:  # 灰度 -> 3通道
            f = np.stack([f] * 3, axis=2)
        if f.shape[2] > 3:
            f = f[:, :, :3]  # 多通道时只取前三个

        # dtype 归一：转 uint8
        if f.dtype != np.uint8:
            # 若是 [0,1] 浮点，×255；若已是 0-255 浮点，直接裁剪
            fmax = float(np.nanmax(f)) if f.size else 1.0
            f = (f * 255.0) if fmax <= 1.5 else np.clip(f, 0, 255)
            f = f.astype(np.uint8)

        h, w = f.shape[:2]
        if (h, w) == (h0, w0):
            out.append(np.ascontiguousarray(f))
            continue

        if method == "letterbox":
            # 等比缩放 + 居中贴到画布
            scale = min(w0 / w, h0 / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            canvas = np.full((h0, w0, 3), pad_color, dtype=np.uint8)
            top, left = (h0 - nh) // 2, (w0 - nw) // 2
            canvas[top : top + nh, left : left + nw] = resized
            f_out = canvas
        else:
            # 直接拉伸到目标大小
            f_out = cv2.resize(f, (w0, h0), interpolation=cv2.INTER_AREA if (h * w) > (h0 * w0) else cv2.INTER_LINEAR)

        out.append(np.ascontiguousarray(f_out))

    return out
