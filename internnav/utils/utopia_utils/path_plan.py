import math
import time

import matplotlib

from internnav.utils.common_log_util import common_logger as log

matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def pixel_to_world(pixel_pose, camera_pose, aperture, width, height):
    cx, cy = (
        camera_pose[0] * 10 / aperture * width,
        -camera_pose[1] * 10 / aperture * height,
    )
    px = height - pixel_pose[0] + cx - height / 2
    py = pixel_pose[1] + cy - width / 2

    world_x = px / 10 / height * aperture
    world_y = -py / 10 / width * aperture

    return [world_x, world_y]


def world_to_pixel(world_pose, camera_pose, aperture, width, height):
    cx, cy = (
        camera_pose[0] * 10 / aperture * width,
        -camera_pose[1] * 10 / aperture * height,
    )

    X, Y = (
        world_pose[0] * 10 / aperture * width,
        -world_pose[1] * 10 / aperture * height,
    )
    pixel_x = width - (X - cx + width / 2)
    pixel_y = Y - cy + height / 2

    return [pixel_x, pixel_y]


def get_real_points(yaw, points, actions, camera_pose, aperture, width, height):
    point_index = 0
    current_real_point = pixel_to_world(points[point_index], camera_pose, aperture, width, height)
    current_yaw = yaw
    real_points = []
    for action in actions:
        if action == 2:  # left
            current_yaw = current_yaw + (math.pi / 12)
            real_points.append(current_yaw)
        elif action == 3:  # right
            current_yaw = current_yaw - (math.pi / 12)
            real_points.append(current_yaw)
        else:  # forward
            point_index = point_index + 1
            current_real_point = pixel_to_world(points[point_index], camera_pose, aperture, width, height)
            real_points.append(current_real_point)
    return real_points


def vis_nav_path(
    start_pixel,
    goal_pixel,
    points,
    occupancy_map,
    img_save_path='path_planning.jpg',
):
    cmap = mcolors.ListedColormap(['white', 'green', 'gray', 'black'])
    bounds = [0, 1, 3, 254, 256]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(10, 10))
    # plt.imshow(occupancy_map, cmap='binary', origin='lower')
    plt.imshow(occupancy_map, cmap=cmap, norm=norm, origin='upper')

    # Plot start and goal points
    plt.plot(start_pixel[1], start_pixel[0], 'ro', markersize=6, label='Start')
    plt.plot(goal_pixel[1], goal_pixel[0], 'bo', markersize=6, label='Goal')

    # Plot the path
    if len(points) > 0:
        path = np.array(points)
        plt.plot(
            path[:, 1],
            path[:, 0],
            'xb-',
            linewidth=1,
            markersize=5,
            label='Path',
        )

    # Customize the plot
    plt.title('Path planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.colorbar(label='Occupancy (0: Free, 1: Occupied)')

    # Save the plot
    plt.savefig(img_save_path, pad_inches=0, bbox_inches='tight', dpi=100)
    log.info(f'Saved path planning visualization to {img_save_path}')
    plt.close()


def plan_and_get_actions_discrete(
    map_info,
    robot_position,
    robot_rotation,
    goal,
    camera_pose,
    aperture,
    width,
    height,
    path_planner,
):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles

    _, _, yaw = quat_to_euler_angles(robot_rotation)
    start_pixel = world_to_pixel(robot_position, camera_pose, aperture, width, height)
    goal_pixel = world_to_pixel(goal, camera_pose, aperture, width, height)
    start_time = time.time()
    points, actions, find_flag, reason = path_planner.planning(
        start_pixel[0],
        start_pixel[1],
        goal_pixel[0],
        goal_pixel[1],
        obs_map=map_info,
        yaw=yaw,
    )
    end_time = time.time()
    log.info(f'path_planning 耗时：{(end_time - start_time)} s')
    if not find_flag:
        return [], [], find_flag, reason
    real_points = get_real_points(yaw, points, actions, camera_pose, aperture, width, height)
    return actions, real_points, find_flag, reason


def plan_and_get_actions_continuous(
    map_info,
    robot_position,
    goal,
    camera_pose,
    aperture,
    width,
    height,
    path_planner,
):
    start_pixel = world_to_pixel(robot_position, camera_pose, aperture, width, height)
    goal_pixel = world_to_pixel(goal, camera_pose, aperture, width, height)
    start_time = time.time()
    points, find_flag, reason = path_planner.planning(
        start_pixel[0],
        start_pixel[1],
        goal_pixel[0],
        goal_pixel[1],
        obs_map=map_info,
    )
    end_time = time.time()
    log.info(f'path_planning 耗时：{(end_time - start_time)} s')
    if find_flag:
        transfer_paths = []
        for node in points:
            world_coords = pixel_to_world(node, camera_pose, aperture, width, height)
            transfer_paths.append([world_coords[0], world_coords[1], robot_position[2]])
    else:
        transfer_paths = None
    if transfer_paths is not None and len(transfer_paths) > 1:
        transfer_paths.pop(0)
    return transfer_paths, find_flag, reason
