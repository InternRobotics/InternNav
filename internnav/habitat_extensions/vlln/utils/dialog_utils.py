import habitat_sim
import numpy as np
import quaternion

from internnav.habitat_extensions.vlln.simple_npc.get_description import (
    get_path_description,
    get_path_description_without_additional_info,
)

DEFAULT_IMAGE_TOKEN = "<image>"


def calculate_path_length(path):
    accumulated_length = [0]
    for i, p in enumerate(path[1:]):
        accumulated_length.append(accumulated_length[i] + np.linalg.norm(np.array(p) - np.array(path[i])))
    return accumulated_length


def get_shortest_path(env, start_position, target_position):
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start_position
    shortest_path.requested_end = target_position

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


def get_description(env, object_dict, region_dict):
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
        path_description = get_path_description(
            quaternion.from_euler_angles([0, current_yaw, 0]),
            questioned_path,
            object_dict,
            region_dict,
            height_list=[env.sim.get_agent_state().position[1]] * len(questioned_path),
        )
    except Exception as e:
        print(e)
        path_description = get_path_description_without_additional_info(
            quaternion.from_euler_angles([0, current_yaw, 0]),
            questioned_path,
            height_list=[env.sim.get_agent_state().position[1]] * len(questioned_path),
        )
    return path_description, pl
