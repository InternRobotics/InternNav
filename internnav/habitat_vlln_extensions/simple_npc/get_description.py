from collections import Counter, defaultdict
from typing import Any, Dict, List

import habitat_sim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from matplotlib.patches import Polygon

GO_INTO_ROOM = [
    "enter the {room}",
    "go into the {room}",
    "step into the {room}",
    "move into the {room}",
    "access the {room}",
    "obtain access to the {room}",
    "make your way into the {room}",
    "proceed into the {room}",
    "get into the {room}",
    "walk into the {room}",
    "step inside the {room}",
    "head into the {room}",
    "go inside the {room}",
]
TURN_BACK = [
    "turn back",
    "make a back turn",
    "take a back turn",
    "turn around",
]

TURN_ANGLE = [
    "turn {turn} about {angle} degrees",
    "make about {angle} degrees {turn} turn",
    "take about {angle} degrees {turn} turn",
    "steer to {turn} about {angle} degrees",
    "change direction to about {angle} degrees {turn}",
    "navigate about {angle} degrees {turn}",
    "execute about {angle} degrees {turn}",
    "adjust your heading to {turn} about {angle} degrees",
    "hook about {angle} degrees {turn}",
    "steer {turn} about {angle} degrees",
]
TURN = [
    "turn {turn}",
    "make a {turn} turn",
    "take a {turn} turn",
    "steer to {turn}",
    "change direction to {turn}",
    "navigate a {turn} turn",
    "execute a {turn} turn",
    "adjust your heading to {turn}",
    "hook a {turn}",
    "steer {turn}",
]

FORWARD = [
    "move forward",
    "go forward",
    "walk forward",
    "step forward",
    "proceed forward",
    "advance forward",
    "make your way forward",
    "continue ahead",
    "keep going forward",
    "progress forward",
    "keep on going",
    "go ahead",
    "trek on",
    "head straight",
    "go straight ahead",
    "keep moving forward",
]
GO_STAIRS = [
    "go {direction}stairs",
    "walk {direction}stairs",
    "climb {direction} the stairs",
    "take the stairs {direction}",
    "move {direction}stairs",
    "proceed {direction}stairs",
    "make your way {direction}stairs",
    "get {direction}stairs",
    "step {direction}stairs",
    "hop {direction}stairs",
    "run {direction} the stairs",
    "go {direction} to the next floor",
]

ROOM_START = [
    "now you are in a {room},",
    "you are in a {room},",
    "you are currently in a {room},",
    "you are now in a {room},",
    "you are standing in a {room},",
]

CONJUNCTION = [
    "and then",
    "then",
    "after that",
    "afterwards",
    "thereafter",
    "and next",
]

SHOW_PATH = [
    "your path to target object is as follows:",
    "here is your path to target object:",
    "your path to target object is:",
    "you can follow the path to target object:",
]

PREPOSITION = [
    "at the {object}",
    "beside the {object}",
    "near the {object}",
    "when see the {object}",
]

FINISH_DESCRIPTION = [
    "you are at the target",
    "you can see the target",
    "you can reach the target",
    "you can arrive at the target",
    "you can reach the destination",
    "you can arrive at the destination",
]


def is_in_poly(ps, poly):
    """
    ps: a numpy array of shape (N, 2)
    poly: a polygon represented as a list of (x, y) tuples
    """
    if isinstance(ps, tuple):
        ps = np.array([ps])
    if len(ps.shape) == 1:
        ps = np.expand_dims(ps, axis=0)
    assert ps.shape[1] == 2
    assert len(ps.shape) == 2
    path = matplotlib.path.Path(poly)
    return path.contains_points(ps)


def get_points_room(points, region_dict, object_dict, poly_type):
    """
    根据点坐标和区域多边形，返回点所在的区域列表
    habitat坐标系与ply坐标系转换关系为:
    x-habitat = x-ply
    y-habitat = z-ply
    z-habitat = -y-ply

    参数:
        point: numpy数组, 点坐标, 坐标系为habitat坐标系
        region_poly: 区域多边形字典，格式为 {区域名称: 多边形顶点列表}, 坐标系为ply坐标系
    """
    region_poly = {
        region + '/' + room['label'] + '_' + str(room['id']): room[poly_type]
        for region, region_info in region_dict.items()
        for room in region_info
    }
    point_rooms = [get_point_room(np.array([i[0], i[2]]), region_poly) for i in points]
    point_rooms = [
        [room.split('/')[0] + '/' + room.split('/')[1].split('_')[0] for room in point_room]
        for point_room in point_rooms
    ]
    # 提取物体名称和坐标
    rooms = list(set([room for point_room in point_rooms for room in point_room]))
    rooms_object_height = defaultdict(list)
    for v in object_dict.values():
        if v['scope'] + '/' + v['room'] in rooms:
            rooms_object_height[v['scope'] + '/' + v['room']].append(v['position'][1])
    rooms_object_height = {room: [min(heights), max(heights)] for room, heights in rooms_object_height.items()}
    new_point_rooms = []
    for idx, point_room in enumerate(point_rooms):
        point_room = [r for r in point_room if r in rooms_object_height]
        new_point_room = [
            r for r in point_room if rooms_object_height[r][0] - 1 < points[idx][1] < rooms_object_height[r][1]
        ]
        new_point_rooms.append(new_point_room)
    return new_point_rooms


def get_point_room(point, region_poly):
    """
    根据点坐标和区域多边形，返回点所在的区域列表
    habitat坐标系与ply坐标系转换关系为:
    x-habitat = x-ply
    y-habitat = z-ply
    z-habitat = -y-ply

    参数:
        point: numpy数组, 点坐标, 坐标系为habitat坐标系
        region_poly: 区域多边形字典，格式为 {区域名称: 多边形顶点列表}, 坐标系为ply坐标系
    """
    if len(point.shape) == 1:
        point = np.expand_dims(point, axis=0)
    point[:, 1] = -point[:, 1]
    regions = []
    for region, poly in region_poly.items():
        if is_in_poly(point, poly):
            regions.append(region)
    return regions


def get_room_name(room):
    room_name_dict = {
        "living region": "living room",
        "stair region": "stairs",
        "bathing region": "bathroom",
        "storage region": "storage room",
        "study region": "study room",
        "cooking region": "kitchen",
        "sports region": "sports room",
        "corridor region": "corridor",
        "toliet region": "toilet",
        "dinning region": "dining room",
        "resting region": "resting room",
        "open area region": "open area",
        "other region": "area",
    }
    return room_name_dict[room]


def get_start_description(angle2first_point, height_diff, room=None):
    # des = np.random.choice(ROOM_START).format(room=get_room_name(room)) + ' ' + np.random.choice(SHOW_PATH) + '\n'
    des = ''
    if height_diff > 0.1:
        des += str(des.count('\n') + 1) + '. ' + np.random.choice(GO_STAIRS).format(direction='up') + ', '
    elif height_diff < -0.1:
        des += str(des.count('\n') + 1) + '. ' + np.random.choice(GO_STAIRS).format(direction='down') + ', '
    else:
        des += str(des.count('\n') + 1) + '. ' + np.random.choice(FORWARD) + ' along the direction '
        if abs(angle2first_point) >= 120:
            des += 'after you ' + np.random.choice(TURN_BACK) + ' from your current view, '
        elif angle2first_point > 20:
            des += (
                'after you '
                + np.random.choice(TURN_ANGLE).format(turn='left', angle=int(round(angle2first_point, -1)))
                + ' from your current view, '
            )
        elif angle2first_point < -20:
            des += (
                'after you '
                + np.random.choice(TURN_ANGLE).format(turn='right', angle=int(round(abs(angle2first_point), -1)))
                + ' from your current view, '
            )
        else:
            des += 'from your current view, '
    return des


def find_shortest_path(env, target_position):
    """
    在habitat环境中找到从当前位置到目标位置的最短路径

    参数:
        env: habitat环境实例
        target_position: 目标位置坐标, numpy数组 [x, y, z]

    返回:
        path: 路径点列表
        success: 是否找到有效路径
    """

    # 获取当前agent位置
    current_position = [float(i) for i in env._sim.get_agent_state().position]

    # 创建路径规划器
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = current_position
    shortest_path.requested_end = target_position

    # 计算最短路径
    success = env.sim.pathfinder.find_path(shortest_path)
    path = shortest_path.points
    return path, success


def get_shortest_path_description(target_position, env, object_dict):
    path, success = find_shortest_path(env, target_position)
    if not success:
        print("未找到有效路径")
        return ''
    current_orientation = env.sim.get_agent_state().rotation
    path_description = get_path_description(current_orientation, path, object_dict)
    return path_description


def get_object_name(point_info, object_dict):
    object_name = point_info['object']
    object_infos_in_room = {
        obj: obj_info
        for obj, obj_info in object_dict.items()
        if obj_info['scope'] == object_dict[object_name]['scope']
        and obj_info['room'] == object_dict[object_name]['room']
    }
    sorted_objects = dict(
        sorted(
            object_infos_in_room.items(),
            key=lambda x: np.linalg.norm(
                np.array([x[1]["position"][i] for i in [0, 2]]) - np.array([point_info['position'][i] for i in [0, 2]])
            ),
        )
    )
    for _, obj_info in sorted_objects.items():
        if abs(obj_info['position'][1] - point_info['position'][1]) > 2:
            continue
        if obj_info['category'] in ['floor', 'ceiling', 'wall']:
            continue
        if isinstance(obj_info['unique_description'], dict):
            adjectives = {
                adj_name: adj
                for adj_name, adj in obj_info['unique_description'].items()
                if adj_name in ['color', 'texture', 'material'] and adj != ''
            }
            if len(adjectives) > 0:
                adj_name = np.random.choice(list(adjectives.keys()))
                if adj_name == 'texture':
                    return obj_info['category'] + ' with ' + adjectives[adj_name].lower() + ' texture'
                else:
                    return adjectives[adj_name].lower() + ' ' + obj_info['category']
        return obj_info['category']
    return None


def get_path_description_without_additional_info(
    orientation: np.ndarray, path: List[np.ndarray], height_list: list = None
):
    if len(path) == 0:
        return ''
    path_info = {idx: {'position': i, 'calc_trun': False, 'turn': []} for idx, i in enumerate(path)}
    # 确定上楼的点
    if height_list is None:
        for i in range(len(path_info) - 1):
            if path_info[i + 1]['position'][1] - path_info[i]['position'][1] > 0.1:
                path_info[i]['turn'].append('up')
            elif path_info[i + 1]['position'][1] - path_info[i]['position'][1] < -0.1:
                path_info[i]['turn'].append('down')
    else:
        assert len(height_list) == len(path), 'height_list and path have different length'
        for i in range(len(height_list) - 1):
            if height_list[i + 1] - height_list[i] > 0.1:
                path_info[i]['turn'].append('up')
            elif height_list[i + 1] - height_list[i] < -0.1:
                path_info[i]['turn'].append('down')
    calc_turn_indices, _ = sample_points([pi['position'] for pi in path_info.values()], [''] * len(path_info), 1.0)
    for i in calc_turn_indices:
        path_info[i]['calc_trun'] = True
    # 正数代表左转，负数代表右转
    new2origin = {new: origin for new, origin in enumerate(calc_turn_indices)}
    move_z_point_to_sky = np.array([path_info[i]['position'] for i in calc_turn_indices]) @ np.array(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    )
    turn_points, turn_angles = find_sharp_turns(move_z_point_to_sky, threshold=40)
    for i, indice in enumerate(turn_points):
        path_info[new2origin[indice]]['turn'].append(turn_angles[i])
    special_point = [i for i in path_info.keys() if len(path_info[i]['turn']) > 0 and i != 0]
    path_description = ''
    # get initial conjunction
    angle2first_point = compute_yaw_rotation(
        orientation, path_info[calc_turn_indices[0]]['position'], path_info[calc_turn_indices[1]]['position']
    )
    height_diff = (
        path_info[calc_turn_indices[1]]['position'][1] - path_info[calc_turn_indices[0]]['position'][1]
        if height_list is None
        else height_list[calc_turn_indices[1]] - height_list[calc_turn_indices[0]]
    )
    path_description += get_start_description(angle2first_point, height_diff)

    last_special_point = 0
    for i in special_point:
        if len(path_info[i]['turn']) > 0:
            for turn in path_info[i]['turn']:
                if isinstance(turn, str):
                    continue
                if turn > 0:
                    length = round(
                        np.linalg.norm(
                            np.array(path_info[i]['position']) - np.array(path_info[last_special_point]['position'])
                        )
                    )
                    path_description += (
                        np.random.choice(CONJUNCTION)
                        + ' '
                        + np.random.choice(TURN).format(turn='left')
                        + ' '
                        + f'after walking around {length} meters'
                        + ', '
                    )
                else:
                    length = round(
                        np.linalg.norm(
                            np.array(path_info[i]['position']) - np.array(path_info[last_special_point]['position'])
                        )
                    )
                    path_description += (
                        np.random.choice(CONJUNCTION)
                        + ' '
                        + np.random.choice(TURN).format(turn='right')
                        + ' '
                        + f'after walking around {length} meters'
                        + ', '
                    )

            if 'up' in path_info[i]['turn']:
                path_description += (
                    np.random.choice(CONJUNCTION) + ' ' + np.random.choice(GO_STAIRS).format(direction='up') + '\n'
                )
                path_description += str(path_description.count('\n') + 1) + '. '
                continue
            elif 'down' in path_info[i]['turn']:
                path_description += (
                    np.random.choice(CONJUNCTION) + ' ' + np.random.choice(GO_STAIRS).format(direction='down') + '\n'
                )
                path_description += str(path_description.count('\n') + 1) + '. '
                continue
        path_description += '\n'
        path_description += str(path_description.count('\n') + 1) + '. ' + np.random.choice(FORWARD) + ', '
    return path_description, path_info


def get_path_description(
    orientation: np.ndarray,
    path: List[np.ndarray],
    object_dict: Dict[str, Dict[str, Any]],
    region_dict: Dict[str, Dict[str, Any]],
    return_finish: bool = True,
    height_list: list = None,
):
    """
    Generate a description of the given path.

    Parameters:
        orientation (np.ndarray): The current orientation of the agent.
        path (list): A list of points representing the path the agent needs to the target position.
        object_dict (dict): A dictionary containing information about the objects in the scene.

    Returns:
        str: A string describing the path. Returns an empty string if the path is empty.
    """
    if len(path) == 0:
        return ''
    path_info = get_passed_objects_and_regions(path, object_dict, region_dict, height_list)
    special_point = [
        i for i in path_info.keys() if (path_info[i]['new_room'] or len(path_info[i]['turn']) > 0) and i != 0
    ]
    path_description = ''
    # get initial conjunction
    angle2first_point = compute_yaw_rotation(orientation, path_info[0]['position'], path_info[1]['position'])
    height_diff = (
        path_info[1]['position'][1] - path_info[0]['position'][1]
        if height_list is None
        else height_list[1] - height_list[0]
    )
    path_description += get_start_description(
        angle2first_point, height_diff, object_dict[path_info[0]['object']]['room']
    )

    for i in special_point:
        if path_info[i]['new_room'] and object_dict[path_info[i]['object']]['room'] != 'stair region':
            path_description += (
                np.random.choice(CONJUNCTION)
                + ' '
                + np.random.choice(GO_INTO_ROOM).format(room=get_room_name(object_dict[path_info[i]['object']]['room']))
                + ', '
            )
        if len(path_info[i]['turn']) > 0:
            object_name = get_object_name(path_info[i], object_dict)
            for turn in path_info[i]['turn']:
                if isinstance(turn, str):
                    continue
                if turn > 0:
                    path_description += (
                        np.random.choice(CONJUNCTION)
                        + ' '
                        + np.random.choice(TURN).format(turn='left')
                        + ' '
                        + np.random.choice(PREPOSITION).format(object=object_name)
                        + ', '
                    )
                else:
                    path_description += (
                        np.random.choice(CONJUNCTION)
                        + ' '
                        + np.random.choice(TURN).format(turn='right')
                        + ' '
                        + np.random.choice(PREPOSITION).format(object=object_name)
                        + ', '
                    )

            if 'up' in path_info[i]['turn']:
                path_description += (
                    np.random.choice(CONJUNCTION) + ' ' + np.random.choice(GO_STAIRS).format(direction='up') + '\n'
                )
                path_description += str(path_description.count('\n') + 1) + '. '
                continue
            elif 'down' in path_info[i]['turn']:
                path_description += (
                    np.random.choice(CONJUNCTION) + ' ' + np.random.choice(GO_STAIRS).format(direction='down') + '\n'
                )
                path_description += str(path_description.count('\n') + 1) + '. '
                continue
        path_description += '\n'
        path_description += str(path_description.count('\n') + 1) + '. ' + np.random.choice(FORWARD) + ', '
    if return_finish:
        path_description += np.random.choice(CONJUNCTION) + ' ' + np.random.choice(FINISH_DESCRIPTION)
        return path_description
    else:
        return path_description, path_info


def plot_polygons(polygons, colors=None):
    """
    参数:
        polygons: List of np.array, 每个元素是一个 Nx2 的顶点数组
        colors: 可选 list，指定每个 poly 的颜色（字符串或 RGB）
    """
    fig, ax = plt.subplots()
    for i, poly in enumerate(polygons):
        color = colors[i] if colors and i < len(colors) else 'blue'
        patch = Polygon(poly, closed=True, facecolor=color, edgecolor='black', alpha=0.5)
        ax.add_patch(patch)
    all_points = np.vstack(polygons)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)

    ax.set_aspect('equal')
    plt.axis('on')
    plt.savefig('polygons.png')
    plt.close()


def fill_empty_with_nearest(strings):
    n = len(strings)
    result = strings[:]

    # 记录每个位置到最近非空字符串的左侧值
    left = [''] * n
    last = ''
    for i in range(n):
        if strings[i]:
            last = strings[i]
        left[i] = last

    # 记录右侧最近非空字符串
    right = [''] * n
    last = ''
    for i in range(n - 1, -1, -1):
        if strings[i]:
            last = strings[i]
        right[i] = last

    # 替换为空的位置
    for i in range(n):
        if strings[i] == '':
            if left[i] and right[i]:
                # 选择更近的一个（如果两边都有）
                dist_left = i - next(j for j in range(i, -1, -1) if strings[j])
                dist_right = next(j for j in range(i, n) if strings[j]) - i
                result[i] = left[i] if dist_left <= dist_right else right[i]
            else:
                # 只存在一边的
                result[i] = left[i] or right[i]

    return result


def minimize_unique_strings(list_of_lists):
    # 统计所有出现词的频率
    flat = [s for sublist in list_of_lists for s in sublist]
    freq = Counter(flat)

    result = []
    for idx, options in enumerate(list_of_lists):
        # 在当前选项中，选择频率最高的词（最常见词)
        if len(options) == 0:
            best = ''
        else:
            best = min(options, key=lambda x: (freq[x], x))  # tie-breaker: alphabet
        result.append(best)
    return result


def get_nearest_object(path, region_dict, object_dict):
    point_rooms = get_points_room(path, region_dict, object_dict, 'poly')
    point_rooms = minimize_unique_strings(point_rooms)
    point_rooms = fill_empty_with_nearest(point_rooms)
    # 提取物体名称和坐标
    rooms = list(set(point_rooms))
    if '' in rooms:
        point_rooms = get_points_room(path, region_dict, object_dict, 'enlarge_poly')
        point_rooms = minimize_unique_strings(point_rooms)
        point_rooms = fill_empty_with_nearest(point_rooms)
        rooms = list(set(point_rooms))
    rooms_object_positions = defaultdict(dict)
    for k, v in object_dict.items():
        if v['scope'] + '/' + v['room'] in rooms and v['category'] not in [
            'floor',
            'ceiling',
            'column',
            'wall',
            'light',
        ]:
            rooms_object_positions[v['scope'] + '/' + v['room']][k] = np.array([v['position'][0], v['position'][2]])
    assert len(rooms_object_positions) == len(rooms), 'exist room has no object'
    # 找到区域内每个查询点的最近物体索引
    nearest_indices = [
        np.linalg.norm(
            np.array([i[0], i[2]]) - np.array(list(rooms_object_positions[point_rooms[idx]].values())), axis=1
        ).argmin()
        for idx, i in enumerate(path)
    ]

    # **获取最近物体名称**
    nearest_objects = [
        list(rooms_object_positions[point_rooms[idx]].keys())[nearest_indices[idx]] for idx in range(len(path))
    ]
    return nearest_objects


def get_passed_objects_and_regions(path, object_dict, region_dict, height_list=None):
    """
    根据路径和区域的边界框，输出路径经过的区域名称

    参数:
        path: 路径点列表，格式为 [[x1, y1, z1], [x2, y2, z2], ...]
        bboxes: 每个区域的边界框字典，格式为 {区域名称: [[x_min, y_min, z_min], [x_max, y_max, z_max]], ...}

    返回:
        passed_regions: 路径经过的区域名称列表
    """
    # 获取区域内最近物体名称
    nearest_objects = get_nearest_object(path, region_dict, object_dict)
    path_info = {
        idx: {'position': path[idx], 'object': obj, 'calc_trun': False, 'turn': [], 'new_room': False}
        for idx, obj in enumerate(nearest_objects)
    }

    # 确定上楼的点
    if height_list is None:
        for i in range(len(path_info) - 1):
            if path_info[i + 1]['position'][1] - path_info[i]['position'][1] > 0.1:
                path_info[i]['turn'].append('up')
            elif path_info[i + 1]['position'][1] - path_info[i]['position'][1] < -0.1:
                path_info[i]['turn'].append('down')
    else:
        assert len(height_list) == len(path), 'height_list and path have different length'
        for i in range(len(height_list) - 1):
            if height_list[i + 1] - height_list[i] > 0.1:
                path_info[i]['turn'].append('up')
            elif height_list[i + 1] - height_list[i] < -0.1:
                path_info[i]['turn'].append('down')
    calc_turn_indices, room_change_indices = sample_points(
        [pi['position'] for pi in path_info.values()],
        [object_dict[pi['object']]['room'] for pi in path_info.values()],
        1.0,
    )
    for i in calc_turn_indices:
        path_info[i]['calc_trun'] = True
    for i in room_change_indices:
        path_info[i]['new_room'] = True
    # 正数代表左转，负数代表右转
    new2origin = {new: origin for new, origin in enumerate(calc_turn_indices)}
    move_z_point_to_sky = np.array([path_info[i]['position'] for i in calc_turn_indices]) @ np.array(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    )
    turn_points, turn_angles = find_sharp_turns(move_z_point_to_sky, threshold=40)
    for i, indice in enumerate(turn_points):
        path_info[new2origin[indice]]['turn'].append(turn_angles[i])
    return path_info


def sample_points(points, rooms, min_dist=1.0):
    """
    对坐标列表进行采样，使得每个点之间的距离都大于 min_dist
    :param points: 形如 [(x, y, z), (x, y, z), ...] 的坐标列表
    :param min_dist: 采样的最小距离（单位：米）
    :return: 采样后的点对应的原列表索引
    """
    points = np.array(points)
    selected_indices = [0]  # 选中第一个点
    last_selected_point = points[0]

    room_change_indices = [0]
    last_room = rooms[0]

    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - last_selected_point) >= min_dist:
            selected_indices.append(i)
            last_selected_point = points[i]
        if rooms[i] != last_room:
            room_change_indices.append(i)
            last_room = rooms[i]
    if len(selected_indices) == 1:
        selected_indices.append(len(points) - 1)

    return selected_indices, room_change_indices


def find_sharp_turns(path_points, threshold=30):
    """
    找出路径中所有转弯大于 `threshold` 度的点，并判断左转还是右转以及转角度数。

    :param path_points: 形如 [(x, y, z), (x, y, z), ...] 的路径点列表
    :param threshold: 角度阈值（默认 30 度）
    :return: (拐点索引列表, 转向角度列表)（正数 = 左转, 负数 = 右转）
    """
    path_points = np.array(path_points)

    # 计算向量：v1 = (p1 -> p2), v2 = (p2 -> p3)
    v1 = path_points[1:-1] - path_points[:-2]  # 前向向量
    v2 = path_points[2:] - path_points[1:-1]  # 后向向量

    # 计算单位向量（归一化）
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)

    # 避免除以零
    v1 = np.divide(v1, v1_norm, where=(v1_norm != 0))
    v2 = np.divide(v2, v2_norm, where=(v2_norm != 0))

    # 计算夹角 cos(theta) = (v1 · v2)
    cos_theta = np.sum(v1 * v2, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差

    # 转换为角度（degree）
    angles = np.degrees(np.arccos(cos_theta))

    # 计算叉积 v1 × v2（用于判断左转还是右转）
    cross_products = np.cross(v1, v2)  # (N-2, 3)

    # 选取 Z 轴分量 cross_z
    cross_z = cross_products[:, 2]  # 仅适用于水平面运动（xy 平面）

    # 计算转向角度：左转为正，右转为负
    turn_angles = angles * np.sign(cross_z)

    # 找出大于 threshold 的拐点索引
    sharp_turn_indices = np.where(np.abs(turn_angles) > threshold)[0] + 1

    return sharp_turn_indices, turn_angles[sharp_turn_indices - 1]


def compute_yaw_rotation(agent_quat, current_pos, target_pos):
    """
    计算 agent 沿 Y 轴旋转的角度:
    - 正数表示左转该角度
    - 负数表示右转该角度

    :param agent_quat: 当前 agent 的四元数(np.quaternion)
    :param current_pos: 当前坐标 (x, y, z)
    :param target_pos: 目标坐标 (x, y, z)
    :return: 旋转角度（度）(正数 = 左转，负数 = 右转)
    """
    # 1. 计算目标方向向量（忽略 y 轴变化，只看水平旋转）
    direction = np.array(target_pos) - np.array(current_pos)
    direction[1] = 0  # 只考虑 XZ 平面的旋转
    direction = direction / np.linalg.norm(direction)  # 归一化

    # 2. 获取当前 agent 朝向的前向向量（使用四元数旋转单位向量）
    forward = np.array([0, 0, -1])  # 假设 agent 默认前向 -Z 方向
    agent_forward = quaternion.as_rotation_matrix(agent_quat) @ forward

    # 3. 计算旋转轴（叉积得到法向量）
    axis = np.cross(agent_forward, direction)
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-6 else np.array([0, 1, 0])  # 避免零向量

    # 4. 计算旋转角度（点积得到夹角）
    cos_theta = np.dot(agent_forward, direction)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止浮点误差
    theta_rad = np.arccos(cos_theta)  # 计算角度（弧度制）
    theta_deg = np.degrees(theta_rad)  # 转换为角度

    # 5. 计算旋转方向（左转为正，右转为负）
    return theta_deg if axis[1] > 0 else -theta_deg
