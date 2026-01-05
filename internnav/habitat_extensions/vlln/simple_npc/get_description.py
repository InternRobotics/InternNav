from collections import Counter, defaultdict
from typing import Any, Dict, List

import matplotlib
import numpy as np
import quaternion

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
    if isinstance(ps, tuple):
        ps = np.array([ps])
    if len(ps.shape) == 1:
        ps = np.expand_dims(ps, axis=0)
    assert ps.shape[1] == 2
    assert len(ps.shape) == 2
    path = matplotlib.path.Path(poly)
    return path.contains_points(ps)


def get_points_room(points, region_dict, object_dict, poly_type):
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

    # Extract object names and coordinates
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
    """Given a point coordinate and region polygons, return the list of regions that contain the point.
    The coordinate transform between the Habitat coordinate system and the PLY coordinate system is:
        x_habitat = x_ply
        y_habitat = z_ply
        z_habitat = -y_ply

    Args:
        point (np.ndarray): A NumPy array representing the point coordinate in the Habitat coordinate system.
        region_poly (Dict[str, List[np.ndarray]]): A dictionary of region polygons.

    Returns:
        List[str]: A list of region names whose polygon contains the given point(s).
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
    """Generate a natural-language description of a navigation path without using scene object/room metadata.

    Args:
        orientation (np.ndarray): Current agent orientation.
        path (List[np.ndarray]): Sequence of 3D waypoints of length T that leads toward the target position.
        height_list (Optional[list]): Optional per-step height values of length T.

    Returns:
        str: A multi-line path instruction string.
    """
    if len(path) == 0:
        return ''
    path_info = {idx: {'position': i, 'calc_trun': False, 'turn': []} for idx, i in enumerate(path)}
    # get the point that changes floor
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
    # positive -> left，negative -> right
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
    return path_description


def get_path_description(
    orientation: np.ndarray,
    path: List[np.ndarray],
    object_dict: Dict[str, Dict[str, Any]],
    region_dict: Dict[str, Dict[str, Any]],
    height_list: list = None,
):
    """Generate a natural-language step-by-step description of a navigation path.

    Args:
        orientation (np.ndarray): Current agent orientation.
        path (List[np.ndarray]): Sequence of 3D waypoints of length T that leads toward the target position.
        object_dict (Dict[str, Dict[str, Any]]): Object metadata dictionary.
        region_dict (Dict[str, Dict[str, Any]]): Region/room metadata used to assign waypoints to rooms and detect room
            transitions.
        height_list (Optional[list]): Optional per-step height values of length T.

    Returns:
        str: A multi-line path instruction string.
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
    return path_description


def fill_empty_with_nearest(strings):
    n = len(strings)
    result = strings[:]

    left = [''] * n
    last = ''
    for i in range(n):
        if strings[i]:
            last = strings[i]
        left[i] = last

    right = [''] * n
    last = ''
    for i in range(n - 1, -1, -1):
        if strings[i]:
            last = strings[i]
        right[i] = last

    for i in range(n):
        if strings[i] == '':
            if left[i] and right[i]:
                dist_left = i - next(j for j in range(i, -1, -1) if strings[j])
                dist_right = next(j for j in range(i, n) if strings[j]) - i
                result[i] = left[i] if dist_left <= dist_right else right[i]
            else:
                result[i] = left[i] or right[i]

    return result


def minimize_unique_strings(list_of_lists):
    flat = [s for sublist in list_of_lists for s in sublist]
    freq = Counter(flat)

    result = []
    for idx, options in enumerate(list_of_lists):
        if len(options) == 0:
            best = ''
        else:
            best = min(options, key=lambda x: (freq[x], x))  # tie-breaker: alphabet
        result.append(best)
    return result


def get_nearest_object(path, region_dict, object_dict):
    """Determine the nearest valid object to each point along a navigation path.

    Args:
        path (List[List[float]]): Sequence of 3D positions of shape (T, 3).
        region_dict (dict): Region/room metadata used to assign each path point to a room.
        object_dict (dict): Object metadata dictionary.

    Returns:
        List[str]: A list of object identifiers of length ``T``, where each element corresponds to the nearest object
            associated with the same room as the corresponding path point.
    """
    point_rooms = get_points_room(path, region_dict, object_dict, 'poly')
    point_rooms = minimize_unique_strings(point_rooms)
    point_rooms = fill_empty_with_nearest(point_rooms)
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
    nearest_indices = [
        np.linalg.norm(
            np.array([i[0], i[2]]) - np.array(list(rooms_object_positions[point_rooms[idx]].values())), axis=1
        ).argmin()
        for idx, i in enumerate(path)
    ]

    nearest_objects = [
        list(rooms_object_positions[point_rooms[idx]].keys())[nearest_indices[idx]] for idx in range(len(path))
    ]
    return nearest_objects


def get_passed_objects_and_regions(path, object_dict, region_dict, height_list=None):
    """Annotate a navigation path with nearest objects, room transitions, and turn events.

    Args:
        path (List[List[float]]): Sequence of 3D positions of shape (T, 3).
        object_dict (dict): Object metadata dictionary.
        region_dict (dict): Region/room metadata used to compute nearest objects.
        height_list (Optional[List[float]]): Optional per-step height values of length ``T``.

    Returns:
        dict: A dictionary keyed by waypoint index. Each entry contains:
            - ``position``: The 3D position at this index.
            - ``object``: Nearest object for this waypoint.
            - ``calc_trun``: Whether this index is selected for turn computation.
            - ``turn``: A list of turn annotations (may include "up"/"down" and/or
              signed turn angles in degrees).
            - ``new_room``: Whether this index marks entering a new room/region.
    """
    nearest_objects = get_nearest_object(path, region_dict, object_dict)
    path_info = {
        idx: {'position': path[idx], 'object': obj, 'calc_trun': False, 'turn': [], 'new_room': False}
        for idx, obj in enumerate(nearest_objects)
    }

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
    new2origin = {new: origin for new, origin in enumerate(calc_turn_indices)}
    move_z_point_to_sky = np.array([path_info[i]['position'] for i in calc_turn_indices]) @ np.array(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    )
    turn_points, turn_angles = find_sharp_turns(move_z_point_to_sky, threshold=40)
    for i, indice in enumerate(turn_points):
        path_info[new2origin[indice]]['turn'].append(turn_angles[i])
    return path_info


def sample_points(points, rooms, min_dist=1.0):
    """Subsample a list of 3D points so that the distance between any two selected points is greater than
    or equal to `min_dist`.

    Args:
        points (List[Tuple[float, float, float]] | np.ndarray): A list of coordinates in the form
            [(x, y, z), (x, y, z), ...].
        rooms (List[str] | List[int] | np.ndarray): A sequence of room identifiers corresponding one-to-one with
            `points`. Each entry indicates the room in which the point lies.
        min_dist (float): Minimum allowed Euclidean distance (in meters) between two selected points.

    Returns:
        List[int]: Indices of the selected points in the original `points` sequence.
        List[int]: Indices where the room label changes compared to the previous point.
    """
    points = np.array(points)
    selected_indices = [0]  # pick the first point
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
    """Identify all points along a path where the turning angle exceeds `threshold` degrees, and determine
    whether each turn is a left or right turn along with its angle.

    Args:
        path_points (List[Tuple[float, float, float]] | np.ndarray): A list of path points in the form
            [(x, y, z), (x, y, z), ...].
        threshold (float | int): Turning angle threshold in degrees (default: 30 degrees).

    Returns:
        np.ndarray: indices (into the original path) for sharp turns.
        np.ndarray: signed turning angles (degrees) at those indices.
    """
    path_points = np.array(path_points)

    v1 = path_points[1:-1] - path_points[:-2]
    v2 = path_points[2:] - path_points[1:-1]

    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)

    v1 = np.divide(v1, v1_norm, where=(v1_norm != 0))
    v2 = np.divide(v2, v2_norm, where=(v2_norm != 0))

    cos_theta = np.sum(v1 * v2, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angles = np.degrees(np.arccos(cos_theta))

    cross_products = np.cross(v1, v2)  # (N-2, 3)

    cross_z = cross_products[:, 2]
    turn_angles = angles * np.sign(cross_z)
    sharp_turn_indices = np.where(np.abs(turn_angles) > threshold)[0] + 1

    return sharp_turn_indices, turn_angles[sharp_turn_indices - 1]


def compute_yaw_rotation(agent_quat, current_pos, target_pos):
    """Compute the agent's rotation angle about the Y axis:
    - Positive values indicate a left turn by that angle.
    - Negative values indicate a right turn by that angle.

    Args:
        agent_quat (np.quaternion): The agent's current quaternion (np.quaternion).
        current_pos (Tuple[float, float, float] | List[float] | np.ndarray): Current position (x, y, z).
        target_pos (Tuple[float, float, float] | List[float] | np.ndarray): Target position (x, y, z).

    Returns:
        float: Rotation angle in degrees (positive = left turn, negative = right turn).
    """
    direction = np.array(target_pos) - np.array(current_pos)
    direction[1] = 0
    direction = direction / np.linalg.norm(direction)

    forward = np.array([0, 0, -1])
    agent_forward = quaternion.as_rotation_matrix(agent_quat) @ forward

    axis = np.cross(agent_forward, direction)
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-6 else np.array([0, 1, 0])

    cos_theta = np.dot(agent_forward, direction)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)

    return theta_deg if axis[1] > 0 else -theta_deg
