import copy
import gzip
import json
import os
from collections import defaultdict

import numpy as np

from internnav.utils.common_log_util import common_logger as log

fall_path_z_0_3 = [
    70,
    121,
    146,
    156,
    172,
    326,
    349,
    372,
    394,
    415,
    434,
    469,
    531,
    550,
    580,
    626,
    674,
    700,
    768,
    808,
    823,
    835,
    854,
    859,
    958,
    1009,
    1058,
    1065,
    1093,
    1105,
    1142,
    1205,
    1238,
    1245,
    1263,
    1290,
    1295,
    1353,
    1400,
    1403,
    1455,
    1470,
    1530,
    1644,
    1645,
    1650,
    1734,
    1771,
    1848,
    1876,
    1880,
    1893,
    1925,
    1928,
    1957,
    1967,
    1995,
    2051,
    2061,
    2100,
    2101,
    2102,
    2156,
    2173,
    2186,
    2252,
    2253,
    2296,
    2335,
    2360,
    2399,
    2441,
    2485,
    2502,
    2508,
    2530,
    2591,
    2609,
    2622,
    2632,
    2651,
    2676,
    2744,
    2752,
    2809,
    2871,
    2911,
    2951,
    2967,
    2968,
    2981,
    2991,
    3023,
    3031,
    3032,
    3078,
    3093,
    3115,
    3145,
    3156,
    3160,
    3183,
    3194,
    3291,
    3304,
    3351,
    3528,
    3534,
    3576,
    3596,
    3605,
    3629,
    3656,
    3665,
    3689,
    3733,
    3749,
    3789,
    3833,
    3838,
    3859,
    3863,
    3868,
    3890,
    3978,
    3984,
    3993,
    4005,
    4022,
    4112,
    4122,
    4136,
    4214,
    4257,
    4264,
    4281,
    4311,
    4318,
    4356,
    4407,
    4460,
    4467,
    4533,
    4536,
    4551,
    4586,
    4656,
    4694,
    4698,
    4725,
    4800,
    4805,
    4807,
    4848,
    4867,
    4927,
    4949,
    5103,
    5170,
    5176,
    5228,
    5325,
    5327,
    5427,
    5443,
    5462,
    5529,
    5552,
    5625,
    5660,
    5690,
    5703,
    5753,
    5757,
    5817,
    5900,
    5928,
    5948,
    5955,
    6004,
    6109,
    6113,
    6120,
    6141,
    6181,
    6206,
    6221,
    6260,
    6283,
    6404,
    6422,
    6529,
    6608,
    6631,
    6660,
    6713,
    6731,
    6736,
    6749,
    6786,
    6800,
    6913,
    6916,
    6938,
    6971,
    6993,
    7021,
    7052,
    7145,
    7180,
    7202,
    7264,
    3477,
    5197,
    6372,
    4175,
    5929,
    7029,
    1924,
    2376,
    4877,
    6463,
    765,
    4415,
    5133,
    59,
    246,
    592,
    604,
    952,
    1185,
    1362,
    2680,
    3727,
    839,
    1444,
    274,
    3265,
    3592,
    4514,
    5847,
    6005,
    6599,
    2461,
    3703,
    219,
    1731,
    1822,
    6055,
    6142,
    7289,
    5280,
    41,
    1982,
    2108,
    2247,
    2554,
    3853,
    4818,
    6768,
    6794,
    7003,
    7033,
    2733,
    4860,
    606,
    1200,
    1083,
    6039,
    651,
    797,
    1014,
    4006,
    5454,
    6826,
    6899,
    6933,
    6373,
    1415,
    1418,
    2457,
    4691,
    6342,
    621,
    602,
    946,
    5431,
    6163,
    6208,
    890,
    1668,
    2031,
    4161,
    4826,
    6183,
    1592,
    3645,
    4376,
    109,
    369,
    743,
    1432,
    2147,
    2190,
    3946,
    5720,
    6680,
    2994,
    3039,
    3781,
    4754,
    4920,
    6774,
    6942,
    2950,
    5624,
    3960,
    4890,
    4994,
    6036,
    2306,
]

skip_list = []

fall_path_custom = {
    6558: [-1, 0, 0],
    454: [0.42, 0.9, 0],
    490: [0.97, 0.25, 0],
    910: [-0.4, 0, 0],
    1253: [-0.4, 0, 0],
    1834: [0, -0.5, 0.3],
    2004: [0.5, 0.5, 0],
    2249: [1, -1, 0],
    2382: [1, -0.5, 0],
    2468: [0.2, 0, 0],
    2498: [-0.2, -0.5, 0],
    2523: [1, 0, 0],
    2529: [1, 0.3, 0],
    2618: [-0.5, 0.2, 0.3],
    2688: [0, -1, 0],
    2768: [-0.86, 0.52, 0],
    3084: [0.88, -0.47, 0],
    3136: [1.0, 0, 0],
    3165: [0, 0, 0.8],
    3231: [0, -0.5, 0.3],
    3277: [0, 1, 0.3],
    3414: [0.5, 0, 0.3],
    3464: [0.7, -1, 0],
    3468: [-0.5, 0, 0],
    3686: [0.2, 0.2, 0],
    4073: [-0.24, 0.5, 0],
    4243: [0.2, 0, 0],
    4305: [0, -0.2, 0],
    4564: [-0.5, 0, 0],
    5252: [0.2, 0, 0.3],
    5328: [0, 0.5, 0],
    5401: [-1, -0.2, 0.0],
    5461: [-1.0, 0, 0.3],
    5560: [0, -0.5, 0.0],
    5609: [0.5, 0, 0.3],
    5930: [0.5, 0, 0],
    6262: [-0.5, 0, 0],
    6640: [0, -0.5, 0],
    6840: [0, -0.5, 0],
    6914: [0, -0.5, 0],
    7108: [0.5, 0, 0],
    7229: [0, -0.5, 0],
    7246: [0, 0.2, 0],
    7273: [0.5, 0, 0],
    338: [1, 1.2, 0.3],
    435: [0, 1, 0],
    2965: [0, 1, 0],
    3258: [0, 1, 0],
    1483: [0.5, 0, 0.3],
    5256: [0.8, 0, 0],
    1234: [0.2, -0.2, 0],
    1954: [0.2, -0.2, 0],
    2322: [0.2, 1, 0],
    6390: [0.2, 1, 0],
    6672: [0, 0.5, 0],
    5372: [0.5, 0, 0],
    2357: [0.3, -0.3, 0],
    95: [0.2, -0.5, 0],
    2778: [0.4, -0.5, 0],
    7281: [0.2, -0.5, 0],
    332: [-0.3, 0, 0],
    648: [-0.3, 0, 0],
    2716: [-0.2, 0, 0],
    2896: [0.2, 0.2, 0],
    3028: [0.2, 0.2, 0],
    3754: [0, 0.2, 0],
    4463: [-0.1, 0, 0],
    4615: [-0.1, 0, 0],
    5773: [-0.1, 0, 0],
    6783: [0.5, 0, 0],
    801: [0.5, 0, 0],
    5661: [0.5, 0, 0],
    675: [0, 0.5, 0],
    6526: [-0.5, 0, 0],
    7285: [-0.5, 0, 0],
    622: [0, -0.3, 0.3],
    4746: [0, -0.3, 0.3],
    1623: [0, -0.5, 0],
    5574: [0, 0.5, 0],
    1847: [0, 1.2, 0],
    2470: [0, 1.2, 0],
    2240: [-1, 0, 0],
    6694: [0, 0.2, 0],
    2180: [0.5, 0, 0],
    138: [0.5, -0.1, 0.1],
    175: [0.2, 0, 0],
    1899: [0.2, 0.2, 0],
    3858: [0, -2, 0.1],
    3952: [0.5, -0.1, 0.1],
    4156: [0.5, -0.1, 0.1],
    6077: [0.2, 0.2, 0],
    6875: [-0.2, -0.2, 0],
    7007: [-0.2, -0.2, 0],
    498: [0.5, 0, 0],
    3406: [0.5, 0, 0],
    3627: [-0.2, -0.5, 0],
    4239: [-0.3, 0, 0],
    412: [0, -0.1, 0],
    3347: [0, -0.1, 0],
    1944: [-0.2, -0.2, 0],
    2668: [-0.2, -0.2, 0],
    2749: [-0.5, 0, 0],
    1182: [0, -0.6, 0],
}


def revise_one_data(origin):
    """
    Apply an offset amendment to the start position and first waypoint of the reference path
    for a given trajectory, if it belongs to a known fall-amend trajectory group.

    The offset is selected based on:
      - `fall_path_z_0_3` → fixed offset [0, 0, 0.3]
      - `fall_path_custom` → custom offset mapped by trajectory_id
      - otherwise → return original unchanged

    Args:
        origin (dict): One navigation episode item containing keys such as
            `trajectory_id`, `start_position`, and `reference_path`.

    Returns:
        dict: The amended item with updated start position and first reference path waypoint,
        or the original if no amendment rule matched.
    """
    trajectory_id = origin['trajectory_id']
    if trajectory_id in fall_path_z_0_3:
        amend_offset = [0, 0, 0.3]
    elif trajectory_id in fall_path_custom:
        amend_offset = fall_path_custom[trajectory_id]
    else:
        return origin
    origin['start_position'][0] = origin['start_position'][0] + amend_offset[0]
    origin['start_position'][1] = origin['start_position'][1] + amend_offset[1]
    origin['start_position'][2] = origin['start_position'][2] + amend_offset[2]
    origin['reference_path'][0][0] = origin['reference_path'][0][0] + amend_offset[0]
    origin['reference_path'][0][1] = origin['reference_path'][0][1] + amend_offset[1]
    origin['reference_path'][0][2] = origin['reference_path'][0][2] + amend_offset[2]
    return origin


def transform_rotation_z_90degrees(rotation):
    """
    Rotate a quaternion by 90 degrees (π/2 radians) around the Z axis.
    """
    z_rot_90 = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # 90 degrees = pi/2 radians
    w1, x1, y1, z1 = rotation
    w2, x2, y2, z2 = z_rot_90
    revised_rotation = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
    ]
    return revised_rotation


def has_stairs(item, height_threshold=0.3):
    """
    Determine if a navigation reference path contains stair-like height jumps when the
    instruction text includes the word 'stair'.

    The function checks the Z-height (3rd axis) differences between consecutive reference
    waypoints and flags True if any jump exceeds the threshold.

    Args:
        item (dict): Episode item containing `instruction.instruction_text` and `reference_path`.
        height_threshold (float, optional): Minimum absolute height delta to consider as stairs. Defaults to 0.3.

    Returns:
        bool: True if stairs are detected, False otherwise.
    """
    has_stairs = False
    if 'stair' in item['instruction']['instruction_text']:
        latest_height = item['reference_path'][0][-1]
        for index in range(1, len(item['reference_path'])):
            position = item['reference_path'][index]
            if abs(position[-1] - latest_height) >= height_threshold:
                has_stairs = True
                break
            else:
                latest_height = position[-1]
    return has_stairs


def different_height(item):
    """
    Check if multiple reference paths (or waypoints across paths) have significantly different
    heights (Z-axis), indicating non-flat terrain.

    Args:
        item (dict): Episode item containing a list of reference paths in `reference_path`.

    Returns:
        bool: True if any adjacent path segment has a height difference > 0.3, else False.
    """
    different_height = False
    paths = item['reference_path']
    for path_idx in range(len(paths) - 1):
        if abs(paths[path_idx + 1][2] - paths[path_idx][2]) > 0.3:
            different_height = True
            break
    return different_height


def load_data(
    dataset_root_dir, split, filter_same_trajectory=True, filter_stairs=True, dataset_type='mp3d', rank=0, world_size=1
):
    """
    Load a compressed navigation dataset split and organize episodes by scan/scene,
    with optional filtering rules for duplicate trajectories and stair terrain.

    Supported behaviors include:
      - Distributed slicing via `rank::world_size`
      - Scene grouping by `scan` (kujiale/grscene) or `scene_id` (mp3d)
      - Coordinate system remapping for mp3d (`x, z, y` → `[x, -y, z]`)
      - Start rotation quaternion remapping + 90° Z rotation
      - Filtering repeated `trajectory_id`
      - Filtering episodes containing stairs or uneven heights

    Args:
        dataset_root_dir (str): Root data directory containing the split folders.
        split (str): Dataset split name (folder & file prefix), e.g. "val_unseen".
        filter_same_trajectory (bool, optional): Remove episodes with duplicate trajectory_id. Defaults to True.
        filter_stairs (bool, optional): Remove episodes where stairs or large height variation are detected. Defaults to True.
        dataset_type (str, optional): Dataset source identifier, such as "mp3d", "kujiale", or "grscene". Defaults to "mp3d".
        rank (int, optional): Distributed process rank used for slicing episodes. Defaults to 0.
        world_size (int, optional): Number of distributed ranks used for striding. Defaults to 1.

    Returns:
        dict: Mapping from `scan` → List of filtered episode items for that scene.
    """
    with gzip.open(os.path.join(dataset_root_dir, split, f"{split}.json.gz"), 'rt', encoding='utf-8') as f:
        data = json.load(f)['episodes'][rank::world_size]

    if dataset_type in ['kujiale', 'grscene']:
        scenes = list(set([x['scan'] for x in data]))
    else:
        scenes = list(set([x['scene_id'] for x in data]))  # e.g. 'mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb'

    scenes.sort()
    new_data = {}
    for scene in scenes:
        if dataset_type in ['kujiale', 'grscene']:
            scene_data = [x for x in data if x['scan'] == scene]
            scan = scene
        else:
            scene_data = [x for x in data if x['scene_id'] == scene]
            scan = scene.split('/')[1]  # e.g. 'zsNo4HB9uLZ'
        new_scene_data = []
        for item in scene_data:
            new_item = copy.deepcopy(item)
            new_item['scan'] = scan
            new_item['original_start_position'] = item['start_position']
            new_item['original_start_rotation'] = item['start_rotation']
            if dataset_type == 'mp3d':
                x, z, y = item['start_position']
                new_item['start_position'] = [x, -y, z]
                r1, r2, r3, r4 = item['start_rotation']
                new_item['start_rotation'] = transform_rotation_z_90degrees([-r4, r1, r3, -r2])
                new_item['reference_path'] = [[x, -y, z] for x, z, y in item['reference_path']]
            new_scene_data.append(new_item)

        new_data[scan] = new_scene_data

    data = copy.deepcopy(new_data)
    new_data = defaultdict(list)

    # filter_same_trajectory
    if filter_same_trajectory:
        total_count = 0
        remaining_count = 0
        trajectory_list = []
        for scan, data_item in data.items():
            for item in data_item:
                total_count += 1
                if item['trajectory_id'] in trajectory_list:
                    continue
                remaining_count += 1
                trajectory_list.append(item['trajectory_id'])
                new_data[scan].append(item)
        log.info(f'[split:{split}]filter_same_trajectory remain: [ {remaining_count} / {total_count} ]')
        data = new_data
        new_data = defaultdict(list)

    if filter_stairs:
        total_count = 0
        remaining_count = 0
        for scan, data_item in data.items():
            for item in data_item:
                total_count += 1
                if has_stairs(item) or different_height(item):
                    continue
                remaining_count += 1
                new_data[scan].append(item)
        log.info(f'[split:{split}]filter_stairs remain: [ {remaining_count} / {total_count} ]')
        data = new_data

    return data
