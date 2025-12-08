import logging
import os
import time
from collections import deque
from dataclasses import dataclass

from internnav import PROJECT_ROOT_PATH

from .common_log_util import get_task_name

progress_logger_multi = logging.getLogger('progress_logger_multi')
progress_logger_multi.setLevel(logging.INFO)


class Queue:
    def __init__(self):
        self.items = deque()

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.popleft()
        else:
            raise IndexError('empty_queue')

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


@dataclass(order=True)
class TrajectoryInfo:
    trajectory_id: str
    start_time: time
    end_time: time
    end_step: int
    result: str


class ProgressInfo:
    def __init__(self, dataset_name, path_count):
        self.dataset_name = dataset_name
        self.path_count = path_count
        self.info_map = {}
        self.start = None
        self.end = None


PROGRESS = None
LAST_TRAJECTORY_ID = set()
INITED = False
FINISH_PATH_NUM = 0


def init(dataset_name, path_count):
    global PROGRESS
    global INITED
    PROGRESS = ProgressInfo(dataset_name, path_count)
    log_dir = f'{PROJECT_ROOT_PATH}/logs/{get_task_name()}/progress/'
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f'{log_dir}/{dataset_name}.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    progress_logger_multi.addHandler(file_handler)
    progress_logger_multi.disabled = False
    INITED = True


def last_log(last_trajectory_id, step_count=-1):
    global PROGRESS  # noqa: F824
    global FINISH_PATH_NUM
    FINISH_PATH_NUM += 1
    last_info = PROGRESS.info_map[last_trajectory_id]
    last_str = f'[{FINISH_PATH_NUM}/{PROGRESS.path_count}][step_index:{step_count}] finish: [trajectory_id:{last_info.trajectory_id}]'
    duration = round(last_info.end_time - last_info.start_time, 2)
    step_count = last_info.end_step
    fps = round((step_count / (duration + 1e-10)), 2)
    last_str = last_str + f'[duration:{duration} s]'
    last_str = last_str + f'[step_count:{step_count}]'
    last_str = last_str + f'[fps:{fps}]'
    last_str = last_str + f'[result:{last_info.result}]'
    progress_logger_multi.info(f'{last_str}')


def trace_start(trajectory_id):
    global INITED  # noqa: F824
    if not INITED:
        return
    global PROGRESS  # noqa: F824
    global LAST_TRAJECTORY_ID  # noqa: F824
    start_time = time.time()
    if PROGRESS.start is None:
        PROGRESS.start = start_time
    ti = TrajectoryInfo(
        trajectory_id=trajectory_id,
        start_time=start_time,
        end_time=None,
        end_step=None,
        result=None,
    )
    PROGRESS.info_map[trajectory_id] = ti
    progress_logger_multi.info(f'start sampling trajectory_id: {trajectory_id}')

    LAST_TRAJECTORY_ID.add(trajectory_id)


def trace_end(trajectory_id, step_count, result):
    global INITED  # noqa: F824
    if not INITED:
        return
    global PROGRESS  # noqa: F824
    end_time = time.time()
    ti = PROGRESS.info_map[trajectory_id]
    ti.end_time = end_time
    ti.end_step = step_count
    ti.result = result
    PROGRESS.info_map[trajectory_id] = ti
    last_log(trajectory_id, step_count)
    LAST_TRAJECTORY_ID.remove(trajectory_id)


def report():
    global PROGRESS  # noqa: F824
    global LAST_TRAJECTORY_ID  # noqa: F824
    result_map = {}
    step_count = 0
    for _, v in PROGRESS.info_map.items():
        result = v.result
        if result in result_map:
            result_map[result] = result_map[result] + 1
        else:
            result_map[result] = 1

        step_count += v.end_step

    PROGRESS.end = time.time()
    duration = round((PROGRESS.end - PROGRESS.start), 2)
    fps = round((step_count / (duration + 1e-10)), 2)
    progress_logger_multi.info(
        f'dataset:{PROGRESS.dataset_name} finished. [duration: {duration} s][step_count: {step_count}][fps :{fps}] result: {result_map}'
    )
