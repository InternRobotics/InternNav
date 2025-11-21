import logging
import os
from datetime import datetime

from internnav import PROJECT_ROOT_PATH

NAME = None

common_logger = logging.getLogger('common_logger')
common_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)
common_logger.addHandler(console_handler)
common_logger.disabled = False


def init(task_name='default'):
    global NAME
    NAME = task_name
    log_dir = f'{PROJECT_ROOT_PATH}/logs/{task_name}/common'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    file_name = f"log_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.getpid()}.log"
    file_handler = logging.FileHandler(f'{log_dir}/{file_name}')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    common_logger.addHandler(file_handler)
    common_logger.disabled = False


def get_task_name():
    global NAME  # noqa: F824
    return NAME
