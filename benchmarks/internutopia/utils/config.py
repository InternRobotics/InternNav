import numpy as np
import yaml
from pydantic import BaseModel

from internnav import PROJECT_ROOT_PATH
from internnav.configs.evaluator import TaskCfg
from internnav.utils.common_log_util import common_logger as log


class Config(BaseModel, extra='allow'):
    pass


def parse_task_config(config_file) -> TaskCfg:
    config_dict = parse_config(config_file)
    return TaskCfg(**config_dict)


def parse_grutopia_config(config_file) -> Config:
    config_dict = parse_config(config_file)
    return Config(**config_dict)


def get_robot_name(grutopia_config: Config):
    robot_name = grutopia_config.task_config.episodes[0].robots[0].name
    return robot_name


def get_robot_offset(grutopia_config: Config):
    robot_name = get_robot_name(grutopia_config)
    if robot_name == 'h1':
        robot_offset = np.array([0.0, 0.0, 1.05])
    else:
        log.info(f'unknow robot_name:{robot_name}')
        robot_offset = np.array([0.0, 0.0, 0])
    return robot_offset


def parse_config(config_file) -> dict:
    if not config_file.endswith('.yaml') or config_file.endswith('.yml'):
        log.error('config file not end with .yaml or .yml')
        raise FileNotFoundError('runtime file not end with .yaml or .yml')
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f.read(), yaml.FullLoader)
    return config_dict


def get_lmdb_path(name):
    return PROJECT_ROOT_PATH + f'/data/sample_episodes/{name}'


def get_lmdb_prefix(run_type):
    return f'{run_type}_rank'
