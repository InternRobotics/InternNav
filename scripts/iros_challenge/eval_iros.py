import sys

sys.path.append('.')

import argparse
import importlib.util

from internnav.configs.evaluator.vln_default_config import get_config
from internnav.evaluator import Evaluator

# This file is the main file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='scripts/eval/configs/h1_cma_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument(
        "--default_config",
        type=str,
        default='scripts/eval/configs/challenge_mp3d_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs='?',
        const='',
        default=None,
        required=False,
    )
    return parser.parse_args()


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


def replace_cfg(evaluator_cfg, default_cfg, split):
    # agent and model settings
    default_cfg.agent = evaluator_cfg.agent

    # split setting
    if split:
        default_cfg.dataset.dataset_settings['split_data_types'] = [split]

    # camera settings
    if (
        evaluator_cfg.task.camera_resolution not in [[256, 256], [640, 480]]
        or evaluator_cfg.task.camera_prim_path != default_cfg.task.camera_prim_path
    ):
        raise ValueError(
            "Please use our provided camera usd `camera_prim_path='torso_link/h1_pano_camera_0'` as the RGB-D camera, the resolution can be `[640, 480]` or `[256, 256]`."
        )
    default_cfg.task.camera_resolution = evaluator_cfg.task.camera_resolution


def main():
    args = parse_args()
    evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')
    default_cfg = load_eval_cfg(args.default_config, attr_name='eval_cfg')
    replace_cfg(evaluator_cfg, default_cfg, args.split)
    cfg = get_config(default_cfg)
    print(cfg)
    evaluator = Evaluator.init(cfg)
    print(type(evaluator))
    evaluator.save_to_json = True
    evaluator.eval()


if __name__ == '__main__':
    main()
