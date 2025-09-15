import sys

sys.path.append('.')

import argparse
import importlib.util

from internnav.configs.evaluator.default_config import get_config
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
        default='scripts/eval/configs/h1_cma_cfg.py',
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
    default_cfg.agent = evaluator_cfg.agent
    if split:
        default_cfg.dataset.dataset_settings['split_data_types'] = [split]


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
