# This file is the main file to run eval with different configs.

import sys

sys.path.append('.')

import argparse
import glob
import importlib
import importlib.util
import os

from internnav.evaluator import Evaluator
from internnav_benchmarks.internutopia.vln_default_config import get_config


# import all agents to register them
def auto_register_agents(agent_dir: str):
    # Get all Python files in the agents directory
    agent_modules = glob.glob(os.path.join(agent_dir, '*.py'))

    # Import each module to trigger the registration
    for module in agent_modules:
        if not module.endswith('__init__.py'):  # Avoid importing __init__.py itself
            module_name = os.path.basename(module)[:-3]  # Remove the .py extension
            importlib.import_module(
                f'internnav_baselines.agents.{module_name}'
            )  # Replace 'agents' with your module's package


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='scripts/eval/configs/h1_cma_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    return parser.parse_args()


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


def main():
    args = parse_args()
    evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')
    cfg = get_config(evaluator_cfg)
    print(cfg)

    # Call this function where you initialize your application
    auto_register_agents('internnav_baselines/agents')

    # TODO: register evaluator here

    print("--- Evaluator start ---")
    evaluator = Evaluator.init(cfg)
    evaluator.eval()


if __name__ == '__main__':
    main()
