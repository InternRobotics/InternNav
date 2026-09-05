import sys

sys.path.append('.')
sys.path.append('./third_party/diffusion-policy')

import argparse
import importlib.util

from internnav.evaluator import Evaluator

# This file is the main file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='scripts/eval/configs/h1_rdp_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument(
        '--kv-cache-continuation',
        action='store_true',
        help='Reuse the generation KV cache for latent readout in Habitat dual-system evaluation (default: off).',
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
    if args.kv_cache_continuation:
        if evaluator_cfg.eval_type != 'habitat_vln' or evaluator_cfg.agent.model_settings.get('mode') != 'dual_system':
            raise ValueError('--kv-cache-continuation requires a Habitat dual_system configuration.')
        evaluator_cfg.agent.model_settings['kv_cache_continuation'] = True

    # fill in evaluator default config
    if evaluator_cfg.eval_type == 'vln_distributed':
        from internnav.configs.evaluator.vln_default_config import get_config

        evaluator_cfg = get_config(evaluator_cfg)

    # create evaluator based on sim backend and run eval
    evaluator = Evaluator.init(evaluator_cfg)
    evaluator.eval()


if __name__ == '__main__':
    main()
