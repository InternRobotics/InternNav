#!/usr/bin/env python
import sys

sys.path.append('.')
sys.path.append('./src/diffusion-policy')

import argparse
import importlib
import importlib.util
import sys

# Import for agent registry side effects — do not remove
from internnav.agent import Agent  # noqa: F401
from internnav.utils import AgentServer


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


if __name__ == '__main__':
    print("Starting Agent Server...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument('--port', type=int, default=8087)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    if args.config:
        eval_cfg = load_eval_cfg(args.config)
        args.port = eval_cfg.agent.server_port
    else:
        print(f"Warning: No config file provided, using port {args.port}")

    server = AgentServer(args.host, args.port)
    server.run()
