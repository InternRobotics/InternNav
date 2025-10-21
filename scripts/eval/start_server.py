#!/usr/bin/env python
import sys

sys.path.append('.')

import argparse
import glob
import importlib
import importlib.util
import os
import sys

from internnav.utils.comm_utils.server import AgentServer


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


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


if __name__ == '__main__':
    print("Starting Agent Server...")

    print("Registering agents...")
    auto_register_agents('internnav_baselines/agents')

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
