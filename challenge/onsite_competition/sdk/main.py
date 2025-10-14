import argparse
import importlib.util
import sys

from real_world_env import RealWorldEnv
from stream import app, start_env

from internnav.agent.utils.client import AgentClient
from internnav.configs.evaluator.default_config import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='scripts/eval/configs/h1_cma_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help='current instruction to follow',
    )
    parser.add_argument("--tag", type=str, help="tag for the run, saved by the tag name which is team-task-trail")
    return parser.parse_args()


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


# TODO add logging for each step, saved by the tag name which is team-task-trail
def main():
    args = parse_args()
    print("--- Loading config from:", args.config, "---")
    evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')
    cfg = get_config(evaluator_cfg)
    print(cfg)

    # initialize user agent
    agent = AgentClient(cfg.agent)

    # initialize real world env
    env = RealWorldEnv()

    # start stream
    start_env(env)
    app.run(host="0.0.0.0", port=8080, threaded=True)

    try:
        while True:
            # print("get observation...")
            # obs contains {rgb, depth, instruction}
            obs = env.get_observation()
            obs["instruction"] = args.instruction

            # print("agent step...")
            # action is a integer in [0, 3], agent return [{'action': [int], 'ideal_flag': bool}] (same to internvla_n1 agent)
            try:
                action = agent.step(obs)[0]['action'][0]
                print(f"agent step success, action is {action}")
            except Exception as e:
                print(f"agent step error {e}")
                continue

            # print("env step...")
            try:
                env.step(action)
                print("env step success")
            except Exception as e:
                print(f"env step error {e}")
                continue
    finally:
        env.close()
