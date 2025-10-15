import argparse
import importlib.util
import sys

from real_world_env import RealWorldEnv
from stream import run

from internnav.configs.agent import AgentCfg
from internnav.configs.model import internvla_n1_cfg
from internnav.utils.comm_utils.client import AgentClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='scripts/eval/configs/h1_internvla_n1_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default='go to the red sofa',
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


def confirm(msg: str) -> bool:
    """
    Ask user to confirm. Return True if user types 'y' (case-insensitive),
    False for anything else (including empty input).
    """
    try:
        answer = input(f"{msg} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return False
    return answer in ("", "y")


def main():
    args = parse_args()
    print("--- Loading config from:", args.config, "---")
    # evaluator_cfg = load_eval_cfg(args.config, attr_name='eval_cfg')
    # cfg = get_config(evaluator_cfg)
    # print(cfg)
    agent_cfg = AgentCfg(
        # server_host="192.168.0.2",
        server_host="192.168.1.63",
        server_port=8087,
        model_name='internvla_n1',
        ckpt_path='',
        model_settings={
            'env_num': 1,
            'sim_num': 1,
            'model_path': "checkpoints/InternVLA-N1",
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
            'resize_w': 384,
            'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            'device': 'cuda:0',
            'predict_step_nums': 32,
            'continuous_traj': True,
            # debug
            'vis_debug': True,  # If vis_debug=True, you can get visualization results
            'vis_debug_path': './logs/test/vis_debug',
        },
    )
    model_settings = internvla_n1_cfg.model_dump()

    model_settings.update(agent_cfg.model_settings)
    agent_cfg.model_settings = model_settings

    # initialize user agentc
    agent = AgentClient(agent_cfg)

    # initialize real world env
    env = RealWorldEnv(fps=30, duration=0.5, distance=0.0, angle=0)
    env.step(0)
    obs = env.get_observation()

    # start stream
    print("--- start running steam app ---")
    run(env=env)

    while True:
        # print("get observation...")
        # obs contains {rgb, depth, instruction}
        obs = env.get_observation()
        # print(obs)
        obs["instruction"] = args.instruction

        print("agent step...")
        # action is a integer in [0, 3], agent return [{'action': [int], 'ideal_flag': bool}] (same to internvla_n1 agent)
        action = agent.step([obs])[0]['action'][0]
        print("agent step success, action:", action)

        if confirm(f"Execute this action {action}?"):
            print("env step...")
            env.step(action)
            print("env step success")
        else:
            print("Stop requested. Exiting loop.")
            break


if __name__ == "__main__":
    main()
