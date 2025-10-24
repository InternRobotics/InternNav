import argparse
import importlib.util
import sys

from real_world_env import RealWorldEnv
from stream import run, set_instruction

from internnav.utils.comm_utils.client import AgentClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='challenge/onsite_competition/configs/h1_internvla_n1_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default='Go straight and pass the sofa and turn right into the hallway. Keep walking down, pass the kitchen and the bathroom, then enter the study room at the far end on the right with a desk, stop next to the white shelf.',
        help='current instruction to follow',
    )
    parser.add_argument("--uninteractive_mode", action='store_true', help="whether to confirm each step")
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


def get_instruction() -> int:
    try:
        import json

        instruction_lst = json.load(open("challenge/onsite_competition/instructions.json"))
        print("Available instructions:")
        for i, item in enumerate(instruction_lst):
            print(f"{i}: {item['instruction_title']}")
        answer = input("input instruction id: ").strip().lower()
        answer = int(answer)
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        sys.exit()
    return instruction_lst[answer]['instruction'][0]


def action_to_word(action: int) -> str:
    action = max(0, min(3, action))
    wl = ["stand still", "move forward", "turn left", "turn right"]
    return wl[action]


def main():
    args = parse_args()
    print("--- Loading config from:", args.config, "---")
    cfg = load_eval_cfg(args.config, attr_name='eval_cfg')
    agent_cfg = cfg.agent

    # initialize user's agent
    agent = AgentClient(agent_cfg)

    # initialize real world env
    env = RealWorldEnv(fps=30, duration=0.1, distance=0.3, angle=15, move_speed=0.5, turn_speed=0.5)
    env.reverse()  # reverse move direction if using a rear camera
    env.step(0)
    obs = env.get_observation()

    # start stream
    print("--- start running steam app ---")
    run(env=env)

    while True:
        instruction = get_instruction()
        print("\nNew instruction:", instruction)
        set_instruction(instruction)

        while True:
            # print("get observation...")
            # obs contains {rgb, depth, instruction}
            obs = env.get_observation()
            # print(obs)
            obs["instruction"] = instruction

            print("agent step...")
            # action is a integer in [0, 3], agent return [{'action': [int], 'ideal_flag': bool}] (same to internvla_n1 agent)
            action = agent.step([obs])[0]['action'][0]
            print("agent step success, action:", action)

            if args.uninteractive_mode or confirm(f"Execute this action [{action_to_word(action)}]?"):
                print("env step...")
                env.step(action)
                print("env step success")
            else:
                print("Stop requested. Exiting loop.")
                print("agent reset...")
                agent.reset()
                break


if __name__ == "__main__":
    main()
