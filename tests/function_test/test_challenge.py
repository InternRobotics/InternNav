'''
Test the evaluator eval logic without model involve.
The main progress:
    Init => warm up => fake one action
'''

import importlib.util
import sys
import time

import numpy as np
from test_server import start_server, stop_server

from internnav.configs.evaluator.default_config import get_config
from internnav.evaluator import Evaluator
from internnav.utils import progress_log_multi_util


def main():
    from enum import Enum

    class runner_status_code(Enum):
        NORMAL = 0
        WARM_UP = 1
        NOT_RESET = 3
        TERMINATED = 2
        STOP = 4

    def load_eval_cfg(config_path, attr_name='eval_cfg'):
        spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["eval_config_module"] = config_module
        spec.loader.exec_module(config_module)
        return getattr(config_module, attr_name)

    evaluator_cfg = load_eval_cfg('scripts/eval/configs/challenge_cfg.py', attr_name='eval_cfg')
    cfg = get_config(evaluator_cfg)
    evaluator = Evaluator.init(cfg)

    print('--- VlnPeEvaluator start ---')
    obs, reset_info = evaluator.env.reset()
    for info in reset_info:
        if info is None:
            continue
        progress_log_multi_util.trace_start(
            trajectory_id=evaluator.now_path_key(info),
        )

    obs = evaluator.warm_up()
    evaluator.fake_obs = obs[0][evaluator.robot_name]
    action = [{evaluator.robot_name: {'stand_still': []}} for _ in range(evaluator.env_num * evaluator.proc_num)]
    obs = evaluator._obs_remove_robot_name(obs)
    evaluator.runner_status = np.full(
        (evaluator.env_num * evaluator.proc_num),
        runner_status_code.NORMAL,
        runner_status_code,
    )
    evaluator.runner_status[[info is None for info in reset_info]] = runner_status_code.TERMINATED

    while evaluator.env.is_running():
        obs, action = evaluator.get_action(obs, action)
        obs, terminated = evaluator.env_step(action)
        env_term, reset_info = evaluator.terminate_ops(obs, reset_info, terminated)
        break

    evaluator.env.close()


def start_evaluator():
    from multiprocessing import get_context

    ctx = get_context("spawn")  # Use 'spawn' to avoid issues on some platforms
    p = ctx.Process(target=main)
    p.start()
    p.join()
    assert p.exitcode == 0
    print("Evaluator process completed successfully.")


if __name__ == '__main__':
    try:
        proc = start_server()
        time.sleep(3)
        start_evaluator()

    except Exception as e:
        print(f'exception is {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)

    except SystemExit as e:
        print(f"Caught SystemExit from env.close(): code={e.code}", flush=True)

    finally:
        stop_server(proc)
