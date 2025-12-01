"""
Test for VLN evaluators.

1. create temp one episode test split.
2. run and finish with ray (env_num=2, proc_num=8).
3. rerun to test the "no more episodes" case.
4. delete the sample_episodes after testing.
"""

'''
Test the evaluator eval logic with ray, set proc_num = 4.
The main progress:
    Init => warm up => one action
'''

import shutil
import sys
import time
from enum import Enum
from pathlib import Path

from test_server import start_server, stop_server

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)
from internnav.configs.evaluator.vln_default_config import get_config
from internnav.evaluator import Evaluator

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8087,
        model_name='rdp',
        ckpt_path='checkpoints/r2r/fine_tuned/rdp',
        model_settings={},
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': True,  # display option: set to False will open isaac-sim interactive window
        },
    ),
    task=TaskCfg(
        task_name='test_evaluation',
        task_settings={
            'env_num': 1,
            'use_distributed': False,  # Ray distributed framework
            'proc_num': 4,
        },
        scene=SceneCfg(
            scene_type='mp3d',
            scene_data_dir='data/scene_data/mp3d_pe',
        ),
        robot_name='h1',
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_vln_pointcloud.usd',
        camera_resolution=[256, 256],  # (W,H)
        camera_prim_path='torso_link/h1_pano_camera_0',
    ),
    dataset=EvalDatasetCfg(
        dataset_type="mp3d",
        dataset_settings={
            'base_data_dir': 'data/vln_pe/raw_data/r2r',
            'split_data_types': ['function_test'],
            'filter_stairs': False,
        },
    ),
    eval_type='vln_distributed',
    eval_settings={
        'save_to_json': False,
        'vis_output': False,
        'use_agent_server': True,
    },  # save result to video under logs/
)


class runner_status_code(Enum):
    NORMAL = 0
    WARM_UP = 1
    NOT_RESET = 3
    TERMINATED = 2
    STOP = 4


def _safe_remove(p: Path):
    try:
        if p.is_symlink() or p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        # Swallow errors to keep tests robust; log if you prefer
        pass


def cleanup_task_dirs(eval_cfg):
    """
    Remove data/sample_episodes/<task_name> and logs/<task_name>
    before the test, and again after the test (teardown).
    """
    task_name = eval_cfg.task.task_name

    data_task_dir = Path("data") / "sample_episodes" / task_name
    logs_task_dir = Path("logs") / task_name
    print(f'"data": {data_task_dir}, "logs": {logs_task_dir}')

    # pre-clean
    _safe_remove(data_task_dir)
    _safe_remove(logs_task_dir)
    print("Cleaned up test directories.")


def main():
    cfg = get_config(eval_cfg)
    evaluator = Evaluator.init(cfg)
    evaluator.eval()


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
        start_evaluator()

    except Exception as e:
        print(f'exception is {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)

    except SystemExit as e:
        print(f"Caught SystemExit from env.close(): code={e.code}", flush=True)

    finally:
        # shut down server
        stop_server(proc)
        # clean up task dirs
        cleanup_task_dirs(eval_cfg)
