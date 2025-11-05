import importlib.util
import sys

sys.path.append('.')
sys.path.append('./src/diffusion-policy/')


import numpy as np
from save_obs import load_obs_from_meta

# from internnav.configs.evaluator.vln_default_config import get_config
from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)
from internnav.utils.comm_utils.client import AgentClient


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


# test if agent behave normally with fake observation
def test_agent(cfg_path=None, obs=None):
    # cfg = load_eval_cfg(cfg_path, attr_name='eval_cfg')
    # cfg = get_config(cfg)
    cfg = EvalCfg(
        agent=AgentCfg(
            server_host='localhost',
            server_port=8087,
            model_name='internvla_n1',
            ckpt_path='',
            model_settings={
                'policy_name': "InternVLAN1_Policy",
                'state_encoder': None,
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
        ),
        env=EnvCfg(
            env_type='internutopia',
            env_settings={
                'use_fabric': False,
                'headless': True,
            },
        ),
        task=TaskCfg(
            task_name='cma_kujiale_eval',
            task_settings={
                'env_num': 2,
                'use_distributed': True,
                'proc_num': 4,
            },
            scene=SceneCfg(
                scene_type='kujiale',
                scene_data_dir='interiornav_data/scene_data',
            ),
            robot_name='h1',
            robot_usd_path='data/Embodiments/vln-pe/h1/h1_vln_pointcloud.usd',
            camera_resolution=[256, 256],  # (W,H)
            camera_prim_path='torso_link/h1_pano_camera_0',
        ),
        dataset=EvalDatasetCfg(
            dataset_type="kujiale",
            dataset_settings={
                'base_data_dir': 'interiornav_data/raw_data',
                'split_data_types': ['val_unseen', 'val_seen'],
                'filter_stairs': False,
            },
        ),
    )

    agent = AgentClient(cfg.agent)
    for _ in range(10):
        action = agent.step([obs])[0]['action'][0]  # modify your agent to match the output format
        print(f"Action taken: {action}")
        assert action in [0, 1, 2, 3]


if __name__ == "__main__":
    # use your own path
    # cfg_path = '/root/InternNav/scripts/eval/configs/h1_rdp_cfg.py'
    cfg_path = '/root/InternNav/scripts/eval/configs/h1_internvla_n1_cfg.py'
    rs_meta_path = 'scripts/iros_challenge/onsite_competition/captures/rs_meta.json'

    fake_obs_256 = {
        'rgb': np.zeros((256, 256, 3), dtype=np.uint8),
        'depth': np.zeros((256, 256), dtype=np.float32),
        'instruction': 'go to the red car',
    }
    fake_obs_640 = load_obs_from_meta(rs_meta_path)
    fake_obs_640['instruction'] = 'go to the red car'
    print(fake_obs_640['rgb'].shape, fake_obs_640['depth'].shape)

    sim_obs = {
        'rgb': np.load('scripts/iros_challenge/onsite_competition/captures/sim_rgb.npy'),
        'depth': np.load('scripts/iros_challenge/onsite_competition/captures/sim_depth.npy'),
    }
    print(sim_obs['rgb'].shape, sim_obs['depth'].shape)  # dtype (uint8 and float32) and value range
    # TODO: crop to 256,256, test with fake_obs_256

    # work in progress, baseline model will be updated soon
    # test_agent(cfg_path=cfg_path, obs=fake_obs_256)
    test_agent(cfg_path=cfg_path, obs=fake_obs_640)
    print("All tests passed!")
