import importlib.util
import sys

import numpy as np
from save_obs import load_obs_from_meta

from internnav.agent.utils.client import AgentClient
from internnav.configs.evaluator.default_config import get_config


def load_eval_cfg(config_path, attr_name='eval_cfg'):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, attr_name)


# test if agent behave normally with fake observation
def test_agent(cfg_path=None, obs=None):
    cfg = load_eval_cfg(cfg_path, attr_name='eval_cfg')
    cfg = get_config(cfg)

    agent = AgentClient(cfg.agent)
    for _ in range(10):
        action = agent.step([obs])[0]['action'][0]  # modify your agent to match the output format
        print(f"Action taken: {action}")
        assert action in [0, 1, 2, 3]


if __name__ == "__main__":
    # use your own path
    # cfg_path = '/root/InternNav/scripts/eval/configs/h1_rdp_cfg.py'
    cfg_path = '/root/InternNav/scripts/eval/configs/h1_internvla_n1_cfg.py'
    rs_meta_path = 'challenge/onsite_competition/captures/rs_meta.json'

    fake_obs_256 = {
        'rgb': np.zeros((256, 256, 3), dtype=np.uint8),
        'depth': np.zeros((256, 256), dtype=np.float32),
        'instruction': 'go to the red car',
    }
    fake_obs_640 = load_obs_from_meta(rs_meta_path)
    fake_obs_640['instruction'] = 'go to the red car'
    print(fake_obs_640['rgb'].shape, fake_obs_640['depth'].shape)

    sim_obs = {
        'rgb': np.load('challenge/onsite_competition/captures/sim_rgb.npy'),
        'depth': np.load('challenge/onsite_competition/captures/sim_depth.npy'),
    }
    print(sim_obs['rgb'].shape, sim_obs['depth'].shape)  # dtype (uint8 and float32) and value range
    # TODO: crop to 256,256, test with fake_obs_256

    # work in progress, baseline model will be updated soon
    # test_agent(cfg_path=cfg_path, obs=fake_obs_256)
    test_agent(cfg_path=cfg_path, obs=fake_obs_640)
    print("All tests passed!")
