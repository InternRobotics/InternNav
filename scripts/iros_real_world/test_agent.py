import numpy as np

from internnav.utils.comm_utils.client import AgentClient

fake_obs = {
    'rgb': np.zeros((256, 256, 3), dtype=np.uint8),
    'depth': np.zeros((256, 256), dtype=np.float32),
    'instruction': 'go to the red car',
}

agent = AgentClient(
    {
        'agent_type': 'cma',
        'ckpt_path': 'ckpt/cma_h1_100M.pth',
        'model_settings': {
            'policy_name': 'CmaPolicy',
            'model': 'CmaModel',
            'state_encoder': {'hidden_size': 512, 'env_num': 1, 'proc_num': 1},
            'action_encoder': {'action_dim': 4},
            'goal_encoder': {'goal_dim': 512},
            'fusion_encoder': {'fusion_dim': 512},
            'policy_head': {'hidden_size': 512},
        },
    }
)
print(agent.step(fake_obs))
