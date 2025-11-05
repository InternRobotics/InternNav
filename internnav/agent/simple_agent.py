import time
from typing import Any, Dict

import torch

from internnav.agent import Agent
from internnav.configs.agent import AgentCfg
from internnav.model import get_config, get_policy


class SimpleAgent(Agent):
    """
    agent template, override the functions for custom policy
    """

    def __init__(self, agent_config: AgentCfg):
        self.agent_config = agent_config
        self.device = torch.device('cuda', 0)

        # get policy by name
        policy = get_policy(agent_config.model_settings.policy_name)

        # load policy checkpoints
        self.policy = policy.from_pretrained(
            agent_config.ckpt_path,
            config=get_config(agent_config.model_settings.policy_name)(
                model_cfg={'model': agent_config.model_settings.model_dump()}
            ),
        ).to(self.device)

    def convert_input(self, obs):
        return obs

    def convert_output(self, action):
        return action

    def inference(self, input):
        return self.policy(input)

    def step(self, obs: Dict[str, Any]):
        print(f'{self.config.model_name} Agent step')
        start = time.time()

        # convert obs to model input
        obs = self.convert_input(obs)
        action = self.inference(obs)
        action = self.convert_output(action)

        end = time.time()
        print(f'time: {round(end-start, 4)}s')
        return action

    def reset(self):
        pass
