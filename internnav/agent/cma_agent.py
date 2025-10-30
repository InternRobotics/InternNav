import time

import numpy as np
import torch
from gym import spaces

from internnav.agent.base import Agent
from internnav.agent.utils.common import batch_obs, set_seed_model
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model import get_config, get_policy


@Agent.register('cma')
class CmaAgent(Agent):
    observation_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(256, 256, 1),
        dtype=np.float32,
    )

    def __init__(self, agent_config: AgentCfg):

        super().__init__(agent_config)
        self._model_settings = ModelCfg(**agent_config.model_settings)
        model_settings = self._model_settings
        set_seed_model(0)
        env_num = getattr(self._model_settings, 'env_num', 1)
        proc_num = getattr(self._model_settings, 'proc_num', 1)
        self.device = torch.device('cuda', 0)
        policy = get_policy(model_settings.policy_name)
        self.policy = policy.from_pretrained(
            agent_config.ckpt_path,
            config=get_config(self._model_settings.policy_name)(model_cfg={'model': self._model_settings.model_dump()}),
        ).to(self.device)

        # step required
        self._env_nums = env_num
        self._proc_num = proc_num
        self._rnn_states = torch.zeros(
            self._env_nums * self._proc_num,
            self.policy.num_recurrent_layers,
            self._model_settings.state_encoder.hidden_size,
            device=self.device,
        )
        self._prev_actions = torch.zeros(
            self._env_nums * self._proc_num,
            1,
            device=self.device,
            dtype=torch.long,
        )
        self._not_done_masks = torch.tensor([0 * self._env_nums * self._proc_num], device=self.device).to(torch.bool)

    def reset(self, reset_ls=None):
        if reset_ls is not None and len(reset_ls) > 0:
            print(f'CmaPolicyAgent{reset_ls} reset')

        if reset_ls is None:
            self._rnn_states = torch.zeros(
                self._env_nums * self._proc_num,
                self.policy.num_recurrent_layers,
                self._model_settings.state_encoder.hidden_size,
                device=self.device,
            )
            self._prev_actions = torch.zeros(
                self._env_nums * self._proc_num,
                1,
                device=self.device,
                dtype=torch.long,
            )
            self._not_done_masks = torch.zeros(
                self._env_nums * self._proc_num,
                1,
                device=self.device,
                dtype=torch.bool,
            )

        elif len(reset_ls) > 0:
            self._rnn_states.index_fill_(dim=0, index=torch.tensor(reset_ls).to(self.device), value=0)
            self._prev_actions.index_fill_(dim=0, index=torch.tensor(reset_ls).to(self.device), value=0)
            self._not_done_masks.index_fill_(
                dim=0,
                index=torch.tensor(reset_ls).to(self.device),
                value=False,
            )

    def inference(self, obs):
        start = time.time()

        # process change to here
        for ob in obs:
            ob['instruction'] = ob['instruction_tokens']
            ob.pop('instruction_tokens', None)
            instr = torch.tensor(ob['instruction'])
            ob['instruction'] = torch.nn.functional.pad(instr, (0, 200 - instr.shape[0]), 'constant', 0)
        obs = batch_obs(obs, device=self.device)

        # need to change
        batch = {
            'mode': 'inference',
            'observations': obs,
            'rnn_states': self._rnn_states,
            'prev_actions': self._prev_actions,
            'masks': self._not_done_masks,
        }

        with torch.no_grad():
            actions, self._rnn_states, _ = self.policy.forward(batch)
        # 确保 actions 的形状与 _prev_actions 匹配
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)  # 从 [2] 变为 [1, 2]
            print(f'actions: {actions}\n prev_actions: {self._prev_actions}')
        self._prev_actions.copy_(actions)
        self._not_done_masks = torch.ones(
            self._env_nums * self._proc_num,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        end = time.time()
        print(f'CmaAgent step time: {round(end-start, 4)}s')
        return actions.cpu().numpy().tolist()

    def step(self, obs):
        print('CmaPolicyAgent step')
        start = time.time()
        action = self.inference(obs)
        end = time.time()
        print(f'Time: {round(end-start, 4)}s')

        # convert from [[x],[y]] to [{'action': [x],'ideal_flag':True}, {'action': [y],'ideal_flag':True}]
        actions = []
        for a in action:
            if not isinstance(a, list):
                a = [a]
            actions.append({'action': a, 'ideal_flag': True})
        return actions
