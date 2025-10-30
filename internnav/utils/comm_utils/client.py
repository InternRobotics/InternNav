import base64
import pickle
from typing import Any, Dict, List, Optional

import requests

from internnav.configs.agent import AgentCfg, InitRequest, ResetRequest, StepRequest


def serialize_obs(obs):
    serialized = pickle.dumps(obs)
    encoded = base64.b64encode(serialized).decode('utf-8')
    return encoded


class AgentClient:
    """
    Client class for Agent service.
    """

    def __init__(self, config: AgentCfg):
        self.base_url = f'http://{config.server_host}:{config.server_port}'
        self.agent_name = self._initialize_agent(config)

    def _initialize_agent(self, config: AgentCfg) -> str:
        request_data = InitRequest(agent_config=config).model_dump(mode='json')

        response = requests.post(
            url=f'{self.base_url}/agent/init',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        return response.json()['agent_name']

    def step(self, obs: List[Dict[str, Any]]) -> List[List[int]]:
        request_data = StepRequest(observation=serialize_obs(obs)).model_dump(mode='json')

        response = requests.post(
            url=f'{self.base_url}/agent/{self.agent_name}/step',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        return response.json()['action']

    def reset(self, reset_index: Optional[List] = None) -> None:
        response = requests.post(
            url=f'{self.base_url}/agent/{self.agent_name}/reset',
            json=ResetRequest(reset_index=reset_index).model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()
