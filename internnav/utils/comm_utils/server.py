#!/usr/bin/env python
import base64
import pickle
from typing import Dict

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, status

from internnav.agent.base import Agent
from internnav.configs.agent import InitRequest, ResetRequest, StepRequest


class AgentServer:
    """
    Server class for Agent service.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.app = FastAPI(title='Agent Service')
        self.agent_instances: Dict[str, Agent] = {}
        self._router = APIRouter(prefix='/agent')
        self._register_routes()
        self.app.include_router(self._router)

    def _register_routes(self):
        route_config = [
            ('/init', self.init_agent, ['POST'], status.HTTP_201_CREATED),
            ('/{agent_name}/step', self.step_agent, ['POST'], None),
            ('/{agent_name}/reset', self.reset_agent, ['POST'], None),
            # TODO: Add stop server route
        ]

        for path, handler, methods, status_code in route_config:
            self._router.add_api_route(
                path=path,
                endpoint=handler,
                methods=methods,
                status_code=status_code,
            )

    async def init_agent(self, request: InitRequest):
        agent_config = request.agent_config
        agent = Agent.init(agent_config)
        agent_name = agent_config.model_name
        self.agent_instances[agent_name] = agent
        return {'status': 'success', 'agent_name': agent_name}

    async def step_agent(self, agent_name: str, request: StepRequest):
        self._validate_agent_exists(agent_name)
        agent = self.agent_instances[agent_name]

        def transfer(obs):
            obs = base64.b64decode(obs)
            obs = pickle.loads(obs)
            return obs

        obs = transfer(request.observation)
        action = agent.step(obs)
        return {'action': action}

    async def reset_agent(self, agent_name: str, request: ResetRequest):
        self._validate_agent_exists(agent_name)
        self.agent_instances[agent_name].reset(getattr(request, 'reset_index', None))
        return {'status': 'success'}

    def _validate_agent_exists(self, agent_name: str):
        if agent_name not in self.agent_instances:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Agent not found')

    def run(self, reload=False):
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            reload_dirs=['./internnav/agent/', './internnav/model/'],
        )
