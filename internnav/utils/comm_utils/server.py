#!/usr/bin/env python
import base64
import multiprocessing
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


def start_server(host='localhost', port=8087, dist=False):
    """
    start a server in the backgrouond process

    Args:
        host
        port

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_server if not dist else _run_server_dist, args=(host, port))
    p.daemon = True
    p.start()
    print(f"Server started on {host}:{port} (pid={p.pid})")
    return p


def _run_server_dist(host='localhost', port=8087):
    import torch

    from internnav.utils.dist import get_rank

    device_idx = get_rank()
    torch.cuda.set_device(device_idx)
    print(f"Server using GPU {device_idx}")
    server = AgentServer(host, port)
    server.run()


def _run_server(host='localhost', port=8087):
    server = AgentServer(host, port)
    server.run()
