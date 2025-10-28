from typing import Any, Dict

from internnav.configs.agent import AgentCfg


class Agent:
    agents = {}

    def __init__(self, config: AgentCfg):
        self.config = config

    def step(self, obs: Dict[str, Any]):
        raise NotImplementedError("This function is not implemented yet.")

    def reset(self):
        raise NotImplementedError("This function is not implemented yet.")

    @classmethod
    def register(cls, agent_type: str):
        """
        Register a agent class.
        """

        def decorator(agent_class):
            if agent_type in cls.agents:
                raise ValueError(f"Agent {agent_type} already registered.")
            cls.agents[agent_type] = agent_class

        return decorator

    @classmethod
    def init(cls, config: AgentCfg):
        """
        Init a agent instance from a config.
        """
        return cls.agents[config.model_name](config)
