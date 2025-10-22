from typing import Any, Dict

from internnav.configs.evaluator import EnvCfg, TaskCfg


class Env:
    """
    Base class of all environments.
    """

    envs = {}

    def __init__(self, env_config: EnvCfg, task_config: TaskCfg):
        self.env_config = env_config
        self.task_config = task_config

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def get_observation(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def register(cls, env_type: str):
        """
        Register a env class.
        """

        def decorator(env_class):
            if env_type in cls.envs:
                raise ValueError(f"Env {env_type} already registered.")
            cls.envs[env_type] = env_class

        return decorator

    @classmethod
    def init(cls, env_config: EnvCfg, task_config: TaskCfg):
        """
        Init a env instance from a config.
        """
        return cls.envs[env_config.env_type](env_config, task_config)
