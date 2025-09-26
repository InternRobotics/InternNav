from typing import Any, Dict

from internnav.configs.model.base_encoders import ModelCfg


class Model:
    models = {}

    def __init__(self, config: ModelCfg):
        self.config = config

    def forward(self, obs: Dict[str, Any]):
        pass

    def inference(self):
        pass

    @classmethod
    def register(cls, model_type: str):
        """
        Register a agent class.
        """

        def decorator(model_class):
            if model_type in cls.models:
                raise ValueError(f"Model {model_type} already registered.")
            cls.models[model_type] = model_class

        return decorator

    @classmethod
    def init(cls, config: ModelCfg):
        """
        Init a agent instance from a config.
        """
        return cls.models[config.model_name](config)
