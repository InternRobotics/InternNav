from typing import Optional

from internutopia.core.config.robot import ControllerCfg


class DiscreteControllerCfg(ControllerCfg):
    type: Optional[str] = 'DiscreteController'
    steps_per_action: int = 50
    forward_distance: float = 0.25
    rotation_angle: float = 15.0
    physics_frequency: int = 240
