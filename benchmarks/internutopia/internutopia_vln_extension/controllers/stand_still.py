from typing import List

import numpy as np
from internutopia.core.robot.articulation import ArticulationAction
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene

from ..configs.controllers.stand_still import StandStillControllerCfg


@BaseController.register('StandStillController')
class StandStillController(BaseController):
    """Stand Still Controller."""

    def __init__(self, config: StandStillControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        self.config = config
        self.forward_speed_base = 0
        self.rotation_speed_base = 0
        self.lateral_speed_base = 0

        super().__init__(config=config, robot=robot, scene=scene)

    def forward(self) -> ArticulationAction:
        forward_speed = 0
        lateral_speed = 0
        rotation_speed = 0

        return self.sub_controllers[0].forward(
            forward_speed=forward_speed,
            rotation_speed=rotation_speed,
            lateral_speed=lateral_speed,
        )

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Args:
            action (List | np.ndarray): 0-element 1d array.
        """
        assert len(action) == 0, 'action must be empty'
        return self.forward()

    def get_obs(self):
        return {
            'finished': True,
        }
