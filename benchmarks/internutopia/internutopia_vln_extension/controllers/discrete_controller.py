from typing import Any, Dict, List

import numpy as np
from internutopia.core.robot.articulation import ArticulationAction
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene

from ..configs.controllers.discrete_controller import DiscreteControllerCfg


@BaseController.register('DiscreteController')
class DiscreteController(BaseController):  # codespell:ignore
    """Discrete Controller (Similar to Habitat)."""  # codespell:ignore

    def __init__(self, config: DiscreteControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        self._user_config = None
        self.current_steps = 0
        self.steps_per_action = config.steps_per_action if config.steps_per_action is not None else 200

        self.forward_distance = config.forward_distance if config.forward_distance is not None else 0.25
        self.rotation_angle = config.rotation_angle if config.rotation_angle is not None else 15.0  # in degrees
        self.physics_frequency = config.physics_frequency if config.physics_frequency is not None else 240

        # self.forward_speed = config.forward_speed if config.forward_speed is not None else 0.25
        # self.rotation_speed = config.rotation_speed if config.rotation_speed is not None else 1.0
        self.forward_speed = self.forward_distance / self.steps_per_action * self.physics_frequency
        self.rotation_speed = np.deg2rad(
            self.rotation_angle / self.steps_per_action * self.physics_frequency
        )  # 200 is the physics_dt

        self.current_action = None

        super().__init__(config=config, robot=robot, scene=scene)

    def forward(self, action: int) -> ArticulationAction:
        if self.current_action != action:
            self.current_action = action
            self.current_steps = 0

        self.current_steps += 1

        # Define actions:
        # 0: stop
        # 1: move forward
        # 2: turn left
        # 3: turn right
        if action == 0:
            return self.sub_controllers[0].forward(
                forward_speed=0.0,
                rotation_speed=0.0,
            )
        elif action == 1:
            return self.sub_controllers[0].forward(
                forward_speed=self.forward_speed,
                rotation_speed=0.0,
            )
        elif action == 2:
            return self.sub_controllers[0].forward(
                forward_speed=0.0,
                rotation_speed=self.rotation_speed,
            )
        elif action == 3:
            return self.sub_controllers[0].forward(
                forward_speed=0.0,
                rotation_speed=-self.rotation_speed,
            )
        else:
            raise ValueError(f'Invalid action: {action}')

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """Convert input action (in 1d array format) to joint signals to apply.

        Args:
            action (List | np.ndarray): 1-element 1d array containing
              0. discrete action (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            ArticulationAction: joint signals to apply.
        """
        assert len(action) == 1, 'action must contain 1 element'
        return self.forward(action=int(action[0]))

    def get_obs(self) -> Dict[str, Any]:
        finished = False
        if self.current_steps >= self.steps_per_action:
            finished = True
            self.current_action = None  # reset action

        return {
            'current_action': self.current_action,
            'current_steps': self.current_steps,
            'finished': finished,
        }
