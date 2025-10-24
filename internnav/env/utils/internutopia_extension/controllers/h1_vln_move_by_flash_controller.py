import math
from typing import Any, Dict, List

import numpy as np
from internutopia.core.robot.articulation import ArticulationAction
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene

from ..configs.controllers.flash_controller import VlnMoveByFlashControllerCfg


@BaseController.register('VlnMoveByFlashController')
class VlnMoveByFlashController(BaseController):  # codespell:ignore
    """Discrete Controller, direct set robot world position to achieve teleport-type locomotion.
    a general controller adaptable to different type of robots.
    """

    def __init__(self, config: VlnMoveByFlashControllerCfg, robot: BaseRobot, scene: IScene) -> None:
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

    def get_new_position_and_rotation(self, robot_position, robot_rotation, action):
        """
        Calculate robot new state by previous state and action. The move should be based on the controller
        settings.
        Caution: the rotation need to reset pitch and roll to prevent robot falling. This may due to no
                    adjustment during the whole path and some rotation accumulated

        Args:
            robot_position (np.ndarray): Current world position of the robot, shape (3,), in [x, y, z] format.
            robot_rotation (np.ndarray): Current world orientation of the robot as a quaternion, shape (4,), in [x, y, z, w] format.
            action (int): Discrete action to apply:
                          - 0: no movement (stand still)
                          - 1: move forward
                          - 2: rotate left
                          - 3: rotate right

        Returns:
            Tuple[np.ndarray, np.ndarray]: The new robot position and rotation as (position, rotation),
                                           both in world frame.
        """
        from omni.isaac.core.utils.rotations import (
            euler_angles_to_quat,
            quat_to_euler_angles,
        )

        _, _, yaw = quat_to_euler_angles(robot_rotation)
        if action == 1:  # forward
            dx = self.forward_distance * math.cos(yaw)
            dy = self.forward_distance * math.sin(yaw)
            new_robot_position = robot_position + [dx, dy, 0]
            new_robot_rotation = robot_rotation
        elif action == 2:  # left
            new_robot_position = robot_position
            new_yaw = yaw + math.radians(self.rotation_angle)
            new_robot_rotation = euler_angles_to_quat(
                np.array([0.0, 0.0, new_yaw])
            )  # using 0 to prevent the robot from falling
        elif action == 3:  # right
            new_robot_position = robot_position
            new_yaw = yaw - math.radians(self.rotation_angle)
            new_robot_rotation = euler_angles_to_quat(np.array([0.0, 0.0, new_yaw]))
        else:
            new_robot_position = robot_position
            new_robot_rotation = robot_rotation

        return new_robot_position, new_robot_rotation

    def reset_robot_state(self, position, orientation):
        """Set robot state to the new position and orientation.

        Args:
            position, orientation: np.array, issac_robot.get_world_pose()
        """
        # robot = self.robot.isaac_robot
        robot = self.robot.articulation
        robot._articulation.set_world_pose(position=position, orientation=orientation)
        robot._articulation.set_world_velocity(np.zeros(6))
        robot._articulation.set_joint_velocities(np.zeros(len(robot.dof_names)))
        robot._articulation.set_joint_positions(np.zeros(len(robot.dof_names)))
        robot._articulation.set_joint_efforts(np.zeros(len(robot.dof_names)))

    def forward(self, action: int) -> ArticulationAction:
        """Teleport robot by position, orientation and action

        Args:
            action: int
                    0. discrete action (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            ArticulationAction: joint signals to apply (nothing).
        """
        # get robot new position
        # positions, orientations = self.robot.isaac_robot.get_world_pose()
        positions, orientations = self.robot.articulation.get_world_pose()
        new_robot_position, new_robot_rotation = self.get_new_position_and_rotation(positions, orientations, action)

        # set robot to new state
        self.reset_robot_state(new_robot_position, new_robot_rotation)

        # Dummy action to do nothing
        return ArticulationAction()

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
        return {
            'finished': True,
        }
