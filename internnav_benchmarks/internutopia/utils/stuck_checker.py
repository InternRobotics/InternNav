import math

import numpy as np


class StuckChecker:
    def __init__(self, offset, robot):
        self.offset = offset
        self.last_iter = 0
        position, rotation = robot.get_world_pose()
        self.agent_last_position = position
        self.agent_last_rotation = rotation

    def check_robot_stuck(
        self,
        robot_position,
        robot_rotation,
        cur_iter,
        max_iter=300,
        threshold=0.2,
        rotation_threshold=math.pi / 12,
    ):
        """Check if the robot is stuck"""
        robot_position = robot_position - self.offset
        if (cur_iter - self.last_iter) <= max_iter:
            return False
        from omni.isaac.core.utils.rotations import quat_to_euler_angles

        position_diff = np.linalg.norm(robot_position[:2] - self.agent_last_position[:2])
        rotation_diff = abs(quat_to_euler_angles(robot_rotation)[2] - quat_to_euler_angles(self.agent_last_rotation)[2])
        if position_diff < threshold and rotation_diff < rotation_threshold:
            return True
        else:
            self.position_diff = 0
            self.rotation_diff = 0
            self.last_iter = cur_iter
            self.agent_last_position = robot_position
            self.agent_last_rotation = robot_rotation
            return False
