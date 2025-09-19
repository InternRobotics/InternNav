import numpy as np
from internutopia.core.config.robot import RobotCfg
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia_extension.robots.h1 import H1Robot


@BaseRobot.register('VLNH1Robot')
class VLNH1Robot(H1Robot):
    def __init__(self, config: RobotCfg, scene: IScene):
        super().__init__(config, scene)
        self.current_action = None

    def post_reset(self):
        super().post_reset()
        self._torso_link = self._rigid_body_map[self.config.prim_path + '/torso_link']
        self._imu_link = self._rigid_body_map[self.config.prim_path + '/imu_link']

    def apply_action(self, action: dict):
        import omni.isaac.core.utils.numpy.rotations as rot_utils

        self.current_action = action
        ret = super().apply_action(action)
        if 'topdown_camera_500' in self.sensors:
            orientation_quat = np.array([-0.70710678, 0.0, 0.0, 0.70710678])
            robot_pos = self.articulation.get_world_pose()[0]
            self.sensors['topdown_camera_500'].set_world_pose(
                [robot_pos[0], robot_pos[1], robot_pos[2] + 0.75],
                orientation_quat,
            )

        if 'topdown_camera_50' in self.sensors:
            orientation_quat = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
            robot_pos = self.articulation.get_world_pose()[0]
            self.sensors['topdown_camera_50']._camera.set_pose(
                [robot_pos[0], robot_pos[1], robot_pos[2] + 0.75],
                orientation_quat,
            )

        return ret
