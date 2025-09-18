from typing import Dict

import numpy as np
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.sensor.camera import ICamera
from internutopia.core.sensor.sensor import BaseSensor

from internnav.utils.common_log_util import common_logger as log

from ..configs.sensors.vln_camera import VLNCameraCfg


@BaseSensor.register('VLNCamera')
class VLNCamera(BaseSensor):
    """
    wrap of isaac sim's Camera class
    """

    def __init__(self, config: VLNCameraCfg, robot: BaseRobot, scene: IScene):
        super().__init__(config, robot, scene)
        self.config = config
        self._camera = None

    def get_data(self) -> Dict:
        output_data = {}
        output_data['rgba'] = self._camera.get_rgba()
        if output_data['rgba'].shape[0] != self.config.resolution[1]:
            output_data['rgba'] = np.random.randint(
                0, 256, (self.config.resolution[1], self.config.resolution[0], 4), dtype=np.uint8
            )
            log.error("rgba shape wrong, use random one!!!")
        output_data['depth'] = self._camera.get_distance_to_image_plane()
        if output_data['depth'].shape[0] != self.config.resolution[1]:
            output_data['depth'] = np.random.uniform(
                0, 256, size=(self.config.resolution[1], self.config.resolution[0])
            ).astype(np.float32)
            log.error("depth shape wrong, use random one!!!")
        return self._make_ordered(output_data)

    def post_reset(self):
        self.restore_sensor_info()

    def restore_sensor_info(self):
        self.cleanup()
        prim_path = self._robot.config.prim_path + '/' + self.config.prim_path
        _camera = ICamera.create(
            name=self.config.name,
            prim_path=prim_path,
            rgba=True,
            bounding_box_2d_tight=False,
            distance_to_image_plane=True,
            camera_params=False,
            resolution=self.config.resolution,
        )
        self._camera: ICamera = _camera

    def cleanup(self) -> None:
        if self._camera is not None:
            self._camera.cleanup()

    def set_world_pose(self, *args, **kwargs):
        self._camera.set_world_pose(*args, **kwargs)

    def get_world_pose(self):
        return self._camera.get_world_pose()
