from typing import Any, List

import numpy as np
import quaternion
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import HabitatEnv, base
from internnav.env.utils.dialog_mp3d import MP3DGTPerception


@base.Env.register('habitat_vlln')
class HabitatVllnEnv(HabitatEnv):
    def __init__(self, env_config: EnvCfg, task_config: TaskCfg = None):
        super().__init__(env_config, task_config)

        agent_config = get_agent_config(self.config.habitat.simulator)
        self.min_depth = agent_config.sim_sensors.depth_sensor.min_depth
        self.max_depth = agent_config.sim_sensors.depth_sensor.max_depth
        self._camera_fov = np.deg2rad(agent_config.sim_sensors.depth_sensor.hfov)
        self._fx = self._fy = agent_config.sim_sensors.depth_sensor.width / (2 * np.tan(self._camera_fov / 2))
        self._camera_height = agent_config.sim_sensors.rgb_sensor.position[1]
        self.segmentation = MP3DGTPerception(self.max_depth, self.min_depth, self._fx, self._fy)

    def reset(self):
        self._last_obs = super().reset()
        if self.task_config and "instance" in self.task_config.task_name:
            self._last_obs['semantic'] = self.get_semantic(self._last_obs)
        return self._last_obs

    def step(self, action: List[Any]):
        obs, reward, done, info = super().step(action)
        if self.task_config and "instance" in self.task_config.task_name:
            obs['semantic'] = self.get_semantic(obs)
        return obs, reward, done, info

    def get_tf_episodic_to_global(self):
        agent_state = self._env.sim.get_agent_state()
        rotation = agent_state.rotation
        translation = agent_state.position
        rotation_matrix = quaternion.as_rotation_matrix(rotation)
        tf_episodic_to_global = np.eye(4)
        tf_episodic_to_global[:3, :3] = rotation_matrix
        tf_episodic_to_global[:3, 3] = translation
        return tf_episodic_to_global

    def get_semantic(self, obs: dict):
        targets = [
            self.get_current_episode().goals[idx].bbox
            for idx, _ in enumerate(self.get_current_episode().instruction.instance_id)
        ]
        targets = np.array(
            [
                [target[0], min(-target[2], -target[5]), target[1], target[3], max(-target[5], -target[2]), target[4]]
                for target in targets
            ]
        )
        depth = filter_depth(obs["depth"].reshape(obs["depth"].shape[:2]), blur_type=None)
        tf_camera_to_global = self.get_tf_episodic_to_global()
        tf_camera_to_global[1, 3] = self._camera_height + self._env.sim.get_agent_state().position[1]
        tf_camera_to_ply = np.dot(
            np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), tf_camera_to_global
        )
        semantic = self.segmentation.predict(depth, targets, tf_camera_to_ply)
        return semantic
