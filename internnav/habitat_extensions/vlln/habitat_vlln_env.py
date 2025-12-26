from typing import Any, Dict, List, Optional

import numpy as np
import quaternion
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import HabitatEnv, base
from internnav.env.utils.dialog_mp3d import MP3DGTPerception


@base.Env.register('habitat_vlln')
class HabitatVllnEnv(HabitatEnv):
    """Habitat-based environment wrapper for VLLN-style tasks.

    Args:
        env_config (EnvCfg): Environment configuration.
        task_config (TaskCfg): Task configuration.
    """

    def __init__(self, env_config: EnvCfg, task_config: TaskCfg = None):

        super().__init__(env_config, task_config)
        self.config = env_config.env_settings['habitat_config']

        self.rank = env_config.env_settings.get('rank', 0)
        self.world_size = env_config.env_settings.get('world_size', 1)
        self._current_episode_index: int = 0
        self._last_obs: Optional[Dict[str, Any]] = None

        self.is_running = True
        self.output_path = env_config.env_settings.get('output_path', './output')

        agent_config = get_agent_config(self.config.habitat.simulator)
        self.min_depth = agent_config.sim_sensors.depth_sensor.min_depth
        self.max_depth = agent_config.sim_sensors.depth_sensor.max_depth
        self._camera_fov = np.deg2rad(agent_config.sim_sensors.depth_sensor.hfov)
        self._fx = self._fy = agent_config.sim_sensors.depth_sensor.width / (2 * np.tan(self._camera_fov / 2))
        self._camera_height = agent_config.sim_sensors.rgb_sensor.position[1]
        self.segmentation = MP3DGTPerception(self.max_depth, self.min_depth, self._fx, self._fy)

        # generate episodes
        self.episodes = self.generate_episodes()

    def reset(self):
        # no more episodes
        if not (0 <= self._current_episode_index < len(self.episodes)):
            self.is_running = False
            return

        # Manually set to next episode in habitat
        self._env.current_episode = self.episodes[self._current_episode_index]
        self._current_episode_index += 1

        # Habitat reset
        self._last_obs = self._env.reset()
        if self._last_obs and "instance" in self.task_config.task_name:
            self._last_obs['semantic'] = self.get_semantic(self._last_obs)
        return self._last_obs

    def step(self, action: List[Any]):
        obs = self._env.step(action)
        if "instance" in self.task_config.task_name:
            obs['semantic'] = self.get_semantic(obs)
        done = self._env.episode_over
        info = self._env.get_metrics()
        reward = info.get('reward', 0.0)
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
