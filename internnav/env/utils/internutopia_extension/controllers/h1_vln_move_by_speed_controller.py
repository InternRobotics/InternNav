from typing import List

import internutopia.core.util.gym as gymutil
import internutopia.core.util.math as math_utils
import numpy as np
import torch
import torch.nn.functional as F
from internutopia.core.robot.articulation import ArticulationAction
from internutopia.core.robot.articulation_subset import ArticulationSubset
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.rigid_body import IRigidBody
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.sensor.sensor import BaseSensor
from internutopia_extension.configs.controllers import H1MoveBySpeedControllerCfg

from .math import quat_apply_yaw


def init_height_points():
    """Returns points at which the height measurements are sampled (in base frame)

    Returns:
        [torch.Tensor]: Tensor of shape (self.num_height_points, 3)
    """
    measured_points_x = [
        -0.55,
        -0.45,
        -0.35,
        -0.25,
        -0.15,
        -0.05,
        0.05,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
    ]
    measured_points_y = [-0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35]
    y = torch.tensor(measured_points_y, device='cpu', requires_grad=False)
    x = torch.tensor(measured_points_x, device='cpu', requires_grad=False)
    grid_x, grid_y = torch.meshgrid(x, y)

    num_height_points = grid_x.numel()
    points = torch.zeros(num_height_points, 3, device='cpu', requires_grad=False)
    points[:, 0] = grid_x.flatten()
    points[:, 1] = grid_y.flatten()

    return points


class StaticHeightSamples:
    """Manually built height map."""

    def __init__(self, resolution: float = 0.1):
        points = torch.zeros(100, 100)

        points[55:65, 35:65] = 0.2
        points[65:69, 35:65] = 0.4
        points[69:73, 35:65] = 0.6
        points[73:77, 35:65] = 0.8
        points[77:100, 35:65] = 1.0
        self.height_map = points
        self.resolution = resolution
        self.x_min = -50
        self.y_min = -50
        self.x_max = 50
        self.y_max = 50

    def get_heights(self, points: torch.Tensor) -> torch.Tensor:
        px = (points[:, 0] / self.resolution).long()
        py = (points[:, 1] / self.resolution).long()
        indices_x = px - self.x_min
        indices_y = py - self.y_min

        indices_x = torch.clip(indices_x, 0, self.x_max - self.x_min)
        indices_y = torch.clip(indices_y, 0, self.y_max - self.y_min)

        return self.height_map[indices_x, indices_y]


class DynamicHeightSamples:
    def __init__(self, resolution: float = 0.1):
        """Initialize height samples.
        Args:
            resolution: resolution of samples.
        """
        self.x_min: int = None
        self.x_max: int = None
        self.y_min: int = None
        self.y_max: int = None
        self.height_map: torch.Tensor = None
        self.resolution = resolution

    def adjust_range(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        padding: float = 0.0,
    ):
        """Adjust the range of x and y of the height map.
        Args:
            x_min, x_max: new range of x.
            y_min, y_max: new range of y.
            padding: padding value to newly added points.
        """
        # Just set if no range has been set yet.
        if self.x_min is None:
            self.x_min, self.x_max = x_min, x_max
            self.y_min, self.y_max = y_min, y_max
            self.height_map = torch.full((x_max - x_min + 1, y_max - y_min + 1), fill_value=padding)
            return

        # Check if x range needs to be expanded.
        if x_min < self.x_min or x_max > self.x_max:
            pad_left = max(0, self.x_min - x_min)
            pad_right = max(0, x_max - self.x_max)
            self.height_map = F.pad(self.height_map, (0, 0, pad_left, pad_right), value=padding)
            self.x_min = min(self.x_min, x_min)
            self.x_max = max(self.x_max, x_max)

        # Check if y range needs to be expanded.
        if y_min < self.y_min or y_max > self.y_max:
            pad_top = max(0, self.y_min - y_min)
            pad_bottom = max(0, y_max - self.y_max)
            self.height_map = F.pad(self.height_map, (pad_top, pad_bottom, 0, 0), value=padding)
            self.y_min = min(self.y_min, y_min)
            self.y_max = max(self.y_max, y_max)

    def set_heights(self, points: torch.Tensor, robot_pos: np.ndarray):
        """Set the height of the points in the map.
        Args:
            points: (N, 3) points from point cloud data.
            robot_pos: (x, y, z) robot base position, for data validation and padding.
        """

        # Filter points to a reasonable surrounding range.
        x_range_max = robot_pos[0] + 3.0
        x_range_min = robot_pos[0] - 3.0
        y_range_max = robot_pos[1] + 3.0
        y_range_min = robot_pos[1] - 3.0
        mask = (
            (points[:, 0] < x_range_max)
            & (points[:, 0] > x_range_min)
            & (points[:, 1] < y_range_max)
            & (points[:, 1] > y_range_min)
        )
        # Disgard points from own body.
        inner_x_range_max = robot_pos[0] + 0.5
        inner_x_range_min = robot_pos[0] - 0.5
        inner_y_range_max = robot_pos[1] + 0.5
        inner_y_range_min = robot_pos[1] - 0.5
        body_mask = (
            (points[:, 0] > inner_x_range_min)
            & (points[:, 0] < inner_x_range_max)
            & (points[:, 1] < inner_y_range_max)
            & (points[:, 1] > inner_y_range_min)
        )
        mask = mask & ~body_mask
        filtered_points = points[mask]
        if filtered_points.numel() == 0:
            return

        px = (filtered_points[:, 0] / self.resolution).long()
        py = (filtered_points[:, 1] / self.resolution).long()
        min_x, max_x = torch.min(px).item(), torch.max(px).item()
        min_y, max_y = torch.min(py).item(), torch.max(py).item()

        # Adjust the range so all points fit in the height map.
        self.adjust_range(min_x, max_x, min_y, max_y, robot_pos[2])

        # Compute the indices.
        indices_x = px - self.x_min
        indices_y = py - self.y_min

        # Set the heights.
        self.height_map[indices_x, indices_y] = filtered_points[:, 2]

    def get_heights(self, points: torch.Tensor) -> torch.Tensor:
        """Get heights of a set of points.
        Args:
            points: (N, ≥2) shaped Tensor.
        Returns:
            heights: N-elements vector of heights.
        """
        if self.x_min is None:
            return torch.zeros(points.shape[0])

        px = (points[:, 0] / self.resolution).long()
        py = (points[:, 1] / self.resolution).long()
        indices_x = px - self.x_min
        indices_y = py - self.y_min

        indices_x = torch.clip(indices_x, 0, self.x_max - self.x_min)
        indices_y = torch.clip(indices_y, 0, self.y_max - self.y_min)

        return self.height_map[indices_x, indices_y]

    @property
    def shape(self):
        return self.height_map.shape


class RLPolicy:
    """RL policy for h1 locomotion."""

    def __init__(self, path: str):
        self.path = path
        return None

    def get_inference_policy(self, device: str = None):
        self.policy = torch.jit.load(self.path, map_location=device)
        self.policy.eval()
        return self.policy


@BaseController.register('VlnMoveBySpeedController')
class VlnMoveBySpeedController(BaseController):
    """Controller class converting locomotion speed control action to joint positions for H1 robot."""

    """
    joint_names_sim and joint_names_gym define default joint orders in isaac-sim and isaac-gym.
    """
    joint_names_sim = [
        'left_hip_yaw_joint',
        'right_hip_yaw_joint',
        'torso_joint',
        'left_hip_roll_joint',
        'right_hip_roll_joint',
        'left_shoulder_pitch_joint',
        'right_shoulder_pitch_joint',
        'left_hip_pitch_joint',
        'right_hip_pitch_joint',
        'left_shoulder_roll_joint',
        'right_shoulder_roll_joint',
        'left_knee_joint',
        'right_knee_joint',
        'left_shoulder_yaw_joint',
        'right_shoulder_yaw_joint',
        'left_ankle_joint',
        'right_ankle_joint',
        'left_elbow_joint',
        'right_elbow_joint',
    ]

    joint_names_gym = [
        'left_hip_yaw_joint',
        'left_hip_roll_joint',
        'left_hip_pitch_joint',
        'left_knee_joint',
        'left_ankle_joint',
        'right_hip_yaw_joint',
        'right_hip_roll_joint',
        'right_hip_pitch_joint',
        'right_knee_joint',
        'right_ankle_joint',
        'torso_joint',
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint',
        'left_elbow_joint',
        'right_shoulder_pitch_joint',
        'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint',
        'right_elbow_joint',
    ]

    def __init__(
        self,
        config: H1MoveBySpeedControllerCfg,
        robot: BaseRobot,
        scene: IScene,
    ) -> None:
        super().__init__(config=config, robot=robot, scene=scene)

        self._policy = RLPolicy(path=config.policy_weights_path).get_inference_policy(device='cpu')
        self.joint_subset = None
        self.joint_names = config.joint_names
        self.gym_adapter = gymutil.gym_adapter(self.joint_names_gym, self.joint_names_sim)
        if self.joint_names is not None:
            self.joint_subset = ArticulationSubset(self.robot.articulation, self.joint_names)
        self._old_joint_positions = np.zeros(19)
        self.policy_input_obs_num = 492
        self._old_policy_obs = np.zeros(self.policy_input_obs_num)
        self._apply_times_left = (
            0  # Specifies how many times the action generated by the policy needs to be repeatedly applied.
        )

        self.height_points = init_height_points()
        self.num_height_points = len(self.height_points)
        self.dynamic_height_samples = DynamicHeightSamples()
        self.point_cloud_sensor = None
        self.update_height_samples_trigger = 0
        self.static_height_samples = StaticHeightSamples()

    def forward(
        self,
        forward_speed: float = 0,
        rotation_speed: float = 0,
        lateral_speed: float = 0,
    ) -> ArticulationAction:
        if self._apply_times_left > 0:
            self._apply_times_left -= 1
            if self.joint_subset is None:
                return ArticulationAction(joint_positions=self.applied_joint_positions)
            return self.joint_subset.make_articulation_action(
                joint_positions=self.applied_joint_positions,
                joint_velocities=None,
            )

        # Get obs for policy.
        robot_base = self.robot.get_robot_base()
        base_pose_w = robot_base.get_world_pose()
        # base_quat_w = torch.tensor(base_pose_w[1]).reshape(1, -1)
        # base_ang_vel_w = torch.tensor(robot_base.get_angular_velocity()[:]).reshape(1, -1)
        # base_ang_vel = np.array(math_utils.quat_rotate_inverse(base_quat_w, base_ang_vel_w).reshape(-1))
        # # base_ang_vel = base_ang_vel * np.pi / 180.0

        torso_link: IRigidBody = self.robot._torso_link
        torso_pos_w, torso_quat_w = torso_link.get_world_pose()

        robot_pos_for_height_samples = base_pose_w[0].copy()
        floor_height = self.robot.get_ankle_height() - 0.05
        robot_pos_for_height_samples[2] = floor_height

        if self.point_cloud_sensor is None:
            if 'tp_pointcloud' in self.robot.sensors:
                self.point_cloud_sensor: BaseSensor = self.robot.sensors['tp_pointcloud']

        # Update height samples.
        if self.update_height_samples_trigger == 0:
            sensor_data = self.point_cloud_sensor.get_data()
            if 'pointcloud' in sensor_data and sensor_data['pointcloud'] is not None:
                point_cloud_data = torch.Tensor(sensor_data['pointcloud'])
                if point_cloud_data.shape[0] > 1:
                    self.dynamic_height_samples.set_heights(point_cloud_data, robot_pos_for_height_samples)
            else:
                self.update_height_samples_trigger -= 1

        self.update_height_samples_trigger += 1
        if self.update_height_samples_trigger == 5:
            self.update_height_samples_trigger = 0

        # Calculate the height points (in shape of [num_height_points, 3]) in world frame.
        height_points_w = quat_apply_yaw(torch.Tensor(torso_quat_w), self.height_points) + torch.Tensor(torso_pos_w)

        heights = self.dynamic_height_samples.get_heights(height_points_w)

        heights[heights > floor_height + 0.2] = floor_height
        heights[heights < floor_height - 0.2] = floor_height

        imu_link: IRigidBody = self.robot._imu_link
        imu_pose_w = imu_link.get_world_pose()
        imu_quat_w = torch.tensor(imu_pose_w[1]).reshape(1, -1)
        imu_ang_vel_w = torch.tensor(imu_link.get_angular_velocity()[:]).reshape(1, -1)
        imu_ang_vel = np.array(math_utils.quat_rotate_inverse(imu_quat_w, imu_ang_vel_w).reshape(-1))
        # imu_ang_vel = imu_ang_vel * np.pi / 180.0

        projected_gravity = torch.tensor([[0.0, 0.0, -1.0]], device='cpu', dtype=torch.float)
        projected_gravity = np.array(math_utils.quat_rotate_inverse(imu_quat_w, projected_gravity).reshape(-1))
        joint_pos = (
            self.joint_subset.get_joint_positions()
            if self.joint_subset is not None
            else self.robot.articulation.get_joint_positions()
        )
        joint_vel = (
            self.joint_subset.get_joint_velocities()
            if self.joint_subset is not None
            else self.robot.articulation.get_joint_velocities()
        )
        default_dof_pos = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.4,
                -0.4,
                0.0,
                0.0,
                0.8,
                0.8,
                0.0,
                0.0,
                -0.4,
                -0.4,
                0.0,
                0.0,
            ]
        )

        joint_pos -= default_dof_pos

        heights = np.clip(torso_pos_w[2] - 1.0 - heights, -1.0, 1.0) * 5.0

        # Set action command.
        tracking_command = np.array([forward_speed, lateral_speed, rotation_speed], dtype=np.float32)

        current_obs = np.concatenate(
            [
                tracking_command * np.array([2.0, 2.0, 0.25]),  # dim = 3
                imu_ang_vel * 0.25,  # dim = 3
                projected_gravity,  # dim = 3
                self.gym_adapter.sim2gym(joint_pos),  # dim = 19
                self.gym_adapter.sim2gym(joint_vel) * 0.05,  # dim = 19
                self.gym_adapter.sim2gym(self._old_joint_positions.reshape(19)),  # dim = 19
                heights,  # dim = 96
            ]
        )
        policy_obs = np.concatenate([self._old_policy_obs[66:396], current_obs])
        self._old_policy_obs = policy_obs
        policy_obs = policy_obs.reshape(1, 492)

        # Infer with policy.
        with torch.inference_mode():
            joint_positions: np.ndarray = (
                self._policy(torch.tensor(policy_obs, dtype=torch.float32).to('cpu')).detach().numpy() * 0.25
            )
            joint_positions = joint_positions[0]
            joint_positions = self.gym_adapter.gym2sim(joint_positions)
            self._old_joint_positions = joint_positions * 4
            self.applied_joint_positions = joint_positions + default_dof_pos
            self._apply_times_left = 3

        if self.joint_subset is None:
            return ArticulationAction(joint_positions=self.applied_joint_positions)
        return self.joint_subset.make_articulation_action(
            joint_positions=self.applied_joint_positions, joint_velocities=None
        )

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """Convert input action (in 1d array format) to joint positions to apply.

        Args:
            action (List | np.ndarray): 3-element 1d array containing:
              0. forward_speed (float)
              1. lateral_speed (float)
              2. rotation_speed (float)

        Returns:
            ArticulationAction: joint positions to apply.
        """
        assert len(action) == 3, 'action must contain 3 elements'

        return self.forward(
            forward_speed=action[0],
            lateral_speed=action[1],
            rotation_speed=action[2],
        )

    def get_obs(self):
        return {
            'finished': True,
        }
