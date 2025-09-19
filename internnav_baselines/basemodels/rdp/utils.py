import base64
import math
import pickle

import numpy as np
import torch

_POLE_LIMIT = 1.0 - 1e-6


class FixedLengthStack:
    def __init__(self, max_size):
        self.max_size = max_size
        self.stack = []

    def push(self, item):
        if len(self.stack) >= self.max_size:
            self.stack.pop(0)  # Remove the oldest item
        self.stack.append(item)  # Add the new item

    def get_stack(self, reverse=False):
        if reverse:
            return self.reverse()
        else:
            return self.stack

    def reverse(self):
        return self.stack[::-1]  # without modifying the original stack


def yaw_rotmat(yaw: float):
    try:
        R = torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), 0.0],
                [torch.sin(yaw), torch.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
    except Exception:
        R = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    return R


def to_local_coords(positions, curr_pos, curr_yaw: float):
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    if isinstance(positions, torch.Tensor):
        rotmat = rotmat.to(positions.device)
        return (positions - curr_pos).matmul(rotmat)
    else:
        return (positions - curr_pos).dot(rotmat)


def yaw_rotmat_batch(yaws: torch.Tensor) -> torch.Tensor:
    """
    Generate batch rotation matrices from yaw angles.

    Args:
        yaws (torch.Tensor): shape (B,), yaw angles in radians

    Returns:
        torch.Tensor: shape (B, 3, 3), rotation matrices
    """
    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    zeros = torch.zeros_like(yaws)
    ones = torch.ones_like(yaws)

    rotmats = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )  # shape: (B, 3, 3)

    return rotmats


def to_local_coords_batch(positions: torch.Tensor, curr_pos: torch.Tensor, curr_yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert global positions to local coordinates for a batch.

    Args:
        positions (torch.Tensor): shape (B, D), global positions
        curr_pos (torch.Tensor): shape (B, D), current positions
        curr_yaw (torch.Tensor): shape (B,), yaw angles in radians

    Returns:
        torch.Tensor: shape (B, D), local coordinates
    """
    D = positions.shape[-1]
    rotmats = yaw_rotmat_batch(curr_yaw)  # (B, 3, 3)
    if D == 2:
        rotmats = rotmats[:, :2, :2]
    elif D == 3:
        rotmats = rotmats
    else:
        raise ValueError('positions must have 2 or 3 dimensions')

    relative_pos = positions - curr_pos  # (B, D)
    local_pos = torch.bmm(relative_pos.unsqueeze(1), rotmats).squeeze(1)  # (B, D)

    return local_pos


def transfer(obs):
    obs = base64.b64decode(obs)
    obs = pickle.loads(obs)
    return obs


def compute_actions(
    globalgps,
    yaws,
    curr_time,
    fill_mode,
    len_traj_pred,
    waypoint_spacing,
    learn_angle,
    metric_waypoint_spacing,
    num_action_params,
    normalize=False,
):
    start_index = curr_time
    end_index = curr_time + len_traj_pred * waypoint_spacing + 1
    yaw = yaws[start_index:end_index:waypoint_spacing]
    globalgps = globalgps[:, [0, 1]]
    positions = globalgps[start_index:end_index:waypoint_spacing]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape != (len_traj_pred + 1,):
        const_len = len_traj_pred + 1 - yaw.shape[0]
        if fill_mode == 'constant':
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate(
                [positions, np.repeat(positions[-1][None], const_len, axis=0)],
                axis=0,
            )
        elif fill_mode == 'zero':
            yaw = np.concatenate([yaw, np.zeros(const_len)])
            positions = np.concatenate([positions, np.zeros((const_len, 2))], axis=0)

    assert yaw.shape == (len_traj_pred + 1,), f'{yaw.shape} and {(len_traj_pred + 1,)} should be equal'
    assert positions.shape == (
        len_traj_pred + 1,
        2,
    ), f'{positions.shape} and {(len_traj_pred + 1, 2)} should be equal'

    waypoints = to_local_coords(positions, positions[0], yaw[0])

    assert waypoints.shape == (
        len_traj_pred + 1,
        2,
    ), f'{waypoints.shape} and {(len_traj_pred + 1, 2)} should be equal'

    # if learn_angle:
    yaw = yaw[1:] - yaw[0]
    actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
    # else:
    #     actions = waypoints[1:]

    if normalize:
        actions[:, :2] /= metric_waypoint_spacing * waypoint_spacing

    if learn_angle:
        assert actions.shape == (
            len_traj_pred,
            num_action_params,
        ), f'{actions.shape} and {(len_traj_pred, num_action_params)} should be equal'

    return actions


def get_delta(actions):
    if isinstance(actions, torch.Tensor):
        # Proceed with 2D case
        if len(actions.shape) == 2:
            ex_actions = torch.cat(
                [
                    torch.zeros((1, actions.shape[-1]), device=actions.device),
                    actions,
                ],
                dim=0,
            )
            # Regular difference for all dimensions except the last one
            delta = ex_actions[1:, :-1] - ex_actions[:-1, :-1]
            # Angular difference for the last dimension
            angle_delta = ex_actions[1:, -1] - ex_actions[:-1, -1]
            angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
            # Combine regular and angular differences
            delta = torch.cat([delta, angle_delta.unsqueeze(-1)], dim=-1)
        else:
            # For higher dimensions (batch dimension)
            ex_actions = torch.cat(
                [
                    torch.zeros(
                        (actions.shape[0], 1, actions.shape[-1]),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            )
            # Regular difference for all dimensions except the last one
            delta = ex_actions[:, 1:, :-1] - ex_actions[:, :-1, :-1]
            # Angular difference for the last dimension
            angle_delta = ex_actions[:, 1:, -1] - ex_actions[:, :-1, -1]
            angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
            # Combine regular and angular differences
            delta = torch.cat([delta, angle_delta.unsqueeze(-1)], dim=-1)
    elif isinstance(actions, np.ndarray):
        if len(actions.shape) == 2:
            ex_actions = np.concatenate([np.zeros((1, actions.shape[-1])), actions], axis=0)
            # Regular difference for all dimensions except the last one
            delta = ex_actions[1:, :-1] - ex_actions[:-1, :-1]
            # Angular difference for the last dimension
            angle_delta = ex_actions[1:, -1] - ex_actions[:-1, -1]
            angle_delta = np.arctan2(np.sin(angle_delta), np.cos(angle_delta))
            # Combine regular and angular differences
            delta = np.concatenate([delta, angle_delta[:, np.newaxis]], axis=-1)
        else:
            ex_actions = np.concatenate(
                [np.zeros((actions.shape[0], 1, actions.shape[-1])), actions],
                axis=1,
            )
            # Regular difference for all dimensions except the last one
            delta = ex_actions[:, 1:, :-1] - ex_actions[:, :-1, :-1]
            # Angular difference for the last dimension
            angle_delta = ex_actions[:, 1:, -1] - ex_actions[:, :-1, -1]
            angle_delta = np.arctan2(np.sin(angle_delta), np.cos(angle_delta))
            # Combine regular and angular differences
            delta = np.concatenate([delta, angle_delta[..., np.newaxis]], axis=-1)

    return delta


def normalize_data(data, stats, device=None):
    if device is not None:
        if isinstance(stats['min'], np.ndarray):
            stats['min'] = torch.from_numpy(stats['min'])
            stats['max'] = torch.from_numpy(stats['max'])
            stats['min'] = stats['min'].to(device)
            stats['max'] = stats['max'].to(device)

    # nomalize to [0,1]
    try:
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    except Exception:
        ndata = (data - stats.min) / (stats.max - stats.min)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    # else:
    #     ndata = (data[:, :2] - stats['min'][:2]) / (stats['max'][:2] - stats['min'][:2])
    #     ndata = ndata * 2 - 1
    return ndata


def map_action_to_2d(delta_actions, max_distance=0.5):
    """convert [delta_x, delta_y, delta_yaw] to normalized polar coordinates [r, theta]
    Args:
        delta_actions: tensor of shape (N, 3), contains [delta_x, delta_y, delta_yaw]
        max_distance: maximum distance value for normalizing r
    Returns:
        actions_2d: tensor of shape (N, 2), contains normalized [r, theta]
        where r and theta are in the range of [-1, 1]
    """
    actions_2d = torch.zeros((delta_actions.shape[0], 2))

    for a_idx, action in enumerate(delta_actions):
        dx, dy, dyaw = action[0], action[1], action[2]

        # calculate moving distance r (Euclidean distance)
        r = torch.sqrt(dx * dx + dy * dy)

        # calculate rotation angle theta (radians)
        # theta = torch.atan2(dy, dx)
        theta = dyaw

        # if not moving, then r and theta are both 0
        if dx == dy == action[2] == 0:
            r = 0
            theta = 0

        actions_2d[a_idx] = torch.tensor([r, theta])

    return actions_2d


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-10:
        return np.identity(3)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
        ),
        dtype=np.float64,
    )


def matrix_to_euler_angles(mat: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert rotation matrix to Euler XYZ extrinsic or intrinsic angles.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.
        degrees (bool, optional): Whether returned angles should be in degrees.
        extrinsic (bool, optional): True if the rotation matrix follows the extrinsic matrix
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic matrix conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray: Euler XYZ angles (intrinsic form) if extrinsic is False and Euler XYZ angles (extrinsic form) if extrinsic is True.
    """
    if extrinsic:
        if mat[2, 0] > _POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[2, 0] < -_POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        roll = np.arctan2(mat[2, 1], mat[2, 2])
        pitch = -np.arcsin(mat[2, 0])
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])
    else:
        if mat[0, 2] > _POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[0, 2] < -_POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])
        roll = -math.atan2(mat[1, 2], mat[2, 2])
        pitch = math.asin(mat[0, 2])
        yaw = -math.atan2(mat[0, 1], mat[0, 0])

        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])


def quat_to_euler_angles(quat: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert input quaternion to Euler XYZ or ZYX angles.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).
        degrees (bool, optional): Whether returned angles should be in degrees. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.


    Returns:
        np.ndarray: Euler XYZ angles (intrinsic form) if extrinsic is False and Euler XYZ angles (extrinsic form) if extrinsic is True.
    """
    return matrix_to_euler_angles(quat_to_rot_matrix(quat), degrees=degrees, extrinsic=extrinsic)
