import numpy as np
import torch
from PIL import Image
from torch import Tensor
from typing import Optional, Tuple
import torch
from dataclasses import dataclass


def open_image(image_or_image_path):
    if isinstance(image_or_image_path, Image.Image):
        return image_or_image_path
    elif isinstance(image_or_image_path, str):
        return Image.open(image_or_image_path)
    else:
        raise ValueError("Unsupported input type!")

def split_and_clean(text):
    # Split by <image> while preserving the delimiter
    import re
    parts = re.split(r'(<image>)', text)
    results = []
    for part in parts:
        if part == '<image>':
            results.append(part)
        else:
            # Remove all newlines and strip whitespace from both ends
            clean_part = part.replace('\n', '').strip()
            if clean_part:  # Skip empty strings
                results.append(clean_part)
    return results

def chunk_token(dp_actions):
    out_list = []
    out_list_read = []
    
    for i in range(len(dp_actions)):
        xyyaw = dp_actions[i]
        x = xyyaw[0]
        yaw = xyyaw[-1]
        x_prop = torch.abs(x / 0.25)
        yaw_prop = torch.abs(yaw * 12 / torch.pi)
        if x < 0.05 and torch.abs(yaw) < 0.05:
            out_list_read.append("stop")
            out_list.append(0)
        else:
            if x_prop >= yaw_prop:
                out_list_read.append("↑")
                out_list.append(1)
            elif yaw < 0:
                out_list_read.append("→")
                out_list.append(3)
            else:
                out_list_read.append("←")
                out_list.append(2)

    return out_list

def traj_to_actions(dp_actions):
    def reconstruct_xy_from_delta(delta_xyt):
        start_xy = np.zeros((len(delta_xyt), 2))
        delta_xy = delta_xyt[:, :, :2]
        cumsum_xy = np.cumsum(delta_xy, axis=1)

        B = delta_xyt.shape[0]
        T = delta_xyt.shape[1]
        xy = np.zeros((B, T + 1, 2))
        xy[:, 0] = start_xy
        xy[:, 1:] = start_xy[:, None, :] + cumsum_xy

        return xy

    def trajectory_to_discrete_actions_close_to_goal(trajectory, step_size=0.25, turn_angle_deg=15, lookahead=4):
        actions = []
        yaw = 0.0
        pos = trajectory[0]
        turn_angle_rad = np.deg2rad(turn_angle_deg)
        traj = trajectory
        goal = trajectory[-1]
        
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        while np.linalg.norm(pos - goal) > 0.2:
            dists = np.linalg.norm(traj - pos, axis=1)
            nearest_idx = np.argmin(dists)
            target_idx = min(nearest_idx + lookahead, len(traj) - 1)
            target = traj[target_idx]
            target_dir = target - pos
            if np.linalg.norm(target_dir) < 1e-6:
                break
            target_yaw = np.arctan2(target_dir[1], target_dir[0])
            delta_yaw = normalize_angle(target_yaw - yaw)
            n_turns = int(round(delta_yaw / turn_angle_rad))
            if n_turns > 0:
                actions += [2] * n_turns
            elif n_turns < 0:
                actions += [3] * (-n_turns)
            yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

            next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])

            if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
                break

            actions.append(1)
            pos = next_pos

        return actions
    
    def reconstruct_trajectory(actions):
        T = len(actions)
        absolute_poses = np.zeros((T+1, 3))
        
        for i in range(1, T+1):
            prev_x, prev_y, prev_theta = absolute_poses[i-1]
            action_index = actions[i-1]
            if action_index == 1:
                dx, dy, dtheta = 0.25, 0, 0
            elif action_index == 2:
                dx, dy, dtheta = 0, 0, np.pi / 12
            elif action_index == 3:
                dx, dy, dtheta = 0, 0, -np.pi / 12
            else:
                print("Warning!!!")
            
            cos_theta = np.cos(prev_theta)
            sin_theta = np.sin(prev_theta)
            x = prev_x + dx * cos_theta - dy * sin_theta
            y = prev_y + dx * sin_theta + dy * cos_theta
            theta = prev_theta + dtheta
            absolute_poses[i] = [x, y, theta]
        return absolute_poses
    
    def visualize(all_trajectory, reconstructed_trajectory):
        from datetime import datetime
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(len(all_trajectory)):
            ax.plot(
                all_trajectory[i,:,0], all_trajectory[i,:,1], 
                color='purple', alpha=0.3, 
                label='diffusion output' if i == 0 else ""
            )
        
        trajectory = np.mean(all_trajectory, axis=0)
        ax.plot(trajectory[:,0], trajectory[:,1], color='red', alpha=0.8, label='mean trajectory')
        ax.plot(reconstructed_trajectory[:,0], reconstructed_trajectory[:,1], color='blue', alpha=0.8, label='reconstructed from discrete')
        
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        plt.show()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pred_trajectory_{current_time}.png"
        plt.savefig(os.path.join("./tmp", filename), dpi=300, bbox_inches='tight')
        
    # unnormalize
    dp_actions[:, :, :2] /= 4.0
    all_trajectory = reconstruct_xy_from_delta(dp_actions.float().cpu().numpy())
    trajectory = np.mean(all_trajectory, axis=0)
    actions = trajectory_to_discrete_actions_close_to_goal(trajectory)
    return actions

@dataclass
class S2Input:
    idx: Optional[int] = -1
    instruction: Optional[str] = None
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    pose: Optional[Tuple[float, float, float]] = None
    look_down: Optional[bool] = False
    should_infer: Optional[bool] = False

@dataclass
class S2Output:
    idx: Optional[int] = -1
    is_infering: Optional[bool] = False
    output_action: Optional[np.ndarray] = None
    output_pixel: Optional[np.ndarray] = None
    output_latent: Optional[torch.Tensor] = None
    rgb_memory: Optional[np.ndarray] = None # 用于记录pixel goal那一帧的rgb
    depth_memory: Optional[np.ndarray] = None # 用于记录pixel goal那一帧的depth
 
    def validate(self):
        """确保output_action、output_pixel和output_latent中只有一个为非None"""
        outputs = [self.output_action, self.output_pixel, self.output_latent]
        non_none_count = sum(1 for x in outputs if x is not None)
        return non_none_count > 0 and self.idx >= 0
        
@dataclass
class S1Input:
    pixel_goal: Optional[np.ndarray] = None
    latent: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None

@dataclass
class S1Output:
    # idx: Optional[int] = None
    idx: Optional[list] = None
    trajectory: Optional[np.ndarray] = None  # Trajectory path
    linear_velocity: Optional[float] = None  # Linear velocity
    angular_velocity: Optional[float] = None  # Angular velocity
    vis_image: Optional[np.ndarray] = None



def image_resize(
    img: Tensor,
    size: Tuple[int, int],
    channels_last: bool = False,
    interpolation_mode: str = "area",
) -> torch.Tensor:
    """Resizes an img.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    if channels_last:
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)

    img = torch.nn.functional.interpolate(img.float(), size=size, mode=interpolation_mode).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> Tuple[float, float]:
    """Calculates polar coordinates (rho, theta) relative to a given position and
    heading to a given goal position. 'rho' is the distance from the agent to the goal,
    and theta is how many radians the agent must turn (to the left, CCW from above) to
    face the goal. Coordinates are in (x, y), where x is the distance forward/backwards,
    and y is the distance to the left or right (right is negative)

    Args:
        curr_pos (np.ndarray): Array of shape (2,) representing the current position.
        curr_heading (float): The current heading, in radians. It represents how many
            radians  the agent must turn to the left (CCW from above) from its initial
            heading to reach its current heading.
        curr_goal (np.ndarray): Array of shape (2,) representing the goal position.

    Returns:
        Tuple[float, float]: A tuple of floats representing the polar coordinates
            (rho, theta).
    """
    rotation_matrix = get_rotation_matrix(-curr_heading, ndims=2)
    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal

    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])

    return float(rho), float(theta)

def get_rotation_matrix(angle: float, ndims: int = 2) -> np.ndarray:
    """Returns a 2x2 or 3x3 rotation matrix for a given angle; if 3x3, the z-axis is
    rotated."""
    if ndims == 2:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
    elif ndims == 3:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("ndims must be 2 or 3")