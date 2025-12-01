import math
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from internnav.utils.common_log_util import common_logger as log
from internnav.utils.geometry_utils import quat_to_euler_angles


def create_robot_mask(topdown_global_map_camera, mask_size=20):
    height, width = topdown_global_map_camera._camera._resolution
    center_x, center_y = width // 2, height // 2
    # Calculate the top-left and bottom-right coordinates
    half_size = mask_size // 2
    top_left_x = center_x - half_size
    top_left_y = center_y - half_size
    bottom_right_x = center_x + half_size
    bottom_right_y = center_y + half_size

    # Create the mask
    robot_mask = np.zeros((width, height), dtype=np.uint8)
    robot_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
    return robot_mask


def create_dilation_structure(voxel_size, radius):
    """
    Creates a dilation structure based on the robot's radius.
    """
    radius_cells = int(np.ceil(radius / voxel_size))
    # Create a structuring element for dilation (a disk of the robot's radius)
    dilation_structure = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    cy, cx = radius_cells, radius_cells
    for y in range(2 * radius_cells + 1):
        for x in range(2 * radius_cells + 1):
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius_cells:
                dilation_structure[y, x] = True
    return dilation_structure


def freemap_to_accupancy_map(
    topdown_global_map_camera,
    freemap,
    dilation_iterations=0,
    voxel_size=0.1,
    agent_radius=0.25,
):
    height, width = topdown_global_map_camera._camera._resolution
    occupancy_map = np.zeros((width, height))
    occupancy_map[freemap == 1] = 2
    occupancy_map[freemap == 0] = 255
    if dilation_iterations > 0:
        dilation_structure = create_dilation_structure(voxel_size, agent_radius)
        for i in range(1, dilation_iterations):
            ob_mask = np.logical_and(occupancy_map != 0, occupancy_map != 2)
            expanded_ob_mask = binary_dilation(ob_mask, structure=dilation_structure, iterations=1)
            occupancy_map[expanded_ob_mask & (np.logical_or(occupancy_map == 0, occupancy_map == 2))] = 255 - i * 10
    return occupancy_map


def check_robot_fall(
    robot_position,
    robot_rotation,
    robots_bottom_z,
    pitch_threshold=35,
    roll_threshold=15,
    height_threshold=0.5,
):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles

    roll, pitch, yaw = quat_to_euler_angles(robot_rotation, degrees=True)
    # Check if the pitch or roll exceeds the thresholds
    if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
        is_fall = True
        log.debug('Robot falls down!!!')
        log.debug(f'Current Position: {robot_position}, Orientation: {roll, pitch, yaw}')
    else:
        is_fall = False

    # Check if the height between the robot base and the robot ankle is smaller than a threshold
    robot_ankle_z = robots_bottom_z
    robot_base_z = robot_position[2]
    if robot_base_z - robot_ankle_z < height_threshold:
        is_fall = True
        log.debug('Robot falls down!!!')
        log.debug(f'Current Position: {robot_position}, Orientation: {roll, pitch, yaw}')
    return is_fall


def describe_action(action):
    if action == 1:
        return '向前走0.25米'
    elif action == 2:
        return '左转15°'
    elif action == 3:
        return '右转15°'
    else:
        return '结束'


def get_action_state(obs, action_name):
    controllers = obs['controllers']
    action_state = controllers[action_name]['finished']
    return action_state


def check_is_on_track(
    robot_position,
    robot_rotation,
    action,
    action_index,
    real_points,
):
    if action == 1:
        distance = np.linalg.norm(robot_position[:2] - real_points[action_index][:2])
        if distance > 0.5:
            log.debug(f'[distance:{round(distance, 2)} > 0.5 ] replanning')
            return False
    else:
        from omni.isaac.core.utils.rotations import quat_to_euler_angles

        _, _, real_yaw = quat_to_euler_angles(robot_rotation)
        yaw_diff = abs(real_yaw - real_points[action_index])
        if yaw_diff > math.pi / 6:
            log.debug(f'[yaw_diff: {round(yaw_diff * (180 / math.pi))} 度 > 30 度] replanning')
            return False
    return True


def get_new_position_and_rotation(robot_position, robot_rotation, action):
    from omni.isaac.core.utils.rotations import (
        euler_angles_to_quat,
        quat_to_euler_angles,
    )

    roll, pitch, yaw = quat_to_euler_angles(robot_rotation)
    if action == 1:  # forward
        dx = 0.25 * math.cos(yaw)
        dy = 0.25 * math.sin(yaw)
        new_robot_position = robot_position + [dx, dy, 0]
        new_robot_rotation = robot_rotation
    elif action == 2:  # left
        new_robot_position = robot_position
        new_yaw = yaw + (math.pi / 12)
        new_robot_rotation = euler_angles_to_quat(np.array([roll, pitch, new_yaw]))
    elif action == 3:  # right
        new_robot_position = robot_position
        new_yaw = yaw - (math.pi / 12)
        new_robot_rotation = euler_angles_to_quat(np.array([roll, pitch, new_yaw]))
    else:
        new_robot_position = robot_position
        new_robot_rotation = robot_rotation
    return new_robot_position, new_robot_rotation


def set_seed(seed):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    from omni.isaac.core.utils.torch.maths import set_seed

    set_seed(seed, torch_deterministic=True)
    import omni.isaac.core.utils.torch as torch_utils

    torch_utils.set_seed(seed)
    import omni.replicator.core as rep

    rep.set_global_seed(seed)


def set_seed_model(seed):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def norm_depth(depth_info, min_depth=0, max_depth=10):
    depth_info[depth_info > max_depth] = max_depth
    depth_info = (depth_info - min_depth) / (max_depth - min_depth)
    return depth_info


def draw_trajectory(array, obs_lst, reference_path):
    """
    Draw the globalgps path and orientation arrows onto the depth array.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    from internnav.evaluator.utils.path_plan import world_to_pixel

    points = []
    arrows = []
    camera_pose = obs_lst[-1]["globalgps"]

    ref_points = []
    for position in reference_path:
        px, py = world_to_pixel(position, camera_pose, 200, 500, 500)
        ref_points.append((py, px))

    for obs in obs_lst:
        position = obs["globalgps"]
        px, py = world_to_pixel(position, camera_pose, 200, 500, 500)
        points.append((py, px))

    if "globalrotation" in obs_lst[-1]:
        quat = obs_lst[-1]["globalrotation"]
        _, _, yaw = quat_to_euler_angles(quat)

        # Arrow endpoint in world space
        arrow_length = 0.1  # meters
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        arrow_tip = (position[0] + dx, position[1] + dy)
        px_tip, py_tip = world_to_pixel(arrow_tip, camera_pose, 200, 500, 500)
        arrows.append(((py, px), (py_tip, px_tip)))

    # Now render the image and draw path + arrows
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    canvas = FigureCanvas(fig)
    ax.imshow(array, cmap="viridis", vmin=0, vmax=2)

    # Draw trajectory
    if ref_points:
        points_np = np.array(ref_points)
        if len(points_np) >= 2:
            ax.plot(points_np[:, 0], points_np[:, 1], 'g-', linewidth=2)
        ax.scatter(points_np[:, 0], points_np[:, 1], c='green', s=1)

    # Draw trajectory
    if points:
        points_np = np.array(points)
        if len(points_np) >= 2:
            ax.plot(points_np[:, 0], points_np[:, 1], 'r-', linewidth=2)
        ax.scatter(points_np[:, 0], points_np[:, 1], c='red', s=1)

    # Draw orientation arrows
    for start, end in arrows:
        ax.arrow(
            start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=5, head_length=5, fc='red', ec='red'
        )

    ax.axis("off")
    fig.tight_layout(pad=0)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


from internnav import PROJECT_ROOT_PATH


def draw_action_with_image(array, action, arrow_color=(255, 0, 0)):  # Default to blue
    """
    Draw colored arrow on the bottom of the numpy array while:
    1. Maintaining original image shape
    2. Removing white backgrounds from icons
    3. Coloring the arrow (default blue)

    Args:
        array: Input numpy array (H,W,3) RGB image
        action: Integer action (0=stop, 1=forward, 2=left, 3=right)
        arrow_color: Tuple (R,G,B) for arrow color (default: blue)

    Returns:
        Numpy array with same shape as input, with colored icon at bottom center
    """
    if 'move_by_discrete' in action:
        move = action['move_by_discrete'][0]  # Extract the movement value
    elif 'move_by_flash' in action:
        move = action['move_by_flash'][0]
    else:
        move = 1
    action = move

    # Load action icon
    action_icons = {1: "forward.png", 2: "left.png", 3: "right.png", 0: "stop.png"}
    icon_path = os.path.join(PROJECT_ROOT_PATH, "internnav/utils/images/")
    icon_path = os.path.join(icon_path, (action_icons.get(action, "stop.png")))

    # Convert array to PIL Image
    img = Image.fromarray(array.copy())  # Keep original unchanged

    try:
        # Load icon and convert to RGBA if not already
        icon = Image.open(icon_path).convert('RGBA').resize((40, 40))

        # Process icon:
        # 1. Convert white background to transparent
        # 2. Convert black arrow to specified color
        data = np.array(icon)
        r, g, b, a = data.T

        # Identify white background (high RGB values)
        white_areas = (r > 200) & (g > 200) & (b > 200)
        # Identify arrow (non-white areas)
        arrow_mask = ~white_areas

        # Set white areas to transparent
        data[..., -1][white_areas.T] = 0

        # Color the arrow
        data[..., 0][arrow_mask.T] = arrow_color[0]  # R
        data[..., 1][arrow_mask.T] = arrow_color[1]  # G
        data[..., 2][arrow_mask.T] = arrow_color[2]  # B

        icon = Image.fromarray(data)

        # Calculate position (bottom center)
        icon_pos = (
            (img.width - icon.width) // 2,  # Center horizontally
            img.height - icon.height - 10,  # 10px from bottom
        )

        # Paste icon onto image using alpha channel as mask
        img.paste(icon, icon_pos, icon)

    except Exception as e:
        print(f"Couldn't process icon: {e}")
        return array  # Return original if icon fails

    return np.array(img)  # Return with same shape as input


def draw_action_pil(array, action, arrow_color=(255, 0, 0)):  # default: red
    """
    Draw a colored arrow (or stop icon) on the bottom-center of the image.

    Args:
        array: np.ndarray (H, W, 3) RGB image
        action: int or dict with 'move_by_discrete'/'move_by_flash' (0=stop, 1=forward, 2=left, 3=right)
        arrow_color: (R, G, B)
    Returns:
        np.ndarray with same shape as input
    """
    # Normalize action to int code
    if isinstance(action, dict):
        if 'move_by_discrete' in action:
            move = action['move_by_discrete'][0]
        elif 'move_by_flash' in action:
            move = action['move_by_flash'][0]
        else:
            move = 1
        action_code = int(move)
    else:
        action_code = int(action)

    img = Image.fromarray(array.copy())

    # Icon size relative to image; anti-aliased via supersampling
    base = min(img.width, img.height)
    size = max(32, min(128, int(base * 0.1)))  # 10% of min dim, clamp 32..128
    scale = 3  # supersample for smoother edges
    W, H = size * scale, size * scale

    # Transparent overlay we’ll paste onto the image
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = tuple(arrow_color) + (255,)
    # Optional subtle shadow for contrast
    shadow = (0, 0, 0, 120)

    cx, cy = W // 2, H // 2

    def draw_up_arrow():
        # Shaft
        shaft_w = int(W * 0.22)
        shaft_h = int(H * 0.48)
        head_h = int(H * 0.36)
        y1 = cy + shaft_h // 2
        y0 = y1 - shaft_h
        # Shadow
        draw.rounded_rectangle(
            [cx - shaft_w // 2 + 2, y0 + 2, cx + shaft_w // 2 + 2, y1 + 2], radius=shaft_w // 2, fill=shadow
        )
        # Color shaft
        draw.rounded_rectangle([cx - shaft_w // 2, y0, cx + shaft_w // 2, y1], radius=shaft_w // 2, fill=color)
        # Head triangle
        apex = (cx, y0 - head_h)
        left = (cx - int(W * 0.32), y0)
        right = (cx + int(W * 0.32), y0)
        # Shadow
        draw.polygon(
            [(apex[0] + 2, apex[1] + 2), (left[0] + 2, left[1] + 2), (right[0] + 2, right[1] + 2)], fill=shadow
        )
        # Color
        draw.polygon([apex, left, right], fill=color)

    def draw_stop_icon():
        r = int(min(W, H) * 0.38)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        # Shadow
        s_off = 2
        draw.ellipse([bbox[0] + s_off, bbox[1] + s_off, bbox[2] + s_off, bbox[3] + s_off], fill=shadow)
        # Red circle (or any arrow_color)
        draw.ellipse(bbox, fill=color)
        # White square inside
        ir = int(r * 0.55)
        draw.rectangle([cx - ir, cy - int(ir * 0.6), cx + ir, cy + int(ir * 0.6)], fill=(255, 255, 255, 255))

    if action_code == 0:
        draw_stop_icon()
        rotated = overlay
    else:
        draw_up_arrow()
        # Rotate according to action: forward=up, left=+90°, right=-90°
        angle = {1: 0, 2: 90, 3: -90}.get(action_code, 0)
        rotated = overlay.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Downsample (anti-alias) to final icon size
    icon = rotated.resize((size, size), Image.LANCZOS)

    # Paste at bottom center with a small margin
    margin = max(6, size // 8)
    x = (img.width - icon.width) // 2
    y = img.height - icon.height - margin
    img.paste(icon, (x, y), icon)

    return np.array(img)


def crop(array):
    # Crop 256x256 (as in your original code)
    height, width = array.shape[:2]
    start_x = (width - 256) // 2
    start_y = (height - 256) // 2
    return array[start_y : start_y + 256, start_x : start_x + 256, :]


def obs_to_image(obs_lst, action, output_path: str, reference_path, normalize: bool = True):
    """
    Load .npy file and save as image

    Args:
        npy_path: Path to input .npy file
        output_path: Output image path (extension determines format)
        normalize: Scale values to 0-255 if True
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    first_obs = obs_lst[-1]
    if 'rgb' not in first_obs:
        return
    rgb_array = first_obs['rgb']
    topdown_array = first_obs['topdown_rgb']

    # draw array on rgb array
    rgb_array = cv2.resize(draw_action_pil(rgb_array, action), (256, 256))

    # draw trajectory on depth
    topdown_array = crop(draw_trajectory(topdown_array, obs_lst, reference_path))

    # Combine horizontally (256x256 + 256x256 = 512x256)
    array = np.concatenate((rgb_array, topdown_array), axis=1)

    # Handle different array types
    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    elif array.dtype in (np.float32, np.float64) and normalize:
        array = (array - array.min()) / (array.max() - array.min()) * 255
        array = array.astype(np.uint8)
    elif np.issubdtype(array.dtype, np.integer) and normalize:
        array = ((array - array.min()) * (255 / (array.max() - array.min()))).astype(np.uint8)

    # Upscaling using interpolation, improve resolution
    array = cv2.resize(array, (array.shape[1] * 2, array.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

    # Create and save image
    if array.ndim == 2:  # Grayscale
        Image.fromarray(array).save(output_path)
    elif array.ndim == 3 and array.shape[2] in (3, 4):  # RGB/RGBA
        Image.fromarray(array).save(output_path)
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

    print(f"Saved to {output_path}")


from glob import glob

import cv2


def images_to_video(image_folder, output_path, fps=10):
    """
    Generate a video from a folder of images.

    Parameters:
        image_folder (str): Path to the folder containing images.
        output_path (str): Path to save the output video (e.g., 'output.mp4').
        fps (int): Frames per second.

    Returns:
        None
    """
    # Get sorted image list (assumes image filenames are sortable)
    image_files = glob(os.path.join(image_folder, '*'))
    # Sort numerically based on filename (e.g., "16.png" → 16)
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    if not image_files:
        print("No images found in the folder.")
        return

    # Read the first image to get frame size
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    # Define the codec and video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: Could not read {image_file}, skipping.")
            continue
        out.write(img)

    out.release()
    print(f"Video saved to: {output_path}")


from tqdm import tqdm


def obs_to_video(obs_lst, output_video_path, fps=30):
    """
    Convert a list of observations (with 'rgb' and 'topdown_rgb') directly into a video.

    Args:
        obs_lst (list): List of observations, each containing 'rgb' and 'topdown_rgb' arrays.
        output_video_path (str): Path to save the output video (e.g., 'output.mp4').
        fps (int): Frames per second (default: 30).
    """
    if not obs_lst:
        raise ValueError("Empty observation list!")

    # Get the first frame to determine video dimensions
    first_obs = obs_lst[0]

    # Process the first frame to get dimensions
    rgb_array = first_obs['rgb']
    topdown_array = first_obs['topdown_rgb']

    # Generate first frame image
    # output_path = "/".join(output_video_path.split('/')[:-1])
    # npy_to_image(topdown_array, output_path + '/topdown.png')
    # npy_to_image(rgb_array, output_path + '/rgb.png')
    # np.save(output_path + '/topdown.npy', topdown_array)
    # np.save(output_path + '/rgb.npy', rgb_array)

    # Crop topdown to 256x256 (as in your original code)
    height, width = topdown_array.shape[:2]
    start_x = (width - 256) // 2
    start_y = (height - 256) // 2
    topdown_array = topdown_array[start_y : start_y + 256, start_x : start_x + 256, :]

    # Combine horizontally (256x256 + 256x256 = 512x256)
    combined_array = np.concatenate((rgb_array, topdown_array), axis=1)
    height, width, _ = combined_array.shape

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process and write each frame
    for i, obs in enumerate(tqdm(obs_lst, desc="Generating video")):
        if 'rgb' not in obs or 'topdown_rgb' not in obs:
            print(f"Warning: Observation {i} missing 'rgb' or 'topdown_rgb'")
            continue

        rgb_array = obs['rgb']
        topdown_array = obs['topdown_rgb']

        # Crop topdown to 256x256
        height_td, width_td = topdown_array.shape[:2]
        start_x = (width_td - 256) // 2
        start_y = (height_td - 256) // 2
        topdown_array = topdown_array[start_y : start_y + 256, start_x : start_x + 256, :]

        # Ensure correct shape and type
        if rgb_array.shape != (256, 256, 3) or topdown_array.shape != (256, 256, 3):
            print(f"Warning: Observation {i} has incorrect dimensions")
            continue

        # Convert float arrays (0-1) to uint8 (0-255)
        if rgb_array.dtype == np.float32 or rgb_array.dtype == np.float64:
            if rgb_array.max() <= 1.0:
                rgb_array = (rgb_array * 255).astype(np.uint8)
        if topdown_array.dtype == np.float32 or topdown_array.dtype == np.float64:
            if topdown_array.max() <= 1.0:
                topdown_array = (topdown_array * 255).astype(np.uint8)

        # Combine and write frame
        combined_frame = np.concatenate((rgb_array, topdown_array), axis=1)
        video_writer.write(combined_frame)

    video_writer.release()
    print(f"Video saved to: {output_video_path}")
