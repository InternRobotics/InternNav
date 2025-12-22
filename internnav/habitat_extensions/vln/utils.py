import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers.image_utils import to_numpy_array

from internnav.model.utils.vln_utils import open_image


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def get_axis_align_matrix():
    ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    return ma


def get_intrinsic_matrix(sensor_cfg) -> np.ndarray:
    width = sensor_cfg.width
    height = sensor_cfg.height
    fov = sensor_cfg.hfov
    fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    fy = fx  # Assuming square pixels (fx = fy)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    intrinsic_matrix = np.array([[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return intrinsic_matrix


def xyz_pitch_to_tf_matrix(xyz: np.ndarray, pitch: float) -> np.ndarray:
    """Converts a given position and pitch angle to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        pitch (float): The pitch angle in radians for y axis.
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """

    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch), x],
            [0, 1, 0, y],
            [-np.sin(pitch), 0, np.cos(pitch), z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def xyz_yaw_pitch_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """Converts a given position and yaw, pitch angles to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians.
        pitch (float): The pitch angle in radians for y axis.
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    rot1 = xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
    rot2 = xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot1 @ rot2
    transformation_matrix[:3, 3] = xyz
    return transformation_matrix


def pixel_to_gps(pixel, depth, intrinsic, tf_camera_to_episodic):
    '''
    Args:
        pixel: (2,) - [u, v] pixel coordinates
        depth: (H, W) - depth image where depth[v, u] gives depth in meters
        intrinsic: (4, 4) - camera intrinsic matrix
        tf_camera_to_episodic: (4, 4) - transformation from camera to episodic frame
    Returns:
        (x, y): (x, y) coordinates in the episodic frame
    '''
    v, u = pixel
    z = depth[v, u]
    print("depth: ", z)

    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    point_camera = np.array([x, y, z, 1.0])

    # Transform to episodic frame
    point_episodic = tf_camera_to_episodic @ point_camera
    point_episodic = point_episodic[:3] / point_episodic[3]

    x = point_episodic[0]
    y = point_episodic[1]

    return (x, y)  # same as habitat gps


def dot_matrix_two_dimensional(
    image_or_image_path,
    save_path=None,
    dots_size_w=8,
    dots_size_h=8,
    save_img=False,
    font_path='fonts/arial.ttf',
    pixel_goal=None,
):
    """
    takes an original image as input, save the processed image to save_path. Each dot is labeled with two-dimensional Cartesian coordinates (x,y). Suitable for single-image tasks.
    control args:
    1. dots_size_w: the number of columns of the dots matrix
    2. dots_size_h: the number of rows of the dots matrix
    """
    with open_image(image_or_image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img, 'RGB')

        width, height = img.size
        grid_size_w = dots_size_w + 1
        grid_size_h = dots_size_h + 1
        cell_width = width / grid_size_w
        cell_height = height / grid_size_h

        font = ImageFont.truetype(font_path, width // 40)  # Adjust font size if needed; default == width // 40

        target_i = target_j = None
        if pixel_goal is not None:
            y_pixel, x_pixel = pixel_goal[0], pixel_goal[1]
            # Validate pixel coordinates
            if not (0 <= x_pixel < width and 0 <= y_pixel < height):
                raise ValueError(f"pixel_goal {pixel_goal} exceeds image dimensions ({width}x{height})")

            # Convert to grid coordinates
            target_i = round(x_pixel / cell_width)
            target_j = round(y_pixel / cell_height)

            # Validate grid bounds
            if not (1 <= target_i <= dots_size_w and 1 <= target_j <= dots_size_h):
                raise ValueError(
                    f"pixel_goal {pixel_goal} maps to grid ({target_j},{target_i}), "
                    f"valid range is (1,1)-({dots_size_h},{dots_size_w})"
                )

        count = 0

        for j in range(1, grid_size_h):
            for i in range(1, grid_size_w):
                x = int(i * cell_width)
                y = int(j * cell_height)

                pixel_color = img.getpixel((x, y))
                # choose a more contrasting color from black and white
                if pixel_color[0] + pixel_color[1] + pixel_color[2] >= 255 * 3 / 2:
                    opposite_color = (0, 0, 0)
                else:
                    opposite_color = (255, 255, 255)

                if pixel_goal is not None and i == target_i and j == target_j:
                    opposite_color = (255, 0, 0)  # Red for target

                circle_radius = width // 240  # Adjust dot size if needed; default == width // 240
                draw.ellipse(
                    [(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)],
                    fill=opposite_color,
                )

                text_x, text_y = x + 3, y
                count_w = count // dots_size_w
                count_h = count % dots_size_w
                label_str = f"({count_w+1},{count_h+1})"
                draw.text((text_x, text_y), label_str, fill=opposite_color, font=font)
                count += 1
        if save_img:
            print(">>> dots overlaid image processed, stored in", save_path)
            img.save(save_path)
        return img


def preprocess_depth_image_v2(depth_image, do_depth_scale=True, depth_scale=1000, target_height=384, target_width=384):
    resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)

    img = to_numpy_array(resized_depth_image)
    if do_depth_scale:
        img = img / depth_scale

    return img, (target_width, target_height)
