import cv2
import numpy as np


def fill_small_holes(depth_img: np.ndarray, area_thresh: int) -> np.ndarray:
    """
    Identifies regions in the depth image that have a value of 0 and fills them in
    with 1 if the region is smaller than a given area threshold.

    Args:
        depth_img (np.ndarray): The input depth image
        area_thresh (int): The area threshold for filling in holes

    Returns:
        np.ndarray: The depth image with small holes filled in
    """
    # Create a binary image where holes are 1 and the rest is 0
    binary_img = np.where(depth_img == 0, 1, 0).astype("uint8")

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filled_holes = np.zeros_like(binary_img)

    for cnt in contours:
        # If the area of the contour is smaller than the threshold
        if cv2.contourArea(cnt) < area_thresh:
            # Fill the contour
            cv2.drawContours(filled_holes, [cnt], 0, 1, -1)

    # Create the filled depth image
    filled_depth_img = np.where(filled_holes == 1, 1, depth_img)

    return filled_depth_img


class MP3DGTPerception:
    def __init__(self, max_depth, min_depth, fx, fy):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.fx = fx
        self.fy = fy

    def predict(self, depth, targets, tf_camera_to_ply, area_threshold=2500):
        '''
        Get the gt semantic map of the target objects
        image: (H, W, 3) current rgb frame
        depth: (H, W) current depth frame
        targets: (N, 6) bboxes of the target objects, first 3 are coordinates of min corner, last 3 are coordinates of max corner
        area_threshold: int
        return: (N, H, W) gt semantic map of the target objects
        '''
        # get the point clouds of current frame
        filled_depth = fill_small_holes(depth, area_threshold)
        scaled_depth = filled_depth * (self.max_depth - self.min_depth) + self.min_depth
        mask = scaled_depth < self.max_depth
        point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, self.fx, self.fy)
        point_cloud_ply_frame = transform_points(tf_camera_to_ply, point_cloud_camera_frame)

        # mark the points in the target objects' bboxes
        semantic_images = []
        for target in targets:
            min_x, min_y, min_z = target[:3]
            max_x, max_y, max_z = target[3:]

            in_bbox = (
                (point_cloud_ply_frame[:, 0] >= min_x)
                & (point_cloud_ply_frame[:, 0] <= max_x)
                & (point_cloud_ply_frame[:, 1] >= min_y)
                & (point_cloud_ply_frame[:, 1] <= max_y)
                & (point_cloud_ply_frame[:, 2] >= min_z)
                & (point_cloud_ply_frame[:, 2] <= max_z)
            )
            in_bbox_points = point_cloud_ply_frame[in_bbox]
            semantic_image = np.zeros(depth.shape, dtype=np.uint8)
            if len(in_bbox_points) > 0:
                # map the marked points back to the image to get the semantic map
                in_bbox_camera_frame = inverse_transform_points(tf_camera_to_ply, in_bbox_points)
                in_box_image_coords = project_points_to_image(in_bbox_camera_frame, self.fx, self.fy, depth.shape)
                try:
                    mask = [
                        in_box_image_coords[i, 0] < 480 and in_box_image_coords[i, 1] < 640
                        for i in range(len(in_box_image_coords))
                    ]
                    in_box_image_coords = in_box_image_coords[mask]
                    semantic_image[in_box_image_coords[:, 0], in_box_image_coords[:, 1]] = 1
                except Exception as e:
                    print(e)
                semantic_image = fill_small_holes(semantic_image, area_threshold)
            semantic_images.append(semantic_image)
        if len(semantic_images) > 0:
            semantic_images = np.stack(semantic_images, axis=0)
        else:
            semantic_images = np.zeros((1, depth.shape[0], depth.shape[1]), dtype=np.uint8)
        return semantic_images


def transform_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    # Add a homogeneous coordinate of 1 to each point for matrix multiplication
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply the transformation matrix to the points
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T

    # Remove the added homogeneous coordinate and divide by the last coordinate
    return transformed_points[:, :3] / transformed_points[:, 3:]


def get_point_cloud(depth_image: np.ndarray, mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Calculates the 3D coordinates (x, y, z) of points in the depth image based on
    the horizontal field of view (HFOV), the image width and height, the depth values,
    and the pixel x and y coordinates.

    Args:
        depth_image (np.ndarray): 2D depth image.
        mask (np.ndarray): 2D binary mask identifying relevant pixels.
        fx (float): Focal length in the x direction.
        fy (float): Focal length in the y direction.

    Returns:
        np.ndarray: Array of 3D coordinates (x, y, z) of the points in the image plane.
    """
    v, u = np.where(mask)
    z = depth_image[v, u]
    x = (u - depth_image.shape[1] // 2) * z / fx
    y = (v - depth_image.shape[0] // 2) * z / fy
    cloud = np.stack((x, -y, -z), axis=-1)

    return cloud


def inverse_transform_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Convert point cloud from episodic coordinate system to camera coordinate system

    Args:
        transformation_matrix (np.ndarray): 4x4 transformation matrix
        points (np.ndarray): Point cloud coordinates (N, 3)

    Returns:
        np.ndarray: Point cloud coordinates in camera coordinate system (N, 3)
    """
    # Calculate the inverse of the transformation matrix
    inv_matrix = np.linalg.inv(transformation_matrix)

    # Add a homogeneous coordinate of 1 to each point for matrix multiplication
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply the inverse transformation
    transformed_points = np.dot(inv_matrix, homogeneous_points.T).T

    # Remove the added homogeneous coordinate
    return transformed_points[:, :3] / transformed_points[:, 3:]


def project_points_to_image(points: np.ndarray, fx: float, fy: float, image_shape: tuple) -> np.ndarray:
    """Project points from camera coordinate system to image plane

    Args:
        points (np.ndarray): Points in camera coordinate system (N, 3)
        fx (float): x-axis focal length
        fy (float): y-axis focal length
        image_shape (tuple): Image dimensions (height, width)

    Returns:
        np.ndarray: Image coordinates (N, 2)
    """
    points = np.stack((points[:, 0], -points[:, 1], -points[:, 2]), axis=-1)
    # Ensure points are in front of the camera
    valid_mask = points[:, 2] > 0  # z > 0

    # Calculate image coordinates
    u = points[:, 0] * fx / points[:, 2] + image_shape[1] // 2
    v = points[:, 1] * fy / points[:, 2] + image_shape[0] // 2

    # Combine coordinates
    image_coords = np.stack((v, u), axis=-1)
    image_coords = image_coords.astype(np.int32)
    # Return valid points only
    return image_coords[valid_mask]
