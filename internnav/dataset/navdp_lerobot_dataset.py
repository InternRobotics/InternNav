# Override the built-in print function with a timestamp version
import builtins
import json
import os
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from PIL import Image
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
from tqdm import tqdm

original_print = builtins.print


def print(*args, **kwargs):
    try:
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            original_print(f"[{timestamp}]", *args, **kwargs)
    except Exception:  # Catch any exception to prevent crashes
        pass


builtins.print = print


class NavDP_Base_Datset(Dataset):
    def __init__(
        self,
        root_dirs,
        preload_path=False,
        memory_size=8,
        predict_size=24,
        batch_size=64,
        image_size=224,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        pixel_channel=7,
        debug=False,
        preload=False,
        random_digit=False,
        prior_sample=False,
    ):

        self.dataset_dirs = np.array([p for p in os.listdir(root_dirs)])
        self.memory_size = memory_size
        self.image_size = image_size
        self.scene_scale_size = scene_data_scale
        self.trajectory_data_scale = trajectory_data_scale
        self.predict_size = predict_size
        self.debug = debug
        self.trajectory_dirs = []
        self.trajectory_data_dir = []
        self.trajectory_rgb_path = []
        self.trajectory_depth_path = []
        self.trajectory_afford_path = []
        self.random_digit = random_digit
        self.prior_sample = prior_sample
        self.pixel_channel = pixel_channel
        self.item_cnt = 0
        self.batch_size = batch_size
        self.batch_time_sum = 0.0
        self._last_time = None

        if preload is False:
            for group_dir in self.dataset_dirs:  # gibson_zed, 3dfront ...
                all_scene_dirs = np.array([p for p in os.listdir(os.path.join(root_dirs, group_dir))])
                select_scene_dirs = all_scene_dirs[
                    np.arange(0, all_scene_dirs.shape[0], 1 / self.scene_scale_size).astype(np.int32)
                ]
                for scene_dir in select_scene_dirs:
                    all_traj_dirs = np.array([p for p in os.listdir(os.path.join(root_dirs, group_dir, scene_dir))])
                    select_traj_dirs = all_traj_dirs[
                        np.arange(0, all_traj_dirs.shape[0], 1 / self.trajectory_data_scale).astype(np.int32)
                    ]
                    for traj_dir in tqdm(select_traj_dirs):
                        entire_task_dir = os.path.join(root_dirs, group_dir, scene_dir, traj_dir)
                        rgb_dir = os.path.join(entire_task_dir, "videos/chunk-000/observation.images.rgb/")
                        depth_dir = os.path.join(entire_task_dir, "videos/chunk-000/observation.images.depth/")
                        data_path = os.path.join(
                            entire_task_dir, 'data/chunk-000/episode_000000.parquet'
                        )  # intrinsic, extrinsic, cam_traj, path
                        afford_path = os.path.join(entire_task_dir, 'data/chunk-000/path.ply')
                        rgbs_length = len([p for p in os.listdir(rgb_dir)])
                        depths_length = len([p for p in os.listdir(depth_dir)])

                        rgbs_path = []
                        depths_path = []
                        if depths_length != rgbs_length:
                            continue
                        for i in range(rgbs_length):
                            rgbs_path.append(os.path.join(rgb_dir, "%d.jpg" % i))
                            depths_path.append(os.path.join(depth_dir, "%d.png" % i))
                        if os.path.exists(data_path) is False:
                            continue
                        self.trajectory_dirs.append(entire_task_dir)
                        self.trajectory_data_dir.append(data_path)
                        self.trajectory_rgb_path.append(rgbs_path)
                        self.trajectory_depth_path.append(depths_path)
                        self.trajectory_afford_path.append(afford_path)

            save_dict = {
                'trajectory_dirs': self.trajectory_dirs,
                'trajectory_data_dir': self.trajectory_data_dir,
                'trajectory_rgb_path': self.trajectory_rgb_path,
                'trajectory_depth_path': self.trajectory_depth_path,
                'trajectory_afford_path': self.trajectory_afford_path,
            }
            with open(preload_path, 'w') as f:
                json.dump(save_dict, f, indent=4)
        else:
            load_dict = json.load(open(preload_path, 'r'))
            self.trajectory_dirs = load_dict['trajectory_dirs'] * 50
            self.trajectory_data_dir = load_dict['trajectory_data_dir'] * 50
            self.trajectory_rgb_path = load_dict['trajectory_rgb_path'] * 50
            self.trajectory_depth_path = load_dict['trajectory_depth_path'] * 50
            self.trajectory_afford_path = load_dict['trajectory_afford_path'] * 50

    def __len__(self):
        return len(self.trajectory_dirs)

    def load_image(self, image_url):
        image = Image.open(image_url)
        image = np.array(image, np.uint8)
        return image

    def load_depth(self, depth_url):
        depth = Image.open(depth_url)
        depth = np.array(depth, np.uint16)
        return depth

    def load_pointcloud(self, pcd_url):
        pcd = o3d.io.read_point_cloud(pcd_url)
        return pcd

    def process_image(self, image_path):
        image = self.load_image(image_path)
        H, W, C = image.shape
        prop = self.image_size / max(H, W)
        image = cv2.resize(image, (-1, -1), fx=prop, fy=prop)
        pad_width = max((self.image_size - image.shape[1]) // 2, 0)
        pad_height = max((self.image_size - image.shape[0]) // 2, 0)
        pad_image = np.pad(
            image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0
        )
        image = cv2.resize(pad_image, (self.image_size, self.image_size))
        image = np.array(image, np.float32) / 255.0
        return image

    def process_depth(self, depth_path):
        depth = self.load_depth(depth_path) / 10000.0
        H, W = depth.shape
        prop = self.image_size / max(H, W)
        depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
        pad_width = max((self.image_size - depth.shape[1]) // 2, 0)
        pad_height = max((self.image_size - depth.shape[0]) // 2, 0)
        pad_depth = np.pad(
            depth, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0
        )
        pad_depth[pad_depth > 5.0] = 0
        pad_depth[pad_depth < 0.1] = 0
        depth = cv2.resize(pad_depth, (self.image_size, self.image_size))
        depth = np.array(depth, np.float32)
        return depth[:, :, np.newaxis]

    def process_data_parquet(self, index):
        if not os.path.isfile(self.trajectory_data_dir[index]):
            raise FileNotFoundError(self.trajectory_data_dir[index])
        df = pd.read_parquet(self.trajectory_data_dir[index])
        camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
        camera_extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
        trajectory_length = len(df['action'].tolist())
        camera_trajectory = np.array([np.stack(frame) for frame in df['action']], dtype=np.float64)
        return camera_intrinsic, camera_extrinsic, camera_trajectory, trajectory_length

    def process_path_points(self, index):
        trajectory_pcd = self.load_pointcloud(self.trajectory_afford_path[index])
        trajectory_color = np.array(trajectory_pcd.colors)
        color_distance = np.abs(trajectory_color - np.array([0, 0, 0])).sum(
            axis=-1
        )  # sometimes, the path are saved as black points
        select_index = np.where(color_distance < 0.05)[0]
        trajectory_path = o3d.geometry.PointCloud()
        trajectory_path.points = o3d.utility.Vector3dVector(np.asarray(trajectory_pcd.points)[select_index])
        trajectory_path.colors = o3d.utility.Vector3dVector(np.asarray(trajectory_pcd.colors)[select_index])
        return np.array(trajectory_path.points), trajectory_path

    def process_obstacle_points(self, index, path_points):
        trajectory_pcd = self.load_pointcloud(self.trajectory_afford_path[index])
        trajectory_color = np.array(trajectory_pcd.colors)
        trajectory_points = np.array(trajectory_pcd.points)
        color_distance = np.abs(trajectory_color - np.array([0, 0, 0.5])).sum(axis=-1)  # the obstacles are save in blue
        path_lower_bound = path_points.min(axis=0)
        path_upper_bound = path_points.max(axis=0)
        condition_x = (trajectory_points[:, 0] >= path_lower_bound[0] - 2.0) & (
            trajectory_points[:, 0] <= path_upper_bound[0] + 2.0
        )
        condition_y = (trajectory_points[:, 1] >= path_lower_bound[1] - 2.0) & (
            trajectory_points[:, 1] <= path_upper_bound[1] + 2.0
        )
        select_index = np.where((color_distance < 0.05) & condition_x & condition_y)[0]
        trajectory_obstacle = o3d.geometry.PointCloud()
        trajectory_obstacle.points = o3d.utility.Vector3dVector(np.asarray(trajectory_pcd.points)[select_index])
        trajectory_obstacle.colors = o3d.utility.Vector3dVector(np.asarray(trajectory_pcd.colors)[select_index])
        return np.array(trajectory_obstacle.points), trajectory_obstacle

    def process_memory(self, rgb_paths, depth_paths, start_step, memory_digit=1):
        memory_index = np.arange(start_step - (self.memory_size - 1) * memory_digit, start_step + 1, memory_digit)
        outrange_sum = (memory_index < 0).sum()
        memory_index = memory_index[outrange_sum:]
        context_image = np.zeros((self.memory_size, self.image_size, self.image_size, 3), np.float32)
        context_image[outrange_sum:] = np.array([self.process_image(rgb_paths[i]) for i in memory_index])
        context_depth = self.process_depth(depth_paths[start_step])
        return context_image, context_depth, memory_index

    def process_pixel_goal(self, image_url, target_point, camera_intrinsic, camera_extrinsic):
        image = Image.open(image_url)
        image = np.array(image, np.uint8)
        resize_image = self.process_image(image_url)

        coordinate = np.array([-target_point[1], target_point[0], camera_extrinsic[2, 3] * 0.8])
        camera_coordinate = np.matmul(camera_extrinsic[0:3, 0:3], coordinate[:, None])
        pixel_coord_x = camera_intrinsic[0, 2] + (camera_coordinate[0] / camera_coordinate[2]) * camera_intrinsic[0, 0]
        pixel_coord_y = camera_intrinsic[1, 2] + (-camera_coordinate[1] / camera_coordinate[2]) * camera_intrinsic[1, 1]
        pixel_mask = np.zeros_like(image)
        visible_flag = False

        if (
            pixel_coord_x > 0
            and pixel_coord_x < image.shape[1]
            and pixel_coord_y > 0
            and pixel_coord_y < image.shape[0]
        ):
            pixel_mask = cv2.rectangle(
                pixel_mask,
                (int(pixel_coord_x - np.random.randint(6, 12)), int(pixel_coord_y - np.random.randint(6, 12))),
                (int(pixel_coord_x + np.random.randint(6, 12)), int(pixel_coord_y + np.random.randint(6, 12))),
                (255, 255, 255),
                -1,
            )
            visible_flag = True

        H, W, C = pixel_mask.shape
        prop = self.image_size / max(H, W)
        pixel_mask = cv2.resize(pixel_mask, (-1, -1), fx=prop, fy=prop)
        pad_width = max((self.image_size - pixel_mask.shape[1]) // 2, 0)
        pad_height = max((self.image_size - pixel_mask.shape[0]) // 2, 0)
        pad_mask = np.pad(
            pixel_mask, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0
        )
        mask = cv2.resize(pad_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask, np.float32) / 255.0
        mask = mask.mean(axis=-1)[:, :, None]
        return np.concatenate((resize_image, mask), axis=-1), visible_flag

    def relative_pose(self, R_base, T_base, R_world, T_world, base_extrinsic):
        R_base = np.matmul(R_base, np.linalg.inv(base_extrinsic[0:3, 0:3]))
        if len(T_world.shape) == 1:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_frame = np.dot(R_world, R_base.T)
            T_frame = np.dot(np.linalg.inv(homo_RT), np.array([*T_world, 1]).T)[0:3]
            T_frame = np.array([T_frame[1], -T_frame[0], T_frame[2]])  # [:T[1],-T[0],T[2]
            return R_frame, T_frame
        else:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_frame = np.dot(R_world, R_base.T)
            T_frame = np.dot(
                np.linalg.inv(homo_RT), np.concatenate((T_world, np.ones((T_world.shape[0], 1))), axis=-1).T
            ).T[:, 0:3]
            T_frame = T_frame[:, [1, 0, 2]]
            T_frame[:, 1] = -T_frame[:, 1]
            return R_frame, T_frame

    def absolute_pose(self, R_base, T_base, R_frame, T_frame, base_extrinsic):
        R_base = np.matmul(R_base, np.linalg.inv(base_extrinsic[0:3, 0:3]))
        if len(T_frame.shape) == 1:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_world = np.dot(R_frame, R_base)
            T_world = np.dot(homo_RT, np.array([-T_frame[1], T_frame[0], T_frame[2], 1]).T)[0:3]
        else:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_world = np.dot(R_frame, R_base)
            T_world = np.dot(
                homo_RT,
                np.concatenate(
                    (np.stack((-T_frame[:, 1], T_frame[:, 0], T_frame[:, 2]), axis=-1), np.ones((T_frame.shape[0], 1))),
                    axis=-1,
                ).T,
            ).T[:, 0:3]
        return R_world, T_world

    def xyz_to_xyt(self, xyz_actions, init_vector):
        xyt_actions = []
        for i in range(0, xyz_actions.shape[0] - 1):
            current_vector = xyz_actions[i + 1] - xyz_actions[i]
            dot_product = np.dot(init_vector[0:2], current_vector[0:2])
            cross_product = np.cross(init_vector[0:2], current_vector[0:2])
            theta = np.arctan2(cross_product, dot_product)
            xyt_actions.append([xyz_actions[i][0], xyz_actions[i][1], theta])
        return np.array(xyt_actions)

    def process_actions(self, extrinsics, base_extrinsic, start_step, end_step, pred_digit=1):
        label_linear_pos = []
        for f_ext in extrinsics[start_step : end_step + 1]:
            R, T = self.relative_pose(
                extrinsics[start_step][0:3, 0:3],
                extrinsics[start_step][0:3, 3],
                f_ext[0:3, 0:3],
                f_ext[0:3, 3],
                base_extrinsic,
            )
            label_linear_pos.append(T)
        label_actions = np.array(label_linear_pos)

        # this is usesd for action augmentations:
        # (1) apply random rotation to the future steps
        # (2) interpolate between the rotated actions and origin actions
        rotate_yaw_angle = np.random.uniform(-np.pi / 3, np.pi / 3)
        rotate_matrix = np.array(
            [
                [np.cos(rotate_yaw_angle), -np.sin(rotate_yaw_angle)],
                [np.sin(rotate_yaw_angle), np.cos(rotate_yaw_angle)],
            ],
            np.float32,
        )
        rotate_local_actions = np.matmul(rotate_matrix, label_actions[:, 0:2].T).T
        rotate_local_actions = np.stack(
            (rotate_local_actions[:, 0], rotate_local_actions[:, 1], np.zeros_like(rotate_local_actions[:, 0])), axis=-1
        )
        rotate_world_points = []
        for act in rotate_local_actions:
            w_rot, w_act = self.absolute_pose(
                extrinsics[start_step, 0:3, 0:3], extrinsics[start_step, 0:3, 3], np.eye(3), act, base_extrinsic
            )
            rotate_world_points.append(w_act)
        rotate_world_points = np.array(rotate_world_points)
        origin_world_points = extrinsics[start_step : end_step + 1, 0:3, 3]
        mix_anchor_points = rotate_world_points

        t = np.linspace(0, 1, mix_anchor_points.shape[0])
        cs_x = CubicSpline(t, mix_anchor_points[:, 0])
        cs_y = CubicSpline(t, mix_anchor_points[:, 1])
        cs_z = CubicSpline(t, mix_anchor_points[:, 2])
        interpolate_nums = origin_world_points.shape[0]
        t_fine = np.linspace(0, 1, int(interpolate_nums))
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        z_fine = cs_z(t_fine)
        result_augment_points = np.stack((x_fine, y_fine, z_fine), axis=-1)
        local_label_points = []
        local_augment_points = []
        for f_ext, g_ext in zip(origin_world_points, result_augment_points):
            Rf, Tf = self.relative_pose(
                extrinsics[start_step][0:3, 0:3], extrinsics[start_step][0:3, 3], np.eye(3), f_ext, base_extrinsic
            )
            Rg, Tg = self.relative_pose(
                extrinsics[start_step][0:3, 0:3], extrinsics[start_step][0:3, 3], np.eye(3), g_ext, base_extrinsic
            )
            local_label_points.append(Tf)
            local_augment_points.append(Tg)
        local_label_points = np.array(local_label_points)
        local_augment_points = np.array(local_augment_points)
        action_indexes = np.clip(np.arange(self.predict_size + 1) * pred_digit, 0, label_actions.shape[0] - 2)
        return local_label_points, local_augment_points, origin_world_points, result_augment_points, action_indexes

    def rank_steps(self, extrinsics, obstacle_points, pred_digit=4):
        points_score = []
        trajectory = extrinsics[:, 0:2, 3]
        # bev_points = obstacle_points[:, 0:2]
        for i in range(0, trajectory.shape[0] - 1):
            future_actions = trajectory[i : min(i + self.predict_size * pred_digit, trajectory.shape[0] - 1)]
            future_bound = [
                np.min(future_actions[:, 0]) - 1,
                np.min(future_actions[:, 1]) - 1,
                np.max(future_actions[:, 0]) + 1,
                np.max(future_actions[:, 1]) + 1,
            ]
            within_bound_points = (
                (obstacle_points[:, 0] > future_bound[0])
                & (obstacle_points[:, 1] > future_bound[1])
                & (obstacle_points[:, 0] < future_bound[2])
                & (obstacle_points[:, 1] < future_bound[3])
            )
            points_score.append(np.sum(within_bound_points))
        points_score = np.array(points_score) / (np.array(points_score).max() + 1e-8)
        probs = np.exp(points_score / 0.2) / np.sum(np.exp(points_score / 0.2))
        start_choice = np.random.choice(np.arange(probs.shape[0]), p=probs)
        target_choice_candidates = np.arange(start_choice + 1, trajectory.shape[0])
        target_choice_p = (target_choice_candidates - start_choice) / (
            (target_choice_candidates - start_choice).max() + 1e-8
        )
        target_choice_p = np.exp(target_choice_p / 0.2) / np.exp(target_choice_p / 0.2).sum()
        target_choice = np.random.choice(target_choice_candidates, p=target_choice_p)
        return start_choice, target_choice

    def __getitem__(self, index):
        import os
        import time

        if self._last_time is None:
            self._last_time = time.time()
        start_time = time.time()

        (
            camera_intrinsic,
            trajectory_base_extrinsic,
            trajectory_extrinsics,
            trajectory_length,
        ) = self.process_data_parquet(index)

        trajectory_path_points, trajectory_path_pcd = self.process_path_points(index)
        trajectory_obstacle_points, trajectory_obstacle_pcd = self.process_obstacle_points(
            index, trajectory_path_points
        )

        if self.prior_sample:
            pixel_start_choice, target_choice = self.rank_steps()
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)
        else:
            pixel_start_choice = np.random.randint(0, trajectory_length // 2)
            target_choice = np.random.randint(pixel_start_choice + 1, trajectory_length - 1)
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)

        # target_extrinsic = trajectory_extrinsics[target_choice]
        if self.random_digit:
            memory_digit = np.random.randint(2, 8)
            pred_digit = memory_digit
        else:
            memory_digit = 4
            pred_digit = 4

        memory_images, depth_image, memory_index = self.process_memory(
            self.trajectory_rgb_path[index],
            self.trajectory_depth_path[index],
            memory_start_choice,
            memory_digit=memory_digit,
        )
        (
            target_local_points,
            augment_local_points,
            target_world_points,
            augment_world_points,
            action_indexes,
        ) = self.process_actions(
            trajectory_extrinsics, trajectory_base_extrinsic, memory_start_choice, target_choice, pred_digit=pred_digit
        )
        # convert the xyz points into xy-theta points
        init_vector = target_local_points[1] - target_local_points[0]
        target_xyt_actions = self.xyz_to_xyt(target_local_points, init_vector)
        augment_xyt_actions = self.xyz_to_xyt(augment_local_points, init_vector)
        # based on the prediction length to decide the final prediction trajectories
        pred_actions = target_xyt_actions[action_indexes]
        augment_actions = augment_xyt_actions[action_indexes]
        if trajectory_obstacle_points.shape[0] != 0:
            pred_distance = (
                np.abs(target_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1)
                .min(axis=-1)
            )
            augment_distance = (
                np.abs(augment_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1)
                .min(axis=-1)
            )
            pred_critic = (
                -5.0 * (pred_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (pred_distance[action_indexes][1:] - pred_distance[action_indexes][:-1]).sum()
            )
            augment_critic = (
                -5.0 * (augment_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (augment_distance[action_indexes][1:] - augment_distance[action_indexes][:-1]).sum()
            )
        else:
            pred_distance = np.ones(pred_actions.shape[0], dtype=np.float32)
            augment_distance = np.ones(pred_actions.shape[0], dtype=np.float32)
            pred_critic = 2.0
            augment_critic = 2.0

        point_goal = target_xyt_actions[-1]
        image_goal = np.concatenate(
            (
                self.process_image(self.trajectory_rgb_path[index][target_choice]),
                self.process_image(self.trajectory_rgb_path[index][memory_start_choice]),
            ),
            axis=-1,
        )

        # process pixel projection
        pixel_target_local_points, _, _, _, _ = self.process_actions(
            trajectory_extrinsics, trajectory_base_extrinsic, pixel_start_choice, target_choice, pred_digit=pred_digit
        )
        pixel_init_vector = pixel_target_local_points[1] - pixel_target_local_points[0]
        pixel_xyt_actions = self.xyz_to_xyt(pixel_target_local_points, pixel_init_vector)
        pixel_goal, pixel_flag = self.process_pixel_goal(
            self.trajectory_rgb_path[index][pixel_start_choice],
            pixel_xyt_actions[-1],
            camera_intrinsic,
            trajectory_base_extrinsic,
        )
        # pixel channel == 7 represents the navdp works pixel navigation under asynchronous pace,
        # pixel_mask (1), the history image with the assigned pixel goal (3), current image (3)
        # if pixel_channel == 4, pixel goal is assigned at current frame, therefore,
        # only pixel_mask (1) and current image (3) are needed
        if self.pixel_channel == 7:
            pixel_goal = np.concatenate((pixel_goal, memory_images[-1]), axis=-1)

        pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
        augment_actions = (augment_actions[1:] - augment_actions[:-1]) * 4.0

        # Summarize avg time of batch
        end_time = time.time()
        self.item_cnt += 1
        self.batch_time_sum += end_time - start_time
        if self.item_cnt % self.batch_size == 0:
            avg_time = self.batch_time_sum / self.batch_size
            print(
                f'__getitem__ pid={os.getpid()}, avg_time(last {self.batch_size})={avg_time:.2f}s, cnt={self.item_cnt}'
            )
            self.batch_time_sum = 0.0
        point_goal = torch.tensor(point_goal, dtype=torch.float32)
        image_goal = torch.tensor(image_goal, dtype=torch.float32)
        pixel_goal = torch.tensor(pixel_goal, dtype=torch.float32)
        memory_images = torch.tensor(memory_images, dtype=torch.float32)
        depth_image = torch.tensor(depth_image, dtype=torch.float32)
        pred_actions = torch.tensor(pred_actions, dtype=torch.float32)
        augment_actions = torch.tensor(augment_actions, dtype=torch.float32)
        pred_critic = torch.tensor(pred_critic, dtype=torch.float32)
        augment_critic = torch.tensor(augment_critic, dtype=torch.float32)
        return (
            point_goal,
            image_goal,
            pixel_goal,
            memory_images,
            depth_image,
            pred_actions,
            augment_actions,
            pred_critic,
            augment_critic,
            float(pixel_flag),
        )


def navdp_collate_fn(batch):

    collated = {
        "batch_pg": torch.stack([item[0] for item in batch]),
        "batch_ig": torch.stack([item[1] for item in batch]),
        "batch_tg": torch.stack([item[2] for item in batch]),
        "batch_rgb": torch.stack([item[3] for item in batch]),
        "batch_depth": torch.stack([item[4] for item in batch]),
        "batch_labels": torch.stack([item[5] for item in batch]),
        "batch_augments": torch.stack([item[6] for item in batch]),
        "batch_label_critic": torch.stack([item[7] for item in batch]),
        "batch_augment_critic": torch.stack([item[8] for item in batch]),
    }
    return collated


if __name__ == "__main__":
    os.makedirs("./navdp_dataset_test/", exist_ok=True)
    dataset = NavDP_Base_Datset(
        "/shared/smartbot_new/liuyu/vln-n1-minival/",
        "./navdp_dataset_test/dataset_lerobot.json",
        8,
        24,
        224,
        trajectory_data_scale=0.1,
        scene_data_scale=0.1,
        preload=False,
    )

    for i in range(10):
        (
            point_goal,
            image_goal,
            pixel_goal,
            memory_images,
            depth_image,
            pred_actions,
            augment_actions,
            pred_critic,
            augment_critic,
            pixel_flag,
        ) = dataset.__getitem__(i)
        if pixel_flag == 1.0:
            pixel_obs = pixel_goal.numpy()[:, :, 0:3] * 255
            pixel_obs[pixel_goal[:, :, 3] == 1] = np.array([0, 0, 255])

            draw_current_image = cv2.cvtColor(image_goal[:, :, 3:6].numpy() * 255, cv2.COLOR_BGR2RGB)
            draw_current_image = cv2.putText(
                draw_current_image, "Current-Image", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            )

            draw_goal_image = cv2.cvtColor(image_goal[:, :, 0:3].numpy() * 255, cv2.COLOR_BGR2RGB)
            draw_goal_image = cv2.putText(
                draw_goal_image, "Image-Goal", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            )

            draw_pixel_image = cv2.cvtColor(pixel_obs.copy(), cv2.COLOR_BGR2RGB)
            draw_pixel_image = cv2.putText(
                draw_pixel_image, "Pixel-Goal", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            )

            goal_info_image = np.concatenate((draw_current_image, draw_goal_image, draw_pixel_image), axis=1)
            goal_info_image = cv2.putText(
                goal_info_image,
                "PointGoal=[{:.3f}, {:.3f}, {:.3f}]".format(*point_goal),
                (190, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
            )
            cv2.imwrite("./navdp_dataset_test/goal_information_%d.png" % i, goal_info_image)
