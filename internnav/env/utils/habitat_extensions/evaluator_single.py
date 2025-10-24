import argparse
import copy
import itertools
import os
import random
import re
import sys
from collections import OrderedDict
from typing import Any

import cv2
import habitat
import numpy as np
import quaternion
import torch
from depth_camera_filtering import filter_depth
from habitat import Env
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor
from transformers.image_utils import to_numpy_array

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT_PATH)
print(f"PROJECT_ROOT_PATH {PROJECT_ROOT_PATH}")
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import (
    chunk_token,
    split_and_clean,
    traj_to_actions,
)
from internnav.utils.dist import get_rank, get_world_size, init_distributed_mode

DEFAULT_IMAGE_TOKEN = "<image>"


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 1,
        output_path: str = None,
        model: Any = None,
        processor: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        # for gradio evaluation
        self.infer_data_ready = False
        self.infer_scene_id = 0
        self.infer_episode_id = 0
        self.infer_success_cnt = -1
        self.infer_instruction = ""
        self.infer_success = False
        self.env = None
        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

        self.model = model
        self.processor = processor

        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history

    def preprocess_depth_image_v2(
        self, depth_image, do_depth_scale=True, depth_scale=1000, target_height=None, target_width=None
    ):
        if target_height is None:
            target_height = self.image_processor.crop_size['height']  # 384
            target_width = self.image_processor.crop_size['width']  # 384

        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)

        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale

        return img, (target_width, target_height)

    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        return intrinsic_matrix

    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):  # (V, 4, 4) (resize_shape) (h, w)
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)

        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic

    def get_axis_align_matrix(self):
        ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        return ma

    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
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

    def xyz_pitch_to_tf_matrix(self, xyz: np.ndarray, pitch: float) -> np.ndarray:
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

    def xyz_yaw_pitch_to_tf_matrix(self, xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
        """Converts a given position and yaw, pitch angles to a 4x4 transformation matrix.

        Args:
            xyz (np.ndarray): A 3D vector representing the position.
            yaw (float): The yaw angle in radians.
            pitch (float): The pitch angle in radians for y axis.
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        x, y, z = xyz
        rot1 = self.xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
        rot2 = self.xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rot1 @ rot2
        transformation_matrix[:3, 3] = xyz
        return transformation_matrix

    def pixel_to_gps(self, pixel, depth, intrinsic, tf_camera_to_episodic):
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
        print("depthhhhhhhhhhhhhh", z)

        x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
        point_camera = np.array([x, y, z, 1.0])

        # Transform to episodic frame
        point_episodic = tf_camera_to_episodic @ point_camera
        point_episodic = point_episodic[:3] / point_episodic[3]

        x = point_episodic[0]
        y = point_episodic[1]

        return (x, y)  # same as habitat gps

    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def run_single_eval(self):

        self.model.eval()
        self.env = self.config_env()
        self.scene_episode_dict = {}
        for episode in self.env.episodes:
            if episode.scene_id not in self.scene_episode_dict:
                self.scene_episode_dict[episode.scene_id] = []
            self.scene_episode_dict[episode.scene_id].append(episode)
        intrinsic_matrix = self.get_intrinsic_matrix(
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        )
        sucs, spls, oss, nes = [], [], [], []

        if True:  # fixme
            scenes_keys = list(sorted(self.scene_episode_dict.keys()))

            self.infer_success = False
            self.infer_data_ready = False
            print('---------------current infer scene:', scenes_keys[self.infer_scene_id])
            selected_scenes = ['17DRP5sb8fy', 'r1Q1Z4BcV1o', 'dhjEzFoUFzH']
            key_name = (
                'data/scene_datasets/mp3d/'
                + selected_scenes[self.infer_scene_id]
                + '/'
                + selected_scenes[self.infer_scene_id]
                + '.glb'
            )
            episodes = self.scene_episode_dict[key_name]
            step_size = len(episodes) // 6
            # episode_id = 0

            episode = episodes[self.infer_episode_id * step_size]

            episode_instruction = (
                self.infer_instruction
            )  # episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
            print("episode start", episode_instruction)

            # episode_id = int(episode.episode_id)
            env = self.env
            env.current_episode = episode
            observations = env.reset()

            agent_state = env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            # agent = ShortestPathFollower(env.sim, 0.25, False)

            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)

            vis_frames = []
            step_id = 0

            if self.save_video:
                os.makedirs(self.output_path, exist_ok=True)
            initial_height = env.sim.get_agent_state().position[1]

            rgb_list = []
            # depth_list = []
            action_seq = []
            # past_key_values = None
            output_ids = None

            goal = None
            action = None
            # look_down_observations = None
            # look_down_id_list = []
            messages = []
            # last_action = None
            local_actions = []

            # begin evaluation main loop
            while not env.episode_over and step_id <= 70:
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                agent_state = env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height  # Habitat GPS makes west negative, so flip y
                camera_position = np.array([x, -y, self._camera_height + height])
                # robot_xy = camera_position[:2]
                # tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw) @ self.get_axis_align_matrix()
                tf_camera_to_episodic = (
                    self.xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
                    @ self.get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')  # raw observation image
                # image_size = image.size  # 640*480
                save_raw_image = image.copy()

                if action == 5:
                    look_down_image = image  # Image.fromarray(look_down_observations['rgb']).convert('RGB')
                    save_raw_image = look_down_image.copy()

                    # rgb_list.append(look_down_image)
                    look_down_depth, resize_shape = self.preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()  # [H, W]
                    # depth clip to 5m
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.args.resize_w, self.args.resize_h))
                    rgb_list.append(image)

                    down_observations = env.step(5)
                    down_observations = env.step(5)

                    look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                    depth = down_observations["depth"]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000
                    look_down_depth, resize_shape = self.preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()  # [H, W]
                    # depth clip to 5m
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    env.step(4)
                    env.step(4)

                info = env.get_metrics()
                current_frame_infer_pixel = False
                if len(action_seq) == 0 and goal is None:  # 只有执行完一次输出的所有 action_seq 才能继续做模型推理
                    if action != 5:
                        sources = copy.deepcopy(self.conversation)
                        if 'objectnav' in self.config_path:
                            sources[0]["value"] = sources[0]["value"].replace(
                                '<instruction>.',
                                random.choice(self.objectnav_instructions).format(
                                    target_object=episode.object_category.replace('_', ' ')
                                ),
                            )
                        else:
                            sources[0]["value"] = sources[0]["value"].replace(
                                '<instruction>.', episode_instruction[:-1]
                            )
                        cur_images = rgb_list[-1:]  # current observation
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        print('history_idddddddd', step_id, history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0
                    else:
                        assert action == 5  # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        # messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]})
                        input_img_id = -1

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    print('sources', step_id, sources)
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    print('step_id', step_id, 'messages:', messages)

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        current_frame_infer_pixel = True
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                        pixel_goal = [int(coord[1]), int(coord[0])]  # switch the goal o

                        goal = self.pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)
                        print('before', goal, depth.shape)
                        goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                        if not env.sim.pathfinder.is_navigable(np.array(goal)):
                            goal = np.array(env.sim.pathfinder.snap_point(np.array(goal)))

                        # look down --> horizontal
                        env.step(4)
                        env.step(4)

                        # action = agent.get_next_action(goal)
                        local_actions = []
                        pixel_values = inputs.pixel_values
                        image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                        with torch.no_grad():
                            traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                        # image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16)
                        # pix_goal_image = copy.copy(image_dp)
                        # images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0)
                        # depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        # pix_goal_depth = copy.copy(depth_dp)
                        # depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0)

                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents)

                        random_choice = np.random.choice(dp_actions.shape[0])
                        if self.args.continuous_traj:
                            action_list = traj_to_actions(dp_actions)
                            if len(action_list) < 8:
                                action_list += [0] * (8 - len(action_list))
                        else:
                            action_list = chunk_token(dp_actions[random_choice])
                        print("first action_list", action_list)
                        local_actions = action_list
                        action = local_actions[0]
                        if action == 0:
                            goal = None
                            output_ids = None
                            action = 2
                            print('conduct a random action 2')
                            observations = env.step(action)
                            step_id += 1
                            messages = []
                            continue

                        print('predicted goal', pixel_goal, goal, flush=True)
                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    if len(local_actions) != 0:
                        action = local_actions.pop(0)
                    else:
                        action = 0
                    forward_action += 1
                    print('forward_action', forward_action, flush=True)
                    if forward_action > 8:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == 0:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0

                if info['top_down_map'] is not None:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if current_frame_infer_pixel:
                        frame = cv2.putText(
                            frame,
                            f"{pixel_goal[1], pixel_goal[0]}",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                        )
                        frame = cv2.circle(frame, (pixel_goal[1], pixel_goal[0]), 5, (255, 0, 0), -1)
                    else:
                        output_str = str(action)
                        output_str = (
                            output_str.replace('1', "Go forward").replace('2', 'Turn left').replace('3', 'Turn right')
                        )
                        output_str = output_str.replace('5', 'Look down').replace('0', 'Stop!')
                        frame = cv2.putText(frame, output_str, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    vis_frames.append(frame)
                    if action == 5:
                        vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if action == 5:
                    env.step(action)
                    observations = env.step(action)
                else:
                    observations = env.step(action)
                    step_id += 1
                    messages = []

            self.infer_success_cnt += 1

            # metrics = env.get_metrics()
            if self.save_video:
                images_to_video(vis_frames, self.output_path, f"res_{self.infer_success_cnt}", fps=6, quality=9)
            self.infer_success = True
            vis_frames.clear()

        env.close()
        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def preprocess_qwenvl(self, source):
        prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
        if len(source[0]["value"]) != 0:
            source[0]["value"] += f" {prompt}."
        else:
            source[0]["value"] = f"{prompt}."  # Please output the next waypoint\'s coordinates in the image."
        return source


def eval():
    global local_rank

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='scripts/eval/configs/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./exps_pix/val_unseen/debug_coord_wm')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--predict_step_nums", type=int, default=16)
    parser.add_argument("--continuous_traj", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--port', default='2333')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank
    np.random.seed(local_rank)
    # Load model and tokenizer
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = 'left'

    device = torch.device(f"cuda:{local_rank}")
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": device}
    )

    # start the evaluation
    evaluate(model, processor, args)


def evaluate(model, processor, args):
    model.eval()
    world_size = get_world_size()
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        processor=processor,
        epoch=0,
        args=args,
    )
    sucs, spls, oss, nes, ep_num = evaluator.eval_action(idx=get_rank())


if __name__ == "__main__":
    eval()
    print('hello world!!')
    sys.exit(0)
