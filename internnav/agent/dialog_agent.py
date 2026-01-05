import argparse
import copy
import itertools
import random
import re
import time
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import quaternion
import torch
from PIL import Image, ImageDraw

from internnav.agent import Agent
from internnav.configs.agent import AgentCfg

try:
    from depth_camera_filtering import filter_depth
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
    )
except Exception as e:
    print(f"Warning: ({e}), Habitat Evaluation is not loaded in this runtime. Ignore this if not using Habitat.")

DEFAULT_IMAGE_TOKEN = "<image>"


def split_and_clean(text):
    import re

    parts = re.split(r'(<image>)', text)
    results = []
    for part in parts:
        if part == '<image>':
            results.append(part)
        else:
            clean_part = part.replace('\n', '').strip()
            if clean_part:
                results.append(clean_part)
    return results


@Agent.register('dialog')
class DialogAgent(Agent):
    """Vision-language navigation agent that can either move or ask an oracle via dialog. The agent builds a multimodal
     chat prompt from current/historical RGB observations (and optional look-down frames), runs a Qwen2.5-VL model to
     produce either an action sequence, a pixel waypoint, or a dialog query, then converts the model output into
     simulator actions and (optionally) a world-space navigation goal.

    Args:
        agent_config (AgentCfg): AgentCfg containing model_settings (e.g., task name, sensor config, model path, mode,
            resizing, dialog flags, and generation parameters) and runtime info such as local_rank.
    """

    def __init__(self, agent_config: AgentCfg):
        self.agent_config = agent_config
        self.task_name = self.agent_config.model_settings['task_name']

        # sensor config
        self.sim_sensors_config = self.agent_config.model_settings['sim_sensors_config']
        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth
        self._camera_fov = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(self._camera_fov / 2))

        # model
        self.model_args = argparse.Namespace(**self.agent_config.model_settings)

        self.task = self.model_args.task
        self.append_look_down = self.model_args.append_look_down
        self.resize_h = self.model_args.resize_h
        self.resize_w = self.model_args.resize_w

        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_path, use_fast=True)
        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer = tokenizer
        processor.tokenizer.padding_side = 'left'

        self.device = torch.device('cuda', self.agent_config.model_settings['local_rank'])
        if self.model_args.mode == 'dual_system':
            raise NotImplementedError("Dual System mode is not supported in DialogAgent.")
        elif self.model_args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": self.device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()

        self.model = model
        self.processor = processor
        self.num_history = self.model_args.num_history

        # prompt
        if 'dialog' in self.task_name or self.agent_config.model_settings['dialog_enabled']:
            prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? There is an oracle can help you to complete the task in current environment, you can either choose to move or talk. If choosing to talk, please say something that can help you better to find the target object. If choosing to move, when you want to output a waypoint you need to TILT DOWN (↓) by 30 degrees then output the next waypoint\'s coordinates in the image. In case the next waypoint is out of view, utilize the turn actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees. Please output STOP when you have successfully completed the task."
        else:
            prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? When you want to output a waypoint you need to TILT DOWN (↓) by 30 degrees then output the next waypoint\'s coordinates in the image. In case the next waypoint is out of view, utilize the turn actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees. Please output STOP when you have successfully completed the task."
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

    def convert_input(self, obs, info):
        # update new information after env.step
        depth = obs["depth"]
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        depth = depth * (self._max_depth - self._min_depth) + self._min_depth
        self.depth = depth * 1000  # get depth

        rgb = obs["rgb"]
        image = Image.fromarray(rgb).convert('RGB')  # raw observation image
        self.save_raw_image = image.copy()  # get rgb

        x, y = obs["gps"]
        camera_yaw = obs["compass"][0]
        agent_state = info['agent state']
        height = agent_state.position[1] - self.initial_height  # Habitat GPS makes west negative, so flip y
        camera_position = np.array([x, -y, self._camera_height + height])
        self.tf_camera_to_episodic = (
            self.xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30)) @ self.get_axis_align_matrix()
        )  # get transformation from camera to agent

        # if last action is look down, save the newest look down image for later pixel selection
        if self.last_action == 5:
            self.look_down_image = image
            self.save_raw_image = self.look_down_image.copy()
        elif self.last_action != 6:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
        return obs

    def convert_output(self, env, llm_outputs: str):
        if '<talk>' in llm_outputs:
            self.question = llm_outputs.replace('<talk>', '')
            return 6
        else:
            if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                # get pixel goal
                self.forward_action = 0
                coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                print('coords:', coord)
                try:
                    pixel_goal = [int(coord[1]), int(coord[0])]  # switch the goal o
                except Exception as e:
                    print(f"Invalid pixel goal: len(coor)!=2: {e}")
                    return 0

                # trans pixel goal to global goal
                try:
                    self.goal = self.pixel_to_gps(
                        pixel_goal, self.depth / 1000, self.intrinsic_matrix, self.tf_camera_to_episodic
                    )
                except Exception as e:
                    print(f"Invalid pixel goal: out of image size range: {e}")
                    return 0
                self.goal = (self.transformation_matrix @ np.array([-self.goal[1], 0, -self.goal[0], 1]))[:3]
                if not env._env.sim.pathfinder.is_navigable(np.array(self.goal)):
                    self.goal = np.array(env._env.sim.pathfinder.snap_point(np.array(self.goal)))

                # paint pixel goal
                draw = ImageDraw.Draw(self.save_raw_image, 'RGB')
                x, y, r = pixel_goal[0], pixel_goal[1], 2
                draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(255, 0, 0))

                # look down --> horizontal
                env.step(4)
                env.step(4)

                if self.append_look_down and self.look_down_image is not None:
                    self.prev_look_image = self.look_down_image.resize((self.resize_w, self.resize_h))
                action = self.agent.get_next_action(self.goal)
                if action == 0:
                    self.goal = None
                    self.messages = []
                    print('conduct a random action 2')
                    self.last_action = 2
                    return 2
                print('predicted goal', pixel_goal, self.goal, flush=True)
            else:
                self.action_seq = self.parse_actions(llm_outputs)
                print('actions', self.action_seq, flush=True)

    def inference(self, obs, info):
        if self.last_action == 6:
            self.dialogs.append({'role': 'navigator', 'message': self.question, 'true_idx': self.step_id})
            self.dialogs.append({'role': 'oracle', 'message': obs['npc_answer'], 'true_idx': self.step_id})
            self.messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': self.last_llm_outputs}]})
            self.messages.append({'role': 'user', 'content': [{'type': 'text', 'text': obs['npc_answer']}]})
        elif self.last_action == 5:
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.input_images += [self.look_down_image]
            self.messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': self.last_llm_outputs}]})
            input_img_id = -1
        else:
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>', info['episode_instruction'])
            cur_images = self.rgb_list[-1:]  # current observation
            if self.step_id == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.step_id - 1, self.num_history, dtype=np.int32)).tolist()
                # add dialod history
                dialogs_idx = np.sort(list(set([i['true_idx'] for i in self.dialogs]))).tolist()
                history_id = np.sort(np.unique(np.concatenate([history_id, dialogs_idx]).astype(np.int32))).tolist()
                placeholder = [''] * (len(history_id) + 1)
                for n in dialogs_idx:
                    pos = history_id.index(n)
                    output = ""
                    for dialog in self.dialogs:
                        if dialog['true_idx'] == n:
                            output += f"<|{dialog['role']}|>{dialog['message']}"
                    placeholder[pos + 1] = "<|dialog_start|>" + output + "<|dialog_end|>"
                # add image history
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n').join(placeholder)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'
                if self.append_look_down:
                    if self.prev_look_image is not None:
                        sources[0]["value"] += f' Your previous look down image is:{DEFAULT_IMAGE_TOKEN}.'
                    else:
                        sources[0]["value"] += ' Your previous look down image is not here.'
            history_id = sorted(history_id)
            print('history_id', self.step_id, history_id)
            # prepare images
            if self.append_look_down:
                if self.prev_look_image is not None:
                    self.input_images = [self.rgb_list[i] for i in history_id] + [self.prev_look_image] + cur_images
                else:
                    self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            else:
                self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0

        if self.last_action != 6:
            # prompt text
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            sources[0]["value"] += f" {prompt}."
            prompt_instruction = copy.deepcopy(sources[0]["value"])

            # prompt images
            parts = split_and_clean(prompt_instruction)
            content = []
            for i in range(len(parts)):
                if parts[i] == "<image>":
                    content.append({"type": "image", "image": self.input_images[input_img_id]})
                    input_img_id += 1
                else:
                    content.append({"type": "text", "text": parts[i]})

            self.messages.append({'role': 'user', 'content': content})
        # inference
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        print('step_id', self.step_id, ' ', text)
        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.agent_config.model_settings['max_new_tokens'], do_sample=False
            )
        llm_outputs = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print('step_id:', self.step_id, 'output text:', llm_outputs)
        return llm_outputs

    def step(self, obs: Dict[str, Any], env, info):
        print(f'{self.agent_config.model_name} Agent step')
        start = time.time()
        # convert obs to model input
        self.step_id = info['step']
        obs = self.convert_input(obs, info)
        if len(self.action_seq) == 0 and self.goal is None:
            llm_outputs = self.inference(obs, info)
            self.last_llm_outputs = llm_outputs
            action = self.convert_output(env, llm_outputs)
            with open(info['output_path'], 'a') as f:
                f.write(str(self.step_id) + " " + llm_outputs + "\n")
        else:
            action = None

        if action is None:
            if len(self.action_seq) != 0:
                action = self.action_seq.pop(0)
            elif self.goal is not None:
                action = self.agent.get_next_action(self.goal)
                action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                action = action[0] if hasattr(action, "__len__") else action

                self.forward_action += 1
                print('forward_action', self.forward_action, flush=True)
                if self.forward_action > 8:
                    self.goal = None
                    self.messages = []
                    self.forward_action = 0
                    end = time.time()
                    print(f'time: {round(end-start, 4)}s')
                    # return a meaningless action to do nothing
                    return 7
                if action == 0:
                    self.goal = None
                    self.messages = []
                    end = time.time()
                    print(f'time: {round(end-start, 4)}s')
                    # return a meaningless action to do nothing
                    return 7
            else:
                action = 0

        end = time.time()
        print(f'time: {round(end-start, 4)}s')
        self.last_action = action
        return action

    def reset(self, env):
        self.intrinsic_matrix = self.get_intrinsic_matrix(self.sim_sensors_config.rgb_sensor)
        self.agent = ShortestPathFollower(env._env.sim, 0.25, False)

        # params saving and initialization
        agent_state = env._env.sim.get_agent_state()
        rotation_matrix = quaternion.as_rotation_matrix(agent_state.rotation)
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3, :3] = rotation_matrix
        self.transformation_matrix[:3, 3] = agent_state.position  # get transformation from world to agent
        self.initial_height = agent_state.position[1]  # get initial height

        self.last_action = None
        self.messages = []
        self.rgb_list = []
        self.action_seq = []
        self.goal = None
        self.prev_look_image = None
        self.look_down_image = None  # params for qwen model

        self.dialogs = []

        # params for saving
        self.save_raw_image = None

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
        """Back-project a 2D image pixel into 3D using the depth map and camera intrinsics.

        Args:
            pixel (Tuple[int, int] | List[int] | np.ndarray): pixel coordinate in (v, u) indexing as used here.
            depth (np.ndarray): depth image of shape (H, W) in meters, where depth[v, u] returns the metric depth.
            intrinsic (np.ndarray): camera intrinsic matrix.
            tf_camera_to_episodic (np.ndarray): homogeneous transform of shape (4, 4) mapping camera-frame points to
                the episodic frame.

        Returns:
            Tuple[float, float]: coordinates in the episodic frame.
        """
        v, u = pixel
        z = depth[v, u]
        print("depth", z)

        x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
        point_camera = np.array([x, y, z, 1.0])

        # Transform to episodic frame
        point_episodic = tf_camera_to_episodic @ point_camera
        point_episodic = point_episodic[:3] / point_episodic[3]

        x = point_episodic[0]
        y = point_episodic[1]

        return (x, y)  # same as habitat gps

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)
