import argparse
import itertools
import json
import os
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers.image_utils import to_numpy_array

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.model.utils.vln_utils import open_image

try:
    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat_vlln_extensions.utils.dialog_utils import (
        get_config,
        get_path_description_,
    )

    # Import for Habitat registry side effects — do not remove
    import internnav.habitat_vlln_extensions.measures  # noqa: F401
    from internnav.habitat_vlln_extensions.simple_npc.simple_npc import SimpleNPC

    # isort: skip
except Exception as e:
    print(f"Warning: ({e}), Habitat Evaluation is not loaded in this runtime. Ignore this if not using Habitat.")

DEFAULT_IMAGE_TOKEN = "<image>"


@Evaluator.register('habitat_dialog')
class HabitatDialogEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.scene_summary = args.scene_summary
        self.output_path = args.output_path

        self.task = cfg.task.task_name
        self.turn = args.turn
        self.dialog_enabled = cfg.agent.model_settings['dialog_enabled']
        self.save_video = args.save_video

        self.npc = SimpleNPC(
            max_interaction_turn=10,
            model_name=args.model_name,
            openai_api_key=args.openai_api_key,
            base_url=args.base_url,
        )

        # create habitat config
        self.config_path = cfg.env.env_settings['habitat_config_path']
        self.config = get_config(self.config_path, cfg.env.env_settings['baseline_config_path'])

        with habitat.config.read_write(self.config):
            self.config.exp.task = self.task
            self.config.habitat.dataset.split = args.eval_split
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
        cfg.env.env_settings['habitat_config'] = self.config.habitat
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        cfg.agent.model_settings['task'] = self.task
        cfg.agent.model_settings['sim_sensors_config'] = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.objectnav_instruction = "search for {target_object}."
        super().__init__(cfg)

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        sucs, spls, oss, nes = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, 'result.json')):
            with open(os.path.join(self.output_path, 'result.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
        env = self.env

        while env.is_running:
            obs = env.reset()
            if not env.is_running or obs is None:
                break

            # recover from last evaluated episode
            episode = env._env.current_episode
            scene_id = episode.scene_id.split('/')[-2]
            if 'coin' in self.task:
                episode_instruction = (
                    self.objectnav_instruction.format(target_object=episode.object_category.replace('_', ' '))
                    + ", "
                    + episode.instruction
                )
            elif 'objectnav' in self.task:
                episode_instruction = self.objectnav_instruction.format(
                    target_object=episode.object_category.replace('_', ' ')
                )
            else:
                episode_instruction = episode.instruction.instruction_text[:-1]
            episode_id = int(episode.episode_id)
            if [scene_id, episode_id, episode_instruction] in done_res:
                continue
            # make directories
            os.makedirs(os.path.join(self.output_path, 'check_sim'), exist_ok=True)
            Image.fromarray(obs['rgb']).save(os.path.join(self.output_path, 'check_sim', f'rgb_{self.rank}.jpg'))
            os.makedirs(os.path.join(self.output_path, 'action', f'{scene_id}'), exist_ok=True)
            # os.makedirs(os.path.join(self.output_path, 'debug_images'), exist_ok=True)

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, 'vis', f'{scene_id}'), exist_ok=True)

            # get agent ready
            self.agent.reset(env)

            # info for npc
            if 'dialog' in self.task or self.dialog_enabled:  # gt of env for npc
                with open(os.path.join(self.scene_summary, scene_id, 'object_dict.json'), 'r', encoding='utf-8') as f:
                    object_dict = json.load(f)
                with open(os.path.join(self.scene_summary, scene_id, 'region_dict.json'), 'r', encoding='utf-8') as f:
                    region_dict = json.load(f)

            # initialization
            step_id = 0

            path_list = []
            action_list = []  # params for saving results

            while not env._env.episode_over and step_id <= self.max_steps_per_episode:
                agent_state = env._env.sim.get_agent_state()
                path_list.append(agent_state.position.tolist())
                info = {
                    'step': step_id,
                    'agent state': agent_state,
                    'episode_instruction': episode_instruction,
                    'output_path': os.path.join(self.output_path, 'action', f'{scene_id}', f'{episode_id}.txt'),
                    'info': env.get_metrics(),
                }
                action = self.agent.step(obs, env, info=info)
                print("step_id", step_id, "action", action)
                action_list.append(action)
                if action in [0, 1, 2, 3]:
                    obs, reward, done, info = env.step(action)
                elif action == 5:
                    env.step(action)
                    obs, reward, done, info = env.step(action)
                    continue
                elif action == 6:
                    if len(self.agent.dialogs) / 2 >= self.turn:
                        npc_answer = 'Sorry, you have reached the question limit. No further answers are available.'
                    else:
                        path_description, pl = get_path_description_(env._env, object_dict, region_dict)
                        task_finish = obs['semantic'][0].sum() > 0 and pl < 3
                        npc_answer = self.npc.answer_question(
                            question=self.agent.question,
                            instance_id=env._env.current_episode.instruction.instance_id[0],
                            object_dict=object_dict,
                            task_done=task_finish,
                            path_description=path_description,
                            mode="two_turn",
                        )
                    if npc_answer is None:
                        npc_answer = 'Sorry, I can not answer your question now.'

                    with open(os.path.join(self.output_path, 'action', f'{scene_id}', f'{episode_id}.txt'), 'a') as f:
                        f.write(npc_answer + "\n")
                    obs['npc_answer'] = npc_answer
                    continue

                step_id += 1
                self.agent.messages = []

            m = env.get_metrics()
            sucs.append(m["success"])
            spls.append(m["spl"])
            oss.append(m["oracle_success"])
            nes.append(m["distance_to_goal"])
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": m["success"],
                "spl": m["spl"],
                "os": m['oracle_success'],
                "ne": m["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
                "path": path_list,
                "action": action_list,
                "object_category": episode.object_category if 'vln' not in self.task else '',
            }
            with open(os.path.join(self.output_path, 'result.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

        env.close()
        return {
            "sucs": torch.tensor(sucs).to(self.agent.device),  # shape [N_local]
            "spls": torch.tensor(spls).to(self.agent.device),  # shape [N_local]
            "oss": torch.tensor(oss).to(self.agent.device),  # shape [N_local]
            "nes": torch.tensor(nes).to(self.agent.device),  # shape [N_local]
        }

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        global_metrics["sucs"] etc. are global 1-D CPU tensors with all episodes.
        """
        sucs_all = global_metrics["sucs"]
        spls_all = global_metrics["spls"]
        oss_all = global_metrics["oss"]
        nes_all = global_metrics["nes"]

        # avoid /0 if no episodes
        denom = max(len(sucs_all), 1)

        return {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

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

    def dot_matrix_two_dimensional(
        self,
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
