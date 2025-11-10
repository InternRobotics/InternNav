import argparse
import json
import os
import sys

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict

import habitat
import numpy as np
import quaternion
import torch
import tqdm
from depth_camera_filtering import filter_depth
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.image_utils import to_numpy_array

# Import for Habitat registry side effects — do not remove
import internnav.env.utils.habitat_extensions.measures  # noqa: F401
from internnav.configs.evaluator import EvalCfg
from internnav.evaluator.base import Evaluator
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import (
    chunk_token,
    open_image,
    split_and_clean,
    traj_to_actions,
)
from internnav.utils.dist import dist, get_rank, get_world_size, init_distributed_mode

try:
    from habitat import Env
    from habitat.config.default import get_agent_config
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from habitat.utils.visualizations.utils import (
        images_to_video,
        observations_to_image,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config
except Exception as e:
    print("Habitat Error:", e)
    print("Habitat Evaluation is not loaded.")


DEFAULT_IMAGE_TOKEN = "<image>"


@Evaluator.register('habitat_vln')
class HabitatVlnEvaluator(Evaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.args = args
        self.save_video = args.save_video

        # distributed setting
        import os
        import socket

        print(
            f"Rank {os.getenv('RANK')} / {os.getenv('WORLD_SIZE')} on {socket.gethostname()}:{os.getenv('MASTER_PORT')}"
        )
        init_distributed_mode(args)
        local_rank = args.local_rank
        np.random.seed(local_rank)
        cfg.env.env_settings['idx'] = get_rank()
        cfg.env.env_settings['world_size'] = get_world_size()

        self.world_size = get_world_size()
        self.output_path = args.output_path  # TODO: modify by rank
        self.epoch = 0

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
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
        cfg.env.env_settings['habitat_config'] = self.config

        # init agent and env
        # super().__init__(cfg)

        # ------------------------------------- model ------------------------------------------
        processor = AutoProcessor.from_pretrained(args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:{local_rank}")
        if args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        elif args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

        model.eval()
        self.device = device

        # ------------------------------------- old ------------------------------------------
        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

        self.model = model
        self.processor = processor

        # refactor: this part used in three places
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

        self.objectnav_instructions = ["Search for the {target_object}."]

        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history
        # ------------------------------------- remove ------------------------------------------

    def eval(self):
        # * 3. do eval
        sucs, spls, oss, nes, ep_num = self.eval_action(self.args.local_rank)
        ep_num_all = [torch.zeros_like(ep_num) for _ in range(self.world_size)]
        # import ipdb; ipdb.set_trace()
        world_size = get_world_size()
        dist.all_gather(ep_num_all, ep_num)
        sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
        spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
        oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
        nes_all = [torch.zeros(ep_num_all[i], dtype=nes.dtype).to(nes.device) for i in range(world_size)]
        dist.barrier()
        dist.all_gather(sucs_all, sucs)
        dist.all_gather(spls_all, spls)
        dist.all_gather(oss_all, oss)
        dist.all_gather(nes_all, nes)

        sucs_all = torch.cat(sucs_all, dim=0)
        spls_all = torch.cat(spls_all, dim=0)
        oss_all = torch.cat(oss_all, dim=0)
        nes_all = torch.cat(nes_all, dim=0)
        result_all = {
            "sucs_all": (sum(sucs_all) / len(sucs_all)).item(),
            "spls_all": (sum(spls_all) / len(spls_all)).item(),
            "oss_all": (sum(oss_all) / len(oss_all)).item(),
            "nes_all": (sum(nes_all) / len(nes_all)).item(),
            'length': len(sucs_all),
        }

        print(result_all)
        if get_rank() == 0:
            with open(os.path.join(self.args.output_path, 'result.json'), 'a') as f:
                f.write(json.dumps(result_all))

    def _eval_action(self):
        obs = self.env.reset()
        action = self.agent.reset()
        while not self.env.is_running():
            action = self.agent.step(action, obs)
            obs, terminated = self.env.step(action)
            if terminated:
                obs = self.env.reset()
                self.agent.reset()
                self.env.update_metric()

    # refactor
    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def eval_action(self, idx=0) -> None:  # noqa: C901
        self.model.eval()
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        intrinsic_matrix = self.get_intrinsic_matrix(
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        )
        sucs, spls, oss, nes = [], [], [], []
        done_res = []

        if os.path.exists(os.path.join(self.output_path, 'result.json')):
            with open(os.path.join(self.output_path, 'result.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:  # noqa: F405 TODO this need to keep in evaluator
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        nes.append(res['ne'])

        # refactor: sort to scene: [episode] but nothing actually used
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes[idx :: self.world_size])), desc=f"scene {scene_id}")
            for episode in episodes[idx :: self.world_size]:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if 'objectnav' not in self.config_path
                    else episode.object_category
                )
                print("episode start", episode_instruction)
                episode_id = int(episode.episode_id)
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                # refactor env warm up
                env.current_episode = episode
                observations = env.reset()

                agent_state = env.sim.get_agent_state()
                rotation = agent_state.rotation
                translation = agent_state.position
                rotation_matrix = quaternion.as_rotation_matrix(rotation)
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = translation

                agent = ShortestPathFollower(env.sim, 0.25, False)

                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(
                    os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg')
                )

                vis_frames = []
                step_id = 0

                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                action_seq = []
                output_ids = None

                goal = None
                action = None
                messages = []
                local_actions = []

                while not env.episode_over and step_id <= 500:
                    # refactor agent get action
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height
                    camera_position = np.array([x, -y, self._camera_height + height])
                    tf_camera_to_episodic = (
                        self.xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
                        @ self.get_axis_align_matrix()
                    )

                    image = Image.fromarray(rgb).convert('RGB')
                    save_raw_image = image.copy()

                    save_dot = False
                    if action == 5:
                        look_down_image = image
                        save_raw_image = look_down_image.copy()
                        look_down_depth, resize_shape = self.preprocess_depth_image_v2(
                            Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                            do_depth_scale=True,
                            depth_scale=1000,
                            target_height=224,
                            target_width=224,
                        )
                        look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                        look_down_depth[look_down_depth > 5.0] = 5.0
                    else:
                        image = image.resize((self.args.resize_w, self.args.resize_h))
                        rgb_list.append(image)

                        if self.args.mode == 'dual_system':
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
                            look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                            look_down_depth[look_down_depth > 5.0] = 5.0

                            env.step(4)
                            env.step(4)

                    info = env.get_metrics()

                    if len(action_seq) == 0 and goal is None:
                        if action != 5:
                            sources = copy.deepcopy(self.conversation)
                            sources[0]["value"] = sources[0]["value"].replace(
                                '<instruction>.', episode.instruction.instruction_text[:-1]
                            )
                            cur_images = rgb_list[-1:]
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
                            assert action == 5
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            input_images += [look_down_image]
                            # messages.append(
                            #     {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                            # )
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

                        inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(
                            self.model.device
                        )

                        with torch.no_grad():
                            output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

                        llm_outputs = self.processor.tokenizer.decode(
                            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                        )
                        print('step_id:', step_id, 'output text:', llm_outputs)

                        if bool(re.search(r'\d', llm_outputs)):
                            forward_action = 0
                            coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                            pixel_goal = [int(coord[1]), int(coord[0])]

                            goal = self.pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)
                            print('before', goal, depth.shape)
                            goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                            if not env.sim.pathfinder.is_navigable(np.array(goal)):
                                goal = np.array(env.sim.pathfinder.snap_point(np.array(goal)))

                            # look down --> horizontal
                            env.step(4)
                            env.step(4)

                            # Forking logic based on mode
                            if self.args.mode == 'system2':
                                action = agent.get_next_action(goal)
                                if action == 0:
                                    goal = None
                                    output_ids = None
                                    action = 2  # random action
                                    print('conduct a random action 2')
                                    observations = env.step(action)
                                    step_id += 1
                                    messages = []
                                    continue
                            else:  # dual-system logic
                                local_actions = []
                                pixel_values = inputs.pixel_values
                                image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                                with torch.no_grad():
                                    traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                                # prepocess align with navdp
                                image_dp = (
                                    torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                                )
                                pix_goal_image = copy.copy(image_dp)
                                images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                                depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                                pix_goal_depth = copy.copy(depth_dp)
                                depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                                with torch.no_grad():
                                    dp_actions = self.model.generate_traj(
                                        traj_latents, images_dp, depths_dp, use_async=True
                                    )

                                random_choice = np.random.choice(dp_actions.shape[0])
                                if self.args.continuous_traj:
                                    action_list = traj_to_actions(dp_actions)
                                    if len(action_list) < 8:
                                        action_list += [0] * (8 - len(action_list))
                                else:
                                    action_list = chunk_token(dp_actions[random_choice])

                                local_actions = action_list
                                if len(local_actions) >= 4:
                                    local_actions = local_actions[:4]
                                action = local_actions[0]
                                if action == 0:
                                    goal = None
                                    output_ids = None
                                    action = 2  # random action
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
                        # Forking logic based on mode
                        if self.args.mode == 'system2':
                            action = agent.get_next_action(goal)
                            action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                            action = action[0] if hasattr(action, "__len__") else action
                        else:  # dual-system logic
                            if len(local_actions) == 0:
                                # navdp
                                local_actions = []
                                image_dp = (
                                    torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                                )

                                images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                                depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                                depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    dp_actions = self.model.generate_traj(
                                        traj_latents, images_dp, depths_dp, use_async=True
                                    )

                                random_choice = np.random.choice(dp_actions.shape[0])
                                if self.args.continuous_traj:
                                    action_list = traj_to_actions(dp_actions)
                                    if len(action_list) < 8:
                                        action_list += [0] * (8 - len(action_list))
                                else:
                                    action_list = chunk_token(dp_actions[random_choice])
                                print("first action_list", action_list)

                                local_actions = action_list
                                if len(local_actions) >= 4:
                                    local_actions = local_actions[:4]
                                # if len(local_actions) >= 2:
                                #     local_actions = local_actions[:2]

                                print("local_actions", local_actions)

                                action = local_actions.pop(0)
                                # navdp
                            else:
                                action = local_actions.pop(0)

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
                        if save_dot:
                            save_raw_image = self.dot_matrix_two_dimensional(
                                save_raw_image, save_img=False, save_path=f'test_{step_id}.jpg', pixel_goal=pixel_goal
                            )
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                        vis_frames.append(frame)

                    print("step_id", step_id, "action", action)

                    # refactor: core
                    if action == 5:
                        env.step(action)
                        observations = env.step(action)
                    else:
                        observations = env.step(action)
                        step_id += 1
                        messages = []

                process_bar.update(1)

                metrics = env.get_metrics()
                if self.save_video:
                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                        f'{episode_id:04d}',
                        fps=6,
                        quality=9,
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                nes.append(metrics["distance_to_goal"])
                print(
                    f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}"
                )

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                }

                with open(os.path.join(self.output_path, 'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")
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
