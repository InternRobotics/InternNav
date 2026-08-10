import argparse
import json
import os
import sys
from enum import IntEnum

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
import time
from collections import OrderedDict

import cv2
import habitat
import imageio
import numpy as np
import quaternion
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.evaluator.utils.diagnostic_logger import (
    DiagnosticLogger,
    depth_statistics,
    seed_everything,
    stable_episode_seed,
)
from internnav.habitat_extensions.vln.utils import (
    get_axis_align_matrix,
    get_intrinsic_matrix,
    pixel_to_gps,
    preprocess_depth_image_v2,
    xyz_yaw_pitch_to_tf_matrix,
)
from internnav.habitat_extensions.vln.recovery_controller import RecoveryController
from internnav.habitat_extensions.vln.trajectory_selector import TrajectorySelector
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions

# Import for Habitat registry side effects — do not remove
import internnav.habitat_extensions.vln.measures  # noqa: F401 # isort: skip


DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


def format_s2_output_for_overlay(output):
    """Render symbolic S2 actions with OpenCV-safe ASCII labels."""
    output = str(output).strip()
    arrow_names = {
        '\u2191': 'FORWARD',
        '\u2190': 'LEFT',
        '\u2192': 'RIGHT',
        '\u2193': 'LOOKDOWN',
    }
    if output and all(character in arrow_names for character in output):
        groups = []
        for character, items in itertools.groupby(output):
            count = sum(1 for _ in items)
            label = arrow_names[character]
            groups.append(f'{label} x{count}' if count > 1 else label)
        return ' | '.join(groups)

    normalized = ''.join(arrow_names.get(character, character) for character in output)
    return normalized.encode('ascii', errors='replace').decode('ascii')


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


@Evaluator.register('habitat_vln')
class HabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path
        self.base_seed = int(getattr(args, 'seed', 42))
        self.diagnostic_enabled = bool(getattr(args, 'diagnostic_log', False))
        self.diagnostic_dir = getattr(args, 'diagnostic_dir', os.path.join(self.output_path, 'diagnostics'))
        self.diagnostic_criteria = getattr(args, 'diagnostic_criteria', None)
        self.minimal_s1_safety = bool(getattr(args, 'minimal_s1_safety', False))
        self.trajectory_selector = TrajectorySelector(
            cluster_min_fraction=float(getattr(args, 's1_cluster_min_fraction', 0.25)),
            cluster_lateral_gap=float(getattr(args, 's1_cluster_lateral_gap', 0.25)),
            depth_lookahead=float(getattr(args, 's1_depth_lookahead', 1.25)),
            unsafe_clearance=float(getattr(args, 's1_unsafe_clearance', 0.12)),
            near_clearance=float(getattr(args, 's1_near_clearance', 0.35)),
            horizontal_fov_deg=float(getattr(args, 's1_horizontal_fov_deg', 79.0)),
        )
        self.recovery_controller = RecoveryController(
            min_forward_displacement=float(getattr(args, 's1_min_forward_displacement', 0.03)),
            system2_retry_limit=int(getattr(args, 's1_system2_retry_limit', 3)),
            max_consecutive_turns=(
                int(getattr(args, 's1_max_consecutive_turns', 24)) if self.minimal_s1_safety else 0
            ),
        )

        # This happens before model construction. The launch script additionally
        # sets PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG before Python starts.
        seed_everything(self.base_seed)

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            if hasattr(self.config.habitat, 'seed'):
                self.config.habitat.seed = self.base_seed
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
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        super().__init__(cfg, init_agent=False)

        # DistributedEvaluator seeds NumPy by rank, so restore the requested
        # experiment seed before constructing or invoking either model.
        seed_everything(self.base_seed)

        # ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)
        self.vis_debug = bool(getattr(self.model_args, "vis_debug", False) or self.save_video)
        self.vis_debug_path = getattr(self.model_args, "vis_debug_path", os.path.join(self.output_path, "vis_debug"))

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:{self.local_rank}")
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        elif self.model_args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor

        self.diagnostic_logger = None
        if self.diagnostic_enabled:
            self.diagnostic_logger = DiagnosticLogger(
                os.path.join(self.diagnostic_dir, f'rank_{self.rank}'),
                run_metadata={
                    'run_id': getattr(args, 'run_id', os.environ.get('DUALVLN_RUN_ID', 'diagnostic')),
                    'base_seed': self.base_seed,
                    'rank': self.rank,
                    'local_rank': self.local_rank,
                    'mode': self.model_args.mode,
                    'model_path': self.model_args.model_path,
                    'habitat_config': self.config_path,
                    'code_commit': os.environ.get('DUALVLN_CODE_COMMIT'),
                    'dataset_id': getattr(args, 'dataset_id', None),
                    'deterministic_algorithms': True,
                    'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                },
                criteria=self.diagnostic_criteria,
            )

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

        self.num_history = self.model_args.num_history

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws = self._run_eval_dual_system()
        elif self.model_args.mode == 'system2':
            sucs, spls, oss, nes, ndtws = self._run_eval_system2()
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        result = {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
        }

        if ndtws is not None:
            result["ndtws"] = ndtws  # shape [N_local]
        return result

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

        # clean NaN in spls, treat as 0.0
        torch.nan_to_num(spls_all, nan=0.0, posinf=0.0, neginf=0.0, out=spls_all)

        # clean inf in nes, only fiinite nes are counted
        nes_finite_mask = torch.isfinite(nes_all)
        nes_all = nes_all[nes_finite_mask]

        result_all = {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

        if "ndtws" in global_metrics:
            ndtws_all = global_metrics["ndtws"]
            result_all["ndtws_all"] = float(ndtws_all.mean().item()) if denom > 0 else 0.0

        return result_all

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def resume_from_output_path(self) -> None:
        sucs, spls, oss, nes, ndtw = [], [], [], [], []
        if self.rank != 0:
            return sucs, spls, oss, nes, ndtw

        # resume from previous results
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
                    if 'ndtw' in res:
                        ndtw.append(res['ndtw'])
        return sucs, spls, oss, nes, ndtw

    @staticmethod
    def _collision_values(metrics):
        collisions = metrics.get('collisions', {}) if metrics else {}
        if isinstance(collisions, dict):
            return bool(collisions.get('is_collision', False)), collisions.get('count')
        return bool(collisions), None

    def _prepare_s1_actions(self, dp_actions, depth=None):
        """Choose one candidate and determine how many actions may run open-loop."""
        if not self.minimal_s1_safety:
            return traj_to_actions(dp_actions), None, MAX_LOCAL_STEPS

        selection = self.trajectory_selector.select(
            dp_actions,
            depth=depth,
            recent_failure=self.recovery_controller.goal_retry_count > 0,
        )
        selected = dp_actions[selection.selected_index : selection.selected_index + 1]
        if hasattr(selected, 'clone'):
            selected = selected.clone()
        else:
            selected = np.array(selected, copy=True)
        return traj_to_actions(selected), selection, selection.chunk_size

    def _depth_observation_to_metres(self, depth):
        depth = np.asarray(depth).squeeze().astype(np.float32)
        return depth * (self._max_depth - self._min_depth) + self._min_depth

    def _diagnostic_env_step(self, action, observations_before, decision_step, phase, **fields):
        """Execute one Habitat action and log both model and camera-only steps."""
        metrics_before = self.env.get_metrics()
        result = self.env.step(action)
        observations_after, _, _, _ = result

        if self.diagnostic_logger is None or observations_after is None:
            return result

        metrics_after = self.env.get_metrics()
        collision, collision_count = self._collision_values(metrics_after)
        gps_before = np.asarray(observations_before.get('gps', [np.nan, np.nan])).reshape(-1)
        gps_after = np.asarray(observations_after.get('gps', [np.nan, np.nan])).reshape(-1)
        compass_before = np.asarray(observations_before.get('compass', [np.nan])).reshape(-1).tolist()
        compass_after = np.asarray(observations_after.get('compass', [np.nan])).reshape(-1).tolist()
        depth_m = self._depth_observation_to_metres(observations_after['depth'])

        try:
            action_name = action_code(int(action)).name
        except ValueError:
            action_name = f'UNKNOWN_{int(action)}'

        self.diagnostic_logger.record_env_step(
            env_step=self._diagnostic_env_step_id,
            decision_step=decision_step,
            action=int(action),
            action_name=action_name,
            gps_before=gps_before,
            gps_after=gps_after,
            distance_before=metrics_before.get('distance_to_goal'),
            distance_after=metrics_after.get('distance_to_goal'),
            collision=collision,
            collision_count=collision_count,
            compass_before=compass_before,
            compass_after=compass_after,
            phase=phase,
            depth=depth_statistics(depth_m),
            **fields,
        )
        if collision:
            self.diagnostic_logger.save_depth(
                f'collision_env_{self._diagnostic_env_step_id:04d}',
                depth_m,
                env_step=self._diagnostic_env_step_id,
                decision_step=decision_step,
                action=action_name,
            )
        self._diagnostic_env_step_id += 1
        return result

    def _run_eval_dual_system(self) -> tuple:  # noqa: C901
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            episode_seed = stable_episode_seed(self.base_seed, scene_id, episode_id)
            seed_everything(episode_seed)
            print("episode start", episode_instruction)

            self._diagnostic_env_step_id = 0
            self.recovery_controller.reset()
            s2_call_id = 0
            s1_plan_id = 0
            current_s2_call_id = None
            current_s1_plan_id = None
            latest_s2_output = None
            recovery_context = None
            stop_reason = 'unknown'
            if self.diagnostic_logger is not None:
                goal_positions = [getattr(goal, 'position', None) for goal in getattr(episode, 'goals', [])]
                self.diagnostic_logger.start_episode(
                    {
                        'scene_id': scene_id,
                        'episode_id': episode_id,
                        'instruction': episode_instruction,
                        'base_seed': self.base_seed,
                        'episode_seed': episode_seed,
                        'start_position': getattr(episode, 'start_position', None),
                        'start_rotation': getattr(episode, 'start_rotation', None),
                        'goal_positions': goal_positions,
                    }
                )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            first_person_frame_id = 0
            top_down_frame_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            action = None
            messages = []
            local_actions = []

            done = False
            flag = False
            pixel_goal = None

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000
                front_depth_m = None
                if action != action_code.LOOKDOWN:
                    front_depth_m = depth / 1000.0

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                    down_observations, _, _, _ = self._diagnostic_env_step(
                        action_code.LOOKDOWN,
                        observations,
                        step_id,
                        'camera_adjustment',
                        reason='prepare_s1_depth',
                    )
                    down_observations, _, _, _ = self._diagnostic_env_step(
                        action_code.LOOKDOWN,
                        down_observations,
                        step_id,
                        'camera_adjustment',
                        reason='prepare_s1_depth',
                    )

                    look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                    depth = down_observations["depth"]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    up_observations, _, _, _ = self._diagnostic_env_step(
                        action_code.LOOKUP,
                        down_observations,
                        step_id,
                        'camera_adjustment',
                        reason='restore_horizontal_view',
                    )
                    horizontal_observations, _, _, _ = self._diagnostic_env_step(
                        action_code.LOOKUP,
                        up_observations,
                        step_id,
                        'camera_adjustment',
                        reason='restore_horizontal_view',
                    )
                    front_depth_m = self._depth_observation_to_metres(horizontal_observations['depth'])
                    observations = horizontal_observations

                if len(action_seq) == 0 and pixel_goal is None:
                    s2_recovery_context = None
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
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
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                        if recovery_context:
                            s2_recovery_context = recovery_context
                            sources[0]["value"] += f" {recovery_context}"
                            recovery_context = None

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
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

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    current_s2_call_id = s2_call_id
                    s2_call_id += 1
                    s2_started = time.perf_counter()
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences
                    s2_generate_ms = (time.perf_counter() - s2_started) * 1000

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    latest_s2_output = llm_outputs
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True

                        # look down --> horizontal
                        up_observations, _, _, _ = self._diagnostic_env_step(
                            action_code.LOOKUP,
                            observations,
                            step_id,
                            'camera_adjustment',
                            reason='restore_from_s2_lookdown',
                        )
                        horizontal_observations, _, _, _ = self._diagnostic_env_step(
                            action_code.LOOKUP,
                            up_observations,
                            step_id,
                            'camera_adjustment',
                            reason='restore_from_s2_lookdown',
                        )
                        front_depth_m = self._depth_observation_to_metres(horizontal_observations['depth'])
                        observations = horizontal_observations

                        local_actions = []
                        pixel_values = inputs.pixel_values
                        image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                        with torch.no_grad():
                            latent_started = time.perf_counter()
                            traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
                            latent_generate_ms = (time.perf_counter() - latent_started) * 1000

                        if self.diagnostic_logger is not None:
                            self.diagnostic_logger.log(
                                's2_inference',
                                s2_call_id=current_s2_call_id,
                                decision_step=step_id,
                                raw_output=llm_outputs,
                                output_type='pixel_goal',
                                pixel_goal=pixel_goal,
                                history_frame_ids=history_id if action != action_code.LOOKDOWN else [],
                                input_image_count=len(input_images),
                                generated_token_count=int(output_ids.shape[-1] - inputs.input_ids.shape[1]),
                                generate_ms=s2_generate_ms,
                                latent_generate_ms=latent_generate_ms,
                                latent_shape=list(traj_latents.shape),
                                latent_l2_norm=float(traj_latents.detach().float().norm().item()),
                                recovery_context=s2_recovery_context,
                            )

                        # prepocess align with navdp
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                        pix_goal_image = copy.copy(image_dp)
                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        pix_goal_depth = copy.copy(depth_dp)
                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                        current_s1_plan_id = s1_plan_id
                        s1_plan_id += 1
                        s1_started = time.perf_counter()
                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)
                        s1_generate_ms = (time.perf_counter() - s1_started) * 1000

                        if self.diagnostic_logger is not None:
                            self.diagnostic_logger.save_s1_plan(
                                current_s1_plan_id,
                                dp_actions,
                                s2_call_id=current_s2_call_id,
                                decision_step=step_id,
                                pixel_goal=pixel_goal,
                                replan_reason='new_s2_pixel_goal',
                                generate_ms=s1_generate_ms,
                            )
                            self.diagnostic_logger.save_depth(
                                f'plan_{current_s1_plan_id:04d}',
                                look_down_depth.numpy(),
                                plan_id=current_s1_plan_id,
                                s2_call_id=current_s2_call_id,
                                decision_step=step_id,
                            )

                        self.recovery_controller.start_new_goal()
                        action_list, trajectory_selection, chunk_size = self._prepare_s1_actions(
                            dp_actions, depth=front_depth_m
                        )
                        if self.diagnostic_logger is not None:
                            self.diagnostic_logger.log(
                                's1_discretization',
                                plan_id=current_s1_plan_id,
                                s2_call_id=current_s2_call_id,
                                decision_step=step_id,
                                discrete_actions=[int(item) for item in action_list],
                                executable_chunk=[int(item) for item in action_list[:chunk_size]],
                                selected_index=(
                                    trajectory_selection.selected_index if trajectory_selection is not None else None
                                ),
                                trajectory_dispersion=(
                                    trajectory_selection.dispersion if trajectory_selection is not None else None
                                ),
                                adaptive_chunk_size=chunk_size,
                                trajectory_bimodal=(
                                    trajectory_selection.bimodal if trajectory_selection is not None else None
                                ),
                                trajectory_cluster_sizes=(
                                    trajectory_selection.cluster_sizes if trajectory_selection is not None else None
                                ),
                                selected_clearance=(
                                    trajectory_selection.clearance if trajectory_selection is not None else None
                                ),
                                selected_smoothness=(
                                    trajectory_selection.smoothness if trajectory_selection is not None else None
                                ),
                                depth_risk=(
                                    trajectory_selection.depth_risk if trajectory_selection is not None else None
                                ),
                            )
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= chunk_size:
                            local_actions = local_actions[:chunk_size]

                        action = local_actions.pop(0)
                        if action == action_code.STOP:
                            pixel_goal = None
                            output_ids = None
                            local_actions = []
                            if self.diagnostic_logger is not None:
                                self.diagnostic_logger.log(
                                    'controller_replan',
                                    decision_step=step_id,
                                    s2_call_id=current_s2_call_id,
                                    plan_id=current_s1_plan_id,
                                    reason='initial_s1_plan_started_with_stop',
                                    next_system='s2',
                                )
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        if self.diagnostic_logger is not None:
                            self.diagnostic_logger.log(
                                's2_inference',
                                s2_call_id=current_s2_call_id,
                                decision_step=step_id,
                                raw_output=llm_outputs,
                                output_type='stop' if action_seq == [action_code.STOP] else 'direct_actions',
                                parsed_actions=[int(item) for item in action_seq],
                                history_frame_ids=history_id if action != action_code.LOOKDOWN else [],
                                input_image_count=len(input_images),
                                generated_token_count=int(output_ids.shape[-1] - inputs.input_ids.shape[1]),
                                generate_ms=s2_generate_ms,
                                recovery_context=s2_recovery_context,
                            )
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                    action_source = 's2_direct'
                elif pixel_goal is not None:
                    action_source = 's1'
                    if len(local_actions) == 0:
                        # navdp
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255

                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                        current_s1_plan_id = s1_plan_id
                        s1_plan_id += 1
                        s1_started = time.perf_counter()
                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)
                        s1_generate_ms = (time.perf_counter() - s1_started) * 1000

                        if self.diagnostic_logger is not None:
                            self.diagnostic_logger.save_s1_plan(
                                current_s1_plan_id,
                                dp_actions,
                                s2_call_id=current_s2_call_id,
                                decision_step=step_id,
                                pixel_goal=pixel_goal,
                                replan_reason='local_chunk_exhausted',
                                generate_ms=s1_generate_ms,
                            )
                            self.diagnostic_logger.save_depth(
                                f'plan_{current_s1_plan_id:04d}',
                                look_down_depth.numpy(),
                                plan_id=current_s1_plan_id,
                                s2_call_id=current_s2_call_id,
                                decision_step=step_id,
                            )

                        action_list, trajectory_selection, chunk_size = self._prepare_s1_actions(
                            dp_actions, depth=front_depth_m
                        )
                        if self.diagnostic_logger is not None:
                            self.diagnostic_logger.log(
                                's1_discretization',
                                plan_id=current_s1_plan_id,
                                s2_call_id=current_s2_call_id,
                                decision_step=step_id,
                                discrete_actions=[int(item) for item in action_list],
                                executable_chunk=[int(item) for item in action_list[:chunk_size]],
                                selected_index=(
                                    trajectory_selection.selected_index if trajectory_selection is not None else None
                                ),
                                trajectory_dispersion=(
                                    trajectory_selection.dispersion if trajectory_selection is not None else None
                                ),
                                adaptive_chunk_size=chunk_size,
                                trajectory_bimodal=(
                                    trajectory_selection.bimodal if trajectory_selection is not None else None
                                ),
                                trajectory_cluster_sizes=(
                                    trajectory_selection.cluster_sizes if trajectory_selection is not None else None
                                ),
                                selected_clearance=(
                                    trajectory_selection.clearance if trajectory_selection is not None else None
                                ),
                                selected_smoothness=(
                                    trajectory_selection.smoothness if trajectory_selection is not None else None
                                ),
                                depth_risk=(
                                    trajectory_selection.depth_risk if trajectory_selection is not None else None
                                ),
                            )
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= chunk_size:
                            local_actions = local_actions[:chunk_size]
                        print("local_actions", local_actions)
                        action = local_actions.pop(0)
                    else:
                        action = local_actions.pop(0)

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == action_code.STOP:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0
                    action_source = 'default_stop'

                info = self.env.get_metrics()

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if pixel_goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)
                    top_down_frame_id += 1

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    vis = cv2.putText(
                        vis,
                        f"source {action_source} s2 {current_s2_call_id} s1 {current_s1_plan_id}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
                    if latest_s2_output:
                        vis = cv2.putText(
                            vis,
                            f"S2: {format_s2_output_for_overlay(latest_s2_output)[:70]}",
                            (20, 108),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            1,
                        )
                    if pixel_goal is not None:
                        if draw_pixel_goal:
                            cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)
                    first_person_frame_id += 1

                if action == action_code.LOOKDOWN:
                    down_observations, _, _, _ = self._diagnostic_env_step(
                        action,
                        observations,
                        step_id,
                        'model_action',
                        action_source=action_source,
                        s2_call_id=current_s2_call_id,
                        plan_id=current_s1_plan_id,
                        first_person_frame_id=first_person_frame_id - 1 if vis_writer is not None else None,
                        top_down_frame_id=top_down_frame_id - 1 if self.save_video else None,
                    )
                    observations, _, done, _ = self._diagnostic_env_step(
                        action,
                        down_observations,
                        step_id,
                        'model_action',
                        action_source=action_source,
                        s2_call_id=current_s2_call_id,
                        plan_id=current_s1_plan_id,
                        first_person_frame_id=first_person_frame_id - 1 if vis_writer is not None else None,
                        top_down_frame_id=top_down_frame_id - 1 if self.save_video else None,
                    )
                    flag = True
                else:
                    gps_before_action = np.asarray(observations.get('gps', [np.nan, np.nan])).copy()
                    observations, _, done, _ = self._diagnostic_env_step(
                        action,
                        observations,
                        step_id,
                        'model_action',
                        action_source=action_source,
                        s2_call_id=current_s2_call_id,
                        plan_id=current_s1_plan_id,
                        pixel_goal=pixel_goal,
                        s2_raw_output=latest_s2_output,
                        remaining_s2_actions=[int(item) for item in action_seq],
                        remaining_s1_actions=[int(item) for item in local_actions],
                        first_person_frame_id=first_person_frame_id - 1 if vis_writer is not None else None,
                        top_down_frame_id=top_down_frame_id - 1 if self.save_video else None,
                    )
                    collision, _ = self._collision_values(self.env.get_metrics())
                    recovery = self.recovery_controller.update(
                        action,
                        gps_before_action,
                        observations.get('gps', [np.nan, np.nan]),
                        collision=collision,
                        is_s1=self.minimal_s1_safety and action_source == 's1',
                    )
                    if recovery.cancel_remaining:
                        cancelled_s1_actions = [int(item) for item in local_actions]
                        cancelled_s2_actions = [int(item) for item in action_seq]
                        local_actions = []
                        action_seq = []
                        if self.diagnostic_logger is not None:
                            self.diagnostic_logger.log(
                                's1_recovery' if action_source == 's1' else 'controller_recovery',
                                decision_step=step_id,
                                plan_id=current_s1_plan_id,
                                action_source=action_source,
                                reason=recovery.reason,
                                gps_displacement=recovery.gps_displacement,
                                cancelled_s1_actions=cancelled_s1_actions,
                                cancelled_s2_actions=cancelled_s2_actions,
                                retry_count=recovery.retry_count,
                                goal_retry_count=recovery.goal_retry_count,
                                turn_count=recovery.turn_count,
                                consecutive_failures=self.recovery_controller.consecutive_failures,
                                failed_directions=dict(self.recovery_controller.failed_directions),
                                replan_system2=recovery.replan_system2,
                            )
                        if recovery.replan_system2:
                            pixel_goal = None
                            output_ids = None
                            forward_action = 0
                            messages = []
                            if recovery.reason == 'repeated_turns':
                                recovery_context = (
                                    'Recovery notice: you just rotated a full circle without translating. '
                                    'Reassess which parts of the original instruction remain unfinished, '
                                    'then choose a genuinely different next action or STOP only if the task is complete. '
                                    'Returning along the previous route is allowed when it is necessary.'
                                )
                            else:
                                recovery_context = (
                                    f'Recovery notice: the previous waypoint failed repeatedly because of '
                                    f'{recovery.reason}. Reassess which parts of the original instruction remain '
                                    'unfinished and select a reachable next waypoint. Returning along the previous '
                                    'route is allowed when it is necessary.'
                                )
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            if done:
                stop_reason = 'habitat_episode_done'
            elif step_id > self.max_steps_per_episode:
                stop_reason = 'model_decision_limit'
            else:
                stop_reason = 'evaluation_loop_exit'

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            diagnostic_summary = None
            if self.diagnostic_logger is not None:
                metric_summary = {
                    'success': metrics['success'],
                    'spl': metrics['spl'],
                    'oracle_success': metrics['oracle_success'],
                    'distance_to_goal': metrics['distance_to_goal'],
                    'ndtw': metrics.get('ndtw'),
                    'collisions': metrics.get('collisions'),
                }
                diagnostic_summary = self.diagnostic_logger.finish_episode(
                    metric_summary,
                    stop_reason,
                    model_decision_steps=step_id,
                    habitat_action_steps=self._diagnostic_env_step_id,
                    first_person_video=os.path.join(
                        self.vis_debug_path,
                        f'epoch_{self.epoch}',
                        f'{scene_id}_{episode_id:04d}.mp4',
                    ),
                    top_down_video=os.path.join(
                        self.output_path,
                        f'vis_{self.epoch}',
                        f'{scene_id}',
                        f'{episode_id:04d}.mp4',
                    ),
                )

            # Write per-episode progress.json entry (still per-rank)
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
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']
            if diagnostic_summary is not None:
                result.update(
                    {
                        'base_seed': self.base_seed,
                        'episode_seed': episode_seed,
                        'stop_reason': stop_reason,
                        'collision_steps': diagnostic_summary['collision_steps'],
                        'collision_streak_events': diagnostic_summary['collision_streak_events'],
                        'stuck_events': diagnostic_summary['stuck_events'],
                        'oscillation_events': diagnostic_summary['oscillation_events'],
                        'no_progress_events': diagnostic_summary['no_progress_events'],
                        'automatic_s1_failure_candidate': diagnostic_summary['automatic_s1_failure_candidate'],
                    }
                )

            # save current progress
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

            # save video
            if self.save_video and vis_frames:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )

    def _run_eval_system2(self) -> tuple:
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False)

            intrinsic_matrix = get_intrinsic_matrix(
                self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
            )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            goal = None
            action = None
            messages = []

            done = False
            flag = False

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height  # Habitat GPS makes west negative, so flip y
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30)) @ get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                if len(action_seq) == 0 and goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
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
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
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

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True

                        # look down --> horizontal
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        goal = pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)

                        goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                        if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                            goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                        action = agent.get_next_action(goal)
                        if action == action_code.STOP:
                            goal = None
                            output_ids = None
                            action = action_code.LEFT  # random action to avoid deadlock
                            observations, _, done, _ = self.env.step(action)
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
                    action = agent.get_next_action(goal)
                    action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                    action = action[0] if hasattr(action, "__len__") else action

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                    if action == action_code.STOP:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if draw_pixel_goal:
                        cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode result.json entry (still per-rank)
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
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            if self.save_video and vis_frames:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )
