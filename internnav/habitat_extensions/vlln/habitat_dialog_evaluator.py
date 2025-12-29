import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator

try:
    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat.utils.visualizations.utils import (
        images_to_video,
        observations_to_image,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config

    # Import for Habitat registry side effects — do not remove
    import internnav.habitat_extensions.vlln.measures  # noqa: F401
    from internnav.habitat_extensions.vlln.simple_npc.simple_npc import SimpleNPC
    from internnav.habitat_extensions.vlln.utils.dialog_utils import get_description

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
        self.config = get_habitat_config(self.config_path)

        with habitat.config.read_write(self.config):
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
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        cfg.agent.model_settings['task'] = self.task
        cfg.agent.model_settings['sim_sensors_config'] = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.objectnav_instruction = "search for {target_object}."
        super().__init__(cfg)

    def eval_action(self):
        sucs, spls, oss, nes = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
        env = self.env

        while env.is_running:
            # ------------ 1. Start of an episode ------------
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

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)

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
            vis_frames = []
            path_list = []
            action_list = []  # params for saving results

            # ---------- 2. Episode step loop -----------
            while not env._env.episode_over and step_id <= self.max_steps_per_episode:
                # save frames
                info = env.get_metrics()
                if info['top_down_map'] is not None and self.save_video:
                    save_image = Image.fromarray(obs["rgb"]).convert('RGB')
                    frame = observations_to_image({'rgb': np.asarray(save_image)}, info)
                    vis_frames.append(frame)

                agent_state = env._env.sim.get_agent_state()
                path_list.append(agent_state.position.tolist())
                info = {
                    'step': step_id,
                    'agent state': agent_state,
                    'episode_instruction': episode_instruction,
                    'output_path': os.path.join(self.output_path, 'action', f'{scene_id}', f'{episode_id}.txt'),
                    'info': info,
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
                        path_description, pl = get_description(env._env, object_dict, region_dict)
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
                elif action == 7:
                    continue
                else:
                    raise ValueError(f"Invalid action {action}!")

                step_id += 1
                self.agent.messages = []

            # ---------- 3. End of an episode -----------
            # collect the metric result of this episode and write progress to the output_path
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
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            if self.save_video:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()

        env.close()
        return {
            "sucs": torch.tensor(sucs).to(self.agent.device),  # shape [N_local]
            "spls": torch.tensor(spls).to(self.agent.device),  # shape [N_local]
            "oss": torch.tensor(oss).to(self.agent.device),  # shape [N_local]
            "nes": torch.tensor(nes).to(self.agent.device),  # shape [N_local]
        }

    def calc_metrics(self, global_metrics: dict) -> dict:
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

        return {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }
