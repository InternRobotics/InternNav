import argparse
import sys

sys.path.append('./src/diffusion-policy')


# Import for Habitat registry side effects — do not remove
import internnav.env.utils.habitat_extensions.measures  # noqa: F401
from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator

try:
    import habitat
    from habitat.config.default import get_agent_config
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat_baselines.config.default import get_config as get_habitat_config
except Exception as e:
    print("Habitat Error:", e)
    print("Habitat Evaluation is not loaded.")


DEFAULT_IMAGE_TOKEN = "<image>"


@Evaluator.register('habitat_vln')
class HabitatVlnEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.args = args
        self.save_video = args.save_video
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path

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
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        super().__init__(cfg)

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        sucs, spls, oss, nes = [], [], [], []
        env = self.env

        while env.is_running:
            obs = env.reset()
            if not env.is_running or obs is None:
                break

            episode = env.env.current_episode
            self.agent.reset(episode, env)

            done = False
            step_id = 0
            while not done and step_id <= self.max_steps_per_episode:
                action = self.agent.act(obs, env, info=None)
                obs, reward, done, info = env.step(action)
                step_id += 1

            m = env.get_metrics()
            sucs.append(m["success"])
            spls.append(m["spl"])
            oss.append(m["oracle_success"])
            nes.append(m["distance_to_goal"])

        env.close()
        return {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
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
