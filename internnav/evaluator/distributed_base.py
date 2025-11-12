import json
import os

import numpy as np
import torch

from internnav.configs.evaluator import EvalCfg
from internnav.env import Env
from internnav.evaluator import Evaluator
from internnav.utils.dist import dist, get_rank, get_world_size, init_distributed_mode


class DistributedEvaluator(Evaluator):
    """
    Base class of distributed evaluators.

    Args:
        cfg (EvalCfg): evaluation configuration
        init_env (bool): whether to initialize the environment
        init_agent (bool): whether to initialize the agent
    """

    def __init__(self, cfg: EvalCfg, init_env: bool = True, init_agent: bool = True):
        # distributed setting
        import socket

        print(
            f"Rank {os.getenv('RANK')} / {os.getenv('WORLD_SIZE')} on {socket.gethostname()}:{os.getenv('MASTER_PORT')}"
        )

        self.output_path = cfg.eval_settings["output_path"]  # TODO: unsafe for distribution

        init_distributed_mode()

        self.local_rank = get_rank()
        np.random.seed(self.local_rank)
        self.world_size = get_world_size()

        # habitat env also need rank to split dataset
        cfg.env.env_settings['local_rank'] = get_rank()
        cfg.env.env_settings['world_size'] = get_world_size()

        self.eval_config = cfg

        if init_env:
            self.env = Env.init(cfg.env, cfg.task)

        # -------- initialize agent config (either remote server or local agent) --------
        if init_agent:
            if cfg.remote_agent:
                # set agent port based on rank
                from internnav.utils import AgentClient

                cfg.agent.agent_settings['port'] = 8000 + get_rank()
                self.agent = AgentClient(cfg.agent)
            else:
                from internnav.agent import Agent

                self.agent = Agent(cfg.agent)

    def eval(self):
        """
        Uniform distributed evaluation pipeline:

        1. Call subclass's eval_action() to get local per-episode tensors.
        2. Use dist all_gather (+ padding) to build global tensors for each metric.
        3. Call subclass's calc_metrics(global_metrics) to compute scalar metrics.
        4. Print + rank 0 writes result.json.
        """
        local_metrics = self.eval_action()  # dict[str, Tensor], each [N_local]

        if not local_metrics:
            raise RuntimeError("eval_action() returned empty metrics dict.")

        first_tensor = next(iter(local_metrics.values()))
        device = first_tensor.device
        local_len = first_tensor.shape[0]

        world_size = get_world_size()

        # -------- 1) Handle non-distributed / world_size == 1 --------
        if world_size == 1:
            global_metrics = {name: tensor.detach().cpu() for name, tensor in local_metrics.items()}
            total_len = int(local_len)
        else:
            # -------- 2) Gather lengths from all ranks --------
            local_len_t = torch.tensor([local_len], dtype=torch.long, device=device)
            len_list = [torch.zeros_like(local_len_t) for _ in range(world_size)]
            dist.all_gather(len_list, local_len_t)
            lens = torch.stack(len_list).cpu()  # shape [world_size, 1]
            lens = lens.view(-1)  # [world_size]
            max_len = int(lens.max().item())
            total_len = int(lens.sum().item())

            # -------- 3) For each metric, pad + all_gather + unpad --------
            global_metrics = {}
            for name, tensor in local_metrics.items():
                assert tensor.shape[0] == local_len, (
                    f"Metric {name} length ({tensor.shape[0]}) " f"!= first metric length ({local_len})"
                )

                # pad to max_len on this rank
                padded = torch.zeros(
                    max_len,
                    dtype=tensor.dtype,
                    device=device,
                )
                padded[:local_len] = tensor

                # gather padded tensors from all ranks
                gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(gathered, padded)

                # unpad & concat using true lengths
                parts = []
                for rank in range(world_size):
                    cur_len = int(lens[rank].item())
                    if cur_len > 0:
                        parts.append(gathered[rank][:cur_len])
                if parts:
                    global_tensor = torch.cat(parts, dim=0)
                else:
                    # no episodes at all (edge case)
                    global_tensor = torch.empty(0, dtype=tensor.dtype)

                global_metrics[name] = global_tensor.detach().cpu()

        # -------- 4) Let subclass compute final metrics from global tensors --------
        result_all = self.calc_metrics(global_metrics)
        result_all.setdefault("length", total_len)

        # -------- 5) Logging --------
        print(result_all)
        if get_rank() == 0:
            os.makedirs(self.output_path, exist_ok=True)
            out_path = os.path.join(self.output_path, "result.json")
            with open(out_path, "a") as f:
                f.write(json.dumps(result_all) + "\n")

        return result_all

    # ================= ABSTRACT HOOKS =================

    def eval_action(self) -> dict:
        """
        Run evaluation on this rank and return per-episode metrics.

        Returns
        -------
        dict[str, torch.Tensor]
            Example:
            {
                "sucs": tensor([0., 1., ...], device=...),
                "spls": tensor([...]),
                "oss": tensor([...]),
                "nes": tensor([...]),
            }
        """
        raise NotImplementedError

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        Compute final scalar metrics from global per-episode tensors.

        Parameters
        ----------
        global_metrics : dict[str, torch.Tensor]
            For each metric name, a 1-D CPU tensor with all episodes across all ranks.
            Example:
                {
                    "sucs": tensor([...], dtype=torch.float32),
                    "spls": tensor([...]),
                    ...
                }

        Returns
        -------
        dict[str, float]
            Final scalar metrics to log.
        """
        raise NotImplementedError
