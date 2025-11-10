import json
import os
from datetime import datetime

import torch

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator.base import Evaluator
from internnav.utils.dist import dist, get_rank, get_world_size


def init_distributed_mode(args):
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])

        num_gpus = torch.cuda.device_count()
        args.gpu = args.rank % num_gpus
        args.local_rank = args.gpu

        node_list = os.environ['SLURM_NODELIST']
        print(f'Node list: {node_list}')
        # addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')

        os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))
        # os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.local_rank = args.gpu
    else:
        print('Not using distributed mode')
        # setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(0, 7200),
    )
    dist.barrier()
    # setup_for_distributed(args.rank == 0)


class DistributedEvaluator(Evaluator):
    """
    Base class of distributed evaluators.
    """

    def __init__(self, cfg: EvalCfg):
        # distributed setting
        import os
        import socket

        print(
            f"Rank {os.getenv('RANK')} / {os.getenv('WORLD_SIZE')} on {socket.gethostname()}:{os.getenv('MASTER_PORT')}"
        )
        # init_distributed_mode(args)
        # local_rank = args.local_rank
        # np.random.seed(local_rank)
        cfg.env.env_settings['idx'] = get_rank()
        cfg.env.env_settings['world_size'] = get_world_size()

        # set agent port based on rank
        cfg.agent.agent_settings['port'] = 8000 + get_rank()
        # start_server(cfg.agent.agent_settings['port'])

        super().__init__(cfg)

    def eval(self):
        # 1. 每个 rank 本地跑一遍
        local_metrics = self.eval_action()  # dict[str, Tensor], 每个 Tensor shape [N]
        # 取出设备 & 本地样本数
        device = next(iter(local_metrics.values())).device
        local_count = torch.tensor([len(next(iter(local_metrics.values())))], dtype=torch.long, device=device)

        # 2. 全局样本数
        world_size = get_world_size()
        global_count = local_count.clone()
        if world_size > 1:
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)

        # 3. 对每个 metric 做全局 sum / mean
        result_all = {}
        for name, tensor in local_metrics.items():
            # tensor: [N]
            local_sum = tensor.sum()
            global_sum = local_sum.clone()
            if world_size > 1:
                dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)

            mean_val = (global_sum / global_count).item()
            result_all[name] = mean_val

        # 4. 统计全局 episode 数
        result_all["length"] = int(global_count.item())

        # 5. 打印 + 只在 rank 0 写文件
        print(result_all)
        if get_rank() == 0:
            os.makedirs(self.args.output_path, exist_ok=True)
            out_path = os.path.join(self.args.output_path, "result.json")
            with open(out_path, "a") as f:
                f.write(json.dumps(result_all) + "\n")

        return result_all

    def eval_action(self):
        """
        跑当前 rank 的 episodes, 返回一个 dict:
        {
            "success": tensor([0., 1., ...], device=...),
            "spl": tensor([...]),
            "os": tensor([...]),
            "ne": tensor([...]),
            ...
        }
        """
        raise NotImplementedError
