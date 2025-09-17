import pickle
import random
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from internnav.utils.common_log_util import common_logger as logger


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)
        
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                logger.info(f'{name} set to {drop_p}')


def set_cuda(opts, device) -> Tuple[bool, int, torch.device]:
    """
    Initialize CUDA for distributed computing
    """
    if not torch.cuda.is_available():
        assert opts.local_rank == -1, opts.local_rank
        return True, 0, torch.device('cpu')

    # get device settings
    if opts.ddp.use and not opts.ddp.use_dp:
        setup_ddp(opts.local_rank, opts.world_size)
        opts.local_rank = dist.get_rank()
        torch.cuda.set_device(opts.local_rank)
        device = torch.device('cuda', opts.local_rank)
        n_gpu = 1
        default_gpu = dist.get_rank() == 0
        if default_gpu:
            logger.info(f'Found {dist.get_world_size()} GPUs')
        logger.info(f'Process rank: {dist.get_rank()}, ' f'Local rank: {opts.local_rank}, ' f'Device: {device}')
    else:
        default_gpu = True
        device = torch.device('cuda') if device is None else device
        n_gpu = len(opts.torch_gpu_ids)

    return default_gpu, n_gpu, device


def wrap_model(
    model: torch.nn.Module,
    device: torch.device,
    local_rank: int,
    logger,
    world_size=1,
    use_dp=False,
) -> torch.nn.Module:
    if world_size > 1 and use_dp:  # DP mode
        if isinstance(device, list):
            model = torch.nn.DataParallel(model, device_ids=device)
            model = model.to(device[0])
        else:
            model = torch.nn.DataParallel(model)
            model = model.to(device)
    elif world_size > 1:  # DDP mode
        # ensure the model is on the correct GPU
        device = torch.device(f'cuda:{local_rank}')
        model = model.to(device)
        print(f'Process {local_rank} using device: {device}')

        model = DDP(
            model,
            device_ids=[local_rank],  # use the GPU corresponding to local_rank
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:  # single GPU mode
        if isinstance(device, list):
            model = model.to(device[0])
        else:
            model = model.to(device)

    return model


class NoOp(object):
    """useful for distributed training No-Ops"""

    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device='cuda')
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device='cuda'))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_ddp(local_rank, world_size):
    # Initialize the distributed environment
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU
        init_method='env://',  # Use environment variables for initialization
        world_size=world_size,  # Total number of processes
        rank=local_rank,  # Rank of the current process
    )
    torch.cuda.set_device(local_rank)  # Set GPU device for the current process
