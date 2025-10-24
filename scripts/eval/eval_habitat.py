import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from internnav.evaluator.habitat_vln_evaluator import VLNEvaluator
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.utils.dist import *


def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate InternVLA-N1 on Habitat')
    parser.add_argument("--mode", default='dual_system', type=str, help="inference mode: dual_system or system2")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='scripts/eval/configs/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./logs/habitat/test')  #!
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
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

    return parser.parse_args()


def main():
    args = parse_args()

    init_distributed_mode(args)
    local_rank = args.local_rank
    np.random.seed(local_rank)

    # * 1. Load model and tokenizer. Currently, we support two modes: dual_system and system2 in Habitat.
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
    world_size = get_world_size()

    # * 2. initialize evaluator
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

    # * 3. do eval
    sucs, spls, oss, nes, ep_num = evaluator.eval_action(idx=get_rank())
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]

    # import ipdb; ipdb.set_trace()
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
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))


if __name__ == '__main__':
    main()
