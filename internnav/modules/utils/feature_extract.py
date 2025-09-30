from typing import Any, Dict, List

import torch

# old
# def extract_instruction_tokens(
#     observations: List[Dict],
#     bert_tokenizer=None,
#     is_clip_long=False,
# ) -> Dict[str, Any]:
#     """Extracts instruction tokens from an instruction sensor if the tokens
#     exist and are in a dict structure.
#     """
#     for i in range(len(observations)):
#         if bert_tokenizer is None:
#             # For habitat-cma
#             # observations[i]['instruction'] = observations[i]['instruction']["tokens"]
#             observations[i]['instruction'] = observations[i]['instruction_tokens']
#             # pad to 200
#             instr = torch.tensor(observations[i]['instruction'])
#             observations[i]['instruction'] = torch.nn.functional.pad(
#                 instr, (0, 200 - instr.shape[0]), 'constant', 0
#             )
#         else:
#             # use bert tokenizer
#             if is_clip_long:
#                 tokens = bert_tokenizer(observations[i]['instruction'])[0].tolist()
#             else:
#                 tokens = bert_tokenizer.text_token(observations[i]['instruction'])[
#                     'input_ids'
#                 ][0].tolist()
#             observations[i]['instruction'] = tokens
#     return observations


def extract_instruction_tokens(
    observations: List[Dict],
    bert_tokenizer=None,
    is_clip_long=False,
    max_instr_len=200,
) -> Dict[str, Any]:
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    for i in range(len(observations)):
        if bert_tokenizer is None:
            # For habitat-cma
            # observations[i]['instruction'] = observations[i]['instruction']["tokens"]
            observations[i]['instruction'] = observations[i]['instruction_tokens']
            # pad to 200
            instr = torch.tensor(observations[i]['instruction'])
            observations[i]['instruction'] = torch.nn.functional.pad(
                instr, (0, max_instr_len - instr.shape[0]), 'constant', 0
            )
        else:
            # use bert tokenizer
            if is_clip_long:
                try:
                    tokens = bert_tokenizer(observations[i]['instruction'])[0].tolist()
                except Exception as e:
                    print(1)
            else:
                tokens = bert_tokenizer.text_token(observations[i]['instruction'])['input_ids'][0].tolist()
            observations[i]['instruction'] = tokens
    return observations


def extract_image_features(
    policy,
    batch,
    img_mod,
    len_traj_act=4,
    world_size=1,
    stack_rgb=None,
    stack_depth=None,
    depth_encoder_type='resnet',
    save_img_raw=False,
    proj=True,
    net_device=None,
    need_rgb_extraction=True,
    classifier_free_mask_depth=False,
):
    """Extracts image features from observations using the policy's image feature extractor."""
    device = batch['globalgps'].device
    bs = batch['globalgps'].shape[0]

    if device.type == 'cpu' and net_device is not None:
        device = net_device

    if world_size > 1:
        net = policy.module
    else:
        net = policy

    if stack_rgb is None and stack_depth is None:
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)

        rgb_feat = net.image_encoder.process_image(rgb).float().to(device)
        if depth_encoder_type == 'resnet':
            depth_feat = depth.type(torch.float32)

        batch_inputs = {
            'mode': 'img_embedding',
            'rgb_inputs': rgb_feat,
            'depth_inputs': depth_feat,
            'depth_return_x_before_fc': True,  # For ResNet depth encoder,
            'img_mod': img_mod,
            'proj': proj,
            'process_images': False,
            'need_rgb_extraction': need_rgb_extraction,
        }
        rgb_features, depth_features = policy(batch_inputs)
        rgb_features = rgb_features.type(torch.float32)
        depth_features = depth_features.type(torch.float32)

        batch['stack_rgb'] = rgb_features
        batch['stack_depth'] = depth_features
    else:
        rgb = stack_rgb
        depth = stack_depth

        rgb = rgb.view(-1, rgb.shape[-2], rgb.shape[-3], rgb.shape[-1])
        depth = depth.view(-1, depth.shape[-2], depth.shape[-3], depth.shape[-1])

        rgb_feat = net.image_encoder.process_image(rgb).float().to(device)
        if depth_encoder_type == 'resnet':
            depth_feat = depth.type(torch.float32)

        if classifier_free_mask_depth:
            depth_null_feat = torch.zeros_like(depth_feat)
            depth_feat = torch.cat([depth_feat, depth_null_feat], dim=0)
            rgb_feat = torch.cat([rgb_feat, rgb_feat], dim=0)
            bs = bs * 2

        batch_inputs = {
            'mode': 'img_embedding',
            'rgb_inputs': rgb_feat,
            'depth_inputs': depth_feat,
            'depth_return_x_before_fc': True,  # For ResNet depth encoder
            'img_mod': img_mod,
            'proj': proj,
            'process_images': False,
            'need_rgb_extraction': need_rgb_extraction,
        }
        rgb_features, depth_features = policy(batch_inputs)
        rgb_features = rgb_features.type(torch.float32)
        depth_features = depth_features.type(torch.float32)

        # padding to the len_traj_act
        if img_mod == 'cls':
            dim = 512 if proj else 768
            rgb_features = rgb_features.reshape(bs, len_traj_act, dim)
        elif img_mod == 'multi_patches_avg_pooling':
            multi_patch_num = rgb_features.shape[-2]
            rgb_features = rgb_features.reshape(bs, len_traj_act, multi_patch_num, rgb_features.shape[-1])
        depth_features = depth_features.reshape(
            bs,
            len_traj_act,
            depth_features.shape[-3],
            depth_features.shape[-2],
            depth_features.shape[-1],
        )
        if classifier_free_mask_depth:
            bs = bs // 2
            rgb_features, _ = torch.split(rgb_features, bs, dim=0)
            depth_features, null_depth_features = torch.split(depth_features, bs, dim=0)
            batch['stack_null_depth'] = null_depth_features

        batch['stack_rgb'] = rgb_features
        batch['stack_depth'] = depth_features

    if not save_img_raw:
        if 'rgb' in batch.keys():
            del batch['rgb']
        if 'depth' in batch.keys():
            del batch['depth']

    return batch
