"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import argparse
import glob
import json
import os

import torch


def save_training_meta(args):
    def convert_namespace_to_dict(obj):
        if isinstance(obj, argparse.Namespace):
            return {k: convert_namespace_to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, list):
            return [convert_namespace_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_namespace_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()  # Convert tensor to list
        else:
            return obj  # Return the value directly if it is already serializable

    if os.path.isdir(args.LOG_DIR):
        args_dict = convert_namespace_to_dict(args)
        with open(os.path.join(args.LOG_DIR, 'training_args.json'), 'w') as writer:
            json.dump(args_dict, writer, indent=4)


def load_checkpoint(checkpoint_path, *args, map_location='cpu', **kwargs):
    try:
        return torch.load(checkpoint_path, map_location=map_location, *args, **kwargs)
    except ModuleNotFoundError as e:
        # use pickle_module to avoid module dependency problem
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            pickle_module=torch.serialization.pickle,
            *args,
            **kwargs,
        )


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, step, optimizer=None):
        output_model_file = os.path.join(self.output_dir, f'{self.prefix}_{step}.{self.suffix}')
        state_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('module.'):
                k = k[7:]
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.cpu()
            else:
                state_dict[k] = v
        torch.save(state_dict, output_model_file)
        if optimizer is not None:
            dump = {'step': step, 'optimizer': optimizer.state_dict()}
            torch.save(dump, f'{self.output_dir}/train_state_{step}.pt')

    def save_latest(self, model, step, optimizer=None, is_max=False):
        state_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('module.'):
                k = k[7:]
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.cpu()
            else:
                state_dict[k] = v
        if is_max:
            output_model_file = os.path.join(self.output_dir, f'{self.prefix}_best.{self.suffix}')
            torch.save(state_dict, output_model_file)
        else:
            output_model_file = os.path.join(self.output_dir, f'{self.prefix}_latest.{self.suffix}')
            torch.save(state_dict, output_model_file)
        if optimizer is not None:
            dump = {'step': step, 'optimizer': optimizer.state_dict()}
            if is_max:
                torch.save(dump, f'{self.output_dir}/train_state_best.pt')
            else:
                torch.save(dump, f'{self.output_dir}/train_state_latest.pt')


def poll_checkpoint_folder(
    checkpoint_folder: str,
    previous_ckpt_ind: int,
    start_eval_epoch=-1,
    first_find_start_epoch=False,
):
    r"""Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), f'invalid checkpoint folder ' f'path {checkpoint_folder}'
    models_paths = list(filter(os.path.isfile, glob.glob(checkpoint_folder + '/*')))
    new_model_paths = []
    for path in models_paths:
        if path.endswith('.pth'):
            new_model_paths.append(path)
    models_paths = new_model_paths
    if len(models_paths) == 0:
        print('No checkpoints found in folder: ', checkpoint_folder)
        return -1
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if start_eval_epoch != -1:
        if first_find_start_epoch:
            for idx, model_path in enumerate(models_paths):
                ckpt_file_ind = int(model_path.split('/')[-1].split('.')[1])
                if ckpt_file_ind >= start_eval_epoch:
                    ind = idx
                    print(f'Find the start eval epoch file for {start_eval_epoch}-th epoch.')
                    break
            previous_ckpt_ind = ind
            return (models_paths[ind], previous_ckpt_ind)

    if ind < len(models_paths):
        return models_paths[ind]
    return None
