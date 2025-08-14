import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import logging
from pathlib import Path
import torch.distributed as dist

import torch
import tyro
from pydantic import BaseModel
from transformers import TrainerCallback, TrainingArguments

from internnav.dataset.cma_lerobot_dataset import CMALerobotDataset, cma_collate_fn
from internnav.dataset.rdp_lerobot_dataset import RDP_LerobotDataset, rdp_collate_fn
from internnav.dataset.navdp_dataset_lerobot import NavDP_Base_Datset, navdp_collate_fn
from internnav.model import (
    CMAModelConfig,
    CMANet,
    RDPModelConfig,
    RDPNet,
    Seq2SeqModelConfig,
    Seq2SeqNet,
    NavDPNet,
    NavDPModelConfig,
)
from internnav.model.utils.logger import MyLogger
from internnav.model.utils.utils import load_dataset
from internnav.trainer import CMATrainer, RDPTrainer, NavDPTrainer
from scripts.train.configs import (
    cma_exp_cfg,
    cma_plus_exp_cfg,
    rdp_exp_cfg,
    seq2seq_exp_cfg,
    seq2seq_plus_exp_cfg,
    navdp_exp_cfg,
)
import sys
from datetime import datetime

class TrainCfg(BaseModel):
    """Training configuration class"""

    name: str = 'cma_train'  # Experiment name
    model_name: str = 'cma'  # Model name, options: 'cma', 'cma_plus', 'seq2seq', 'seq2seq_plus', 'rdp', 'navdp'


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(self, run_name: str, exp_cfg_dir: Path | None = None):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f'checkpoint-{state.global_step}'  # noqa: F841


def _make_dir(config):
    config.tensorboard_dir = config.tensorboard_dir % config.name  
    config.checkpoint_folder = config.checkpoint_folder % config.name
    config.log_dir = config.log_dir % config.name
    config.output_dir = config.output_dir % config.name
    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir,exist_ok=True)
    if not os.path.exists(config.checkpoint_folder):
        os.makedirs(config.checkpoint_folder,exist_ok=True)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir,exist_ok=True)


def main(config, model_class, model_config_class):
    try:
        """Main training function."""
        _make_dir(config)

        print(f"=== Start training ===")
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Environment variables:")
        print(f"  RANK: {os.getenv('RANK', 'Not set')}")
        print(f"  LOCAL_RANK: {os.getenv('LOCAL_RANK', 'Not set')}")
        print(f"  WORLD_SIZE: {os.getenv('WORLD_SIZE', 'Not set')}")
        print(f"  MASTER_ADDR: {os.getenv('MASTER_ADDR', 'Not set')}")
        print(f"  MASTER_PORT: {os.getenv('MASTER_PORT', 'Not set')}")

        if config.model_name == "navdp":
            local_rank = int(os.getenv('LOCAL_RANK', '0'))
            world_size = int(os.getenv('WORLD_SIZE', '1'))
            rank = int(os.getenv('RANK', '0'))
            
            # Set CUDA device for each process
            device_id = local_rank
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
            print(f"World size: {world_size}, Local rank: {local_rank}, Global rank: {rank}")
            
            # Initialize distributed training environment
            if world_size > 1:
                try:
                    dist.init_process_group(
                        backend='nccl',
                        init_method='env://',
                        world_size=world_size,
                        rank=rank
                    )
                    print("Distributed initialization SUCCESS")
                except Exception as e:
                    print(f"Distributed initialization FAILED: {str(e)}")
                    world_size = 1

            print("="*50)
            print("After distributed init:")
            print(f"LOCAL_RANK: {local_rank}")
            print(f"WORLD_SIZE: {world_size}")

        if dist.is_initialized():
            print(f"Dist WORLD_SIZE: {dist.get_world_size()}")
            print(f"Dist RANK: {dist.get_rank()}")
        else:
            print("Distributed NOT initialized")

        # ------------ load model ------------
        model_cfg = model_config_class(model_cfg=config.model_dump())
        if config.il.ckpt_to_load:
            print(f"load model from:{config.il.ckpt_to_load}")
        model = model_class.from_pretrained(pretrained_model_name_or_path=config.il.ckpt_to_load, config=model_cfg)
        if config.model_name == "navdp":
            model.to(device)
            # Check that all parameters and buffers are on the correct device
            for name, param in model.named_parameters():
                if param.device != device:
                    print(f"Parameter {name} is on wrong device {param.device}, should be moved to {device}")
                    param.data = param.data.to(device)

            for name, buffer in model.named_buffers():
                if buffer.device != device:
                    print(f"Buffer {name} is on wrong device {buffer.device}, should be moved to {device}")
                    buffer.data = buffer.data.to(device)
            
            # If distributed training, wrap the model with DDP
            if world_size > 1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True
                )
        # ------------ load logger ------------
        train_logger_filename = os.path.join(config.log_dir, 'train.log')
        if dist.is_initialized() and dist.get_rank() == 0:
            train_logger = MyLogger(
                name='train', level=logging.INFO, format_str='%(asctime)-15s %(message)s', filename=train_logger_filename
            )
        else:
            # Other processes use console logging
            train_logger = MyLogger(
                name='train', level=logging.INFO, format_str='%(asctime)-15s %(message)s'
            )
        transformers_logger = logging.getLogger("transformers")
        if transformers_logger.hasHandlers():
            transformers_logger.handlers = []
        if config.model_name == "navdp" and local_rank in [0, -1]:  # Only main process or non-distributed
            transformers_logger.addHandler(train_logger.handlers[0])
        transformers_logger.setLevel(logging.INFO)


        # ------------ load dataset ------------
        if config.model_name == "navdp":
            train_dataset_data = NavDP_Base_Datset(config.il.root_dir,
                                    config.il.dataset_navdp,
                                    config.il.memory_size,
                                    config.il.predict_size,
                                    config.il.batch_size,
                                    config.il.image_size,
                                    config.il.scene_scale,
                                    preload = config.il.preload,
                                    random_digit = config.il.random_digit,
                                    prior_sample = config.il.prior_sample)
        else:
            if '3dgs' in config.il.lmdb_features_dir or '3dgs' in config.il.lmdb_features_dir:
                dataset_root_dir = config.il.dataset_six_floor_root_dir
                dataset_type = '3dgs'
            elif 'grutopia' in config.il.lmdb_features_dir:
                dataset_root_dir = config.il.dataset_grutopia10_root_dir
                dataset_type = 'grutopia'
            else:
                dataset_root_dir = config.il.dataset_r2r_root_dir
                dataset_type = 'r2r'
            train_dataset_data = load_dataset(dataset_root_dir, 'train', logger=train_logger, dataset_type=dataset_type)
            global_batch_size = config.il.batch_size * len(config.torch_gpu_ids)

        # ------------ data_loader ------------
        if config.model_name in ['cma', 'seq2seq']:
            policy_trainer = CMATrainer
            train_dataset = CMALerobotDataset(
                config,
                config.il.lerobot_features_dir,
                config.il.use_iw,
                dataset_data=train_dataset_data,
                inflection_weight_coef=config.il.inflection_weight_coef,
                lmdb_map_size=config.il.lmdb_map_size,
                batch_size=config.il.batch_size,
            )
            collate_fn = cma_collate_fn

        elif config.model_name == 'rdp':
            policy_trainer = RDPTrainer
            train_dataset = RDP_LerobotDataset(
                config,
                config.il.lerobot_features_dir,
                dataset_data=train_dataset_data,
                batch_size=config.il.batch_size,  
            )
            collate_fn = rdp_collate_fn(global_batch_size=global_batch_size)
        elif config.model_name == 'navdp':
            policy_trainer = NavDPTrainer
            train_dataset = train_dataset_data
            collate_fn = navdp_collate_fn

        # ------------ training args ------------
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            run_name=config.name,
            remove_unused_columns=False,
            deepspeed='',
            gradient_checkpointing=False,
            bf16=False,#fp16=False,
            tf32=False,
            per_device_train_batch_size=config.il.batch_size,
            gradient_accumulation_steps=1,
            dataloader_num_workers=config.il.num_workers,
            dataloader_pin_memory=False,
            optim='adamw_torch',
            learning_rate=config.il.lr,
            lr_scheduler_type='cosine',
            logging_steps=10.0,
            num_train_epochs=config.il.epochs,
            save_strategy='epoch',# no
            save_steps=config.il.save_interval_epochs,
            save_total_limit=8,
            report_to=config.il.report_to,
            seed=0,
            do_eval=False,
            ddp_find_unused_parameters=config.il.ddp_find_unused_parameters,
            ddp_bucket_cap_mb=100,
            torch_compile_mode=None,
            dataloader_drop_last=True,
            disable_tqdm=True,
            log_level="info"
        )

        # Create the trainer
        trainer = policy_trainer(
            config=config, model=model, args=training_args, train_dataset=train_dataset, data_collator=collate_fn
        )

        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint
        run_name = config.name
        ckpt_format_callback = CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=config.log_dir)
        trainer.add_callback(ckpt_format_callback)

        trainer.train()
        if train_logger:
            for handler in train_logger.handlers:
                handler.flush()
    except Exception as e:
        import traceback
        print(f"Unhandled exception: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        
        # If distributed environment, ensure all processes exit
        if dist.is_initialized():
            dist.destroy_process_group()
        
        raise


if __name__ == '__main__':
    # Parse command line arguments using tyro
    config = tyro.cli(TrainCfg)

    # Print configuration information
    print('\n' + '=' * 50)
    print('FINE-TUNING CONFIGURATION:')
    print('=' * 50)
    for key, value in vars(config).items():
        print(f'{key}: {value}')
    print('=' * 50 + '\n')

    # Select configuration based on model_name
    supported_cfg = {
        'seq2seq': [seq2seq_exp_cfg, Seq2SeqNet, Seq2SeqModelConfig],
        'seq2seq_plus': [seq2seq_plus_exp_cfg, Seq2SeqNet, Seq2SeqModelConfig],
        'cma': [cma_exp_cfg, CMANet, CMAModelConfig],
        'cma_plus': [cma_plus_exp_cfg, CMANet, CMAModelConfig],
        'rdp': [rdp_exp_cfg, RDPNet, RDPModelConfig],
        'navdp': [navdp_exp_cfg, NavDPNet, NavDPModelConfig],
    }

    if config.model_name not in supported_cfg:
        raise ValueError(f'Invalid model name: {config.model_name}. Supported models are: {list(supported_cfg.keys())}')

    exp_cfg, model_class, model_config_class = supported_cfg[config.model_name]
    exp_cfg.name = config.name
    exp_cfg.num_gpus = len(exp_cfg.torch_gpu_ids)
    exp_cfg.world_size = exp_cfg.num_gpus

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        exp_cfg.num_gpus <= available_gpus
    ), f'Number of GPUs requested ({exp_cfg.num_gpus}) is greater than the available GPUs ({available_gpus})'
    assert exp_cfg.num_gpus > 0, 'Number of GPUs must be greater than 0'
    print(f'Using {exp_cfg.num_gpus} GPUs')

    main(exp_cfg, model_class, model_config_class)
