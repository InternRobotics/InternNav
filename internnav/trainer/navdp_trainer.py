import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from internnav.trainer.base import BaseTrainer


class NavDPTrainer(BaseTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.writer = None
        # self.iterations = config.checkpoint * 141
        self.start_time = time.time()
        # add device attribute
        if hasattr(self.model, 'module'):  # DDP wrapped model
            self.model_device = self.model.module.device
        else:
            self.model_device = self.model.device

        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model device: {self.model_device}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # get model device
        model_device = next(model.parameters()).device

        # ensure all inputs are on the model device
        inputs_on_device = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                # use non_blocking=True to improve efficiency
                inputs_on_device[key] = value.to(model_device, non_blocking=True)
            else:
                inputs_on_device[key] = value

        import os

        import psutil

        current_pid = os.getpid()
        process = psutil.Process(current_pid)
        parent = process.parent()

        if parent:
            children = parent.children()
            if len(children) == 8:
                print("There are 8 training processes running")
            else:
                print(f"There are {len(children)} training processes running")
        else:
            print("Cannot determine parent process")

        # Ensure all inputs are on the model device
        inputs_on_device = {
            "batch_pg": inputs["batch_pg"].to(model_device),
            "batch_ig": inputs["batch_ig"].to(model_device),
            "batch_tg": inputs["batch_tg"].to(model_device),
            "batch_rgb": inputs["batch_rgb"].to(model_device),
            "batch_depth": inputs["batch_depth"].to(model_device),
            "batch_labels": inputs["batch_labels"].to(model_device),
            "batch_augments": inputs["batch_augments"].to(model_device),
            "batch_label_critic": inputs["batch_label_critic"].to(model_device),
            "batch_augment_critic": inputs["batch_augment_critic"].to(model_device),
        }
        torch.cuda.synchronize(model_device)

        # unpack input data and move to device
        # batch_pg = inputs["batch_pg"]
        # batch_ig = inputs["batch_ig"]
        # batch_rgb = inputs["batch_rgb"]
        # batch_depth = inputs["batch_depth"]
        # batch_labels = inputs["batch_labels"]
        # batch_augments = inputs["batch_augments"]
        batch_label_critic = inputs["batch_label_critic"]
        batch_augment_critic = inputs["batch_augment_critic"]

        pred_ng, pred_mg, critic_pred, augment_pred, noise, aux_pred = model(
            inputs_on_device["batch_pg"],
            inputs_on_device["batch_ig"],
            inputs_on_device["batch_tg"],
            inputs_on_device["batch_rgb"],
            inputs_on_device["batch_depth"],
            inputs_on_device["batch_labels"],
            inputs_on_device["batch_augments"],
        )

        ng_action_loss = (pred_ng - noise[0]).square().mean()
        mg_action_loss = (pred_mg - noise[1]).square().mean()
        aux_loss = (
            0.5 * (inputs_on_device["batch_pg"] - aux_pred[0]).square().mean()
            + 0.5 * (inputs_on_device["batch_pg"] - aux_pred[1]).square().mean()
        )
        action_loss = 0.5 * mg_action_loss + 0.5 * ng_action_loss
        critic_loss = (critic_pred - batch_label_critic).square().mean() + (
            augment_pred - batch_augment_critic
        ).square().mean()
        loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss

        outputs = {
            'pred_ng': pred_ng,
            'pred_mg': pred_mg,
            'critic_pred': critic_pred,
            'augment_pred': augment_pred,
            'noise': noise,
            'loss': loss,
            'ng_action_loss': ng_action_loss,
            'mg_action_loss': mg_action_loss,
            'aux_loss': aux_loss,
            'critic_loss': critic_loss,
        }
        # if self.logger:
        #     self.logger.info(
        #         f"[Step {self.state.global_step}] "
        #         f"Loss: {loss.item():.4f}, "
        #         f"Action Loss: {action_loss.item():.4f}, "
        #         f"Critic Loss: {critic_loss.item():.4f}"
        #     )

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """create and return optimizer"""
        rank = dist.get_rank() if dist.is_initialized() else 0

        # get learning rate
        try:
            lr = self.config.il.lr
            if rank == 0:
                print(f"[Rank 0] Using learning rate: {lr}")
        except AttributeError:
            lr = 1e-4
            if rank == 0:
                print(f"[Rank 0] Warning: Using default learning rate: {lr}")

        # Ensure the model is on the correct device
        if hasattr(self.model, 'module'):
            model_for_optim = self.model.module
        else:
            model_for_optim = self.model

        # Create optimizer
        optimizer = torch.optim.Adam(model_for_optim.parameters(), lr=lr)

        if rank == 0:
            print(f"[Rank 0] Optimizer created with {len(optimizer.param_groups)} param groups")
            total_params = sum(p.numel() for p in model_for_optim.parameters() if p.requires_grad)
            print(f"[Rank 0] Total trainable parameters: {total_params:,}")

        return optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        """Create learning rate scheduler"""
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=10000)
        return scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """override parent class method, completely control the creation process"""
        print("\n=== create optimizer and scheduler ===")

        # create optimizer
        self.optimizer = self.create_optimizer()

        # create scheduler (note the parameter order)
        self.lr_scheduler = self.create_scheduler(self.optimizer, num_training_steps)

        return self.optimizer, self.lr_scheduler

    def get_train_dataloader(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=1234)

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.il.batch_size,
            sampler=sampler,
            num_workers=self.config.il.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.data_collator,
        )
        # print(loader)
        return loader

    def save_model(self, output_dir, state_dict=None, **kwargs):
        """
        save model to specified directory

        handle DDP wrapped model
        """
        # check if it is a DDP wrapped model
        if hasattr(self.model, 'module'):
            # get original model
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model_to_save.state_dict(), output_dir + "navdp.ckpt")

        print(f"Saving model to {output_dir} (is DDP: {hasattr(self.model, 'module')})")
