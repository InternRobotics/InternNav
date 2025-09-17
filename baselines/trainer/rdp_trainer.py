import numpy as np
import torch
import torch.nn.functional as F

from internnav.modules.basemodel.LongCLIP.model import longclip
from internnav.trainer.base import BaseTrainer


def action_reduce(action_mask, unreduced_loss: torch.Tensor):
    # Reduce over non-batch dimensions to get loss per batch element
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    assert unreduced_loss.shape == action_mask.shape, f'{unreduced_loss.shape} != {action_mask.shape}'
    return (unreduced_loss * action_mask).mean() / (action_mask.float().mean() + 1e-2)


class RDPTrainer(BaseTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.bert_tokenizer = longclip.tokenize
        self.use_bert = True
        self.is_clip_long = True

        if self.config.model.learn_angle:
            self.action_dim = 3
        else:
            self.action_dim = 2

        # Init the action stats
        self.action_stats = None
        if hasattr(self.config.model, 'Diffusion_Policy'):
            self.action_stats = {
                'min': torch.Tensor(np.asarray(self.config.model.Diffusion_Policy.action_stats.min)).to(self.device),
                'max': torch.Tensor(np.asarray(self.config.model.Diffusion_Policy.action_stats.max)).to(self.device),
            }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        (
            observations_batch,
            prev_actions_batch,
            not_done_masks,
            N,
        ) = inputs

        N = N[0][0]

        observations_batch = {
            k: v.to(
                device=self.args.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            for k, v in observations_batch.items()
        }

        # N = self.config.il.batch_size
        masks = not_done_masks

        recurrent_hidden_states = torch.zeros(
            N,
            self.config.model.state_encoder.num_recurrent_layers,
            self.config.model.state_encoder.hidden_size,
            device=self.args.device,
        )

        need_img_extraction = True
        depth_return_x_before_fc = True

        batch = {
            'mode': 'pred_actions',
            'observations': observations_batch,
            'rnn_states': recurrent_hidden_states,
            'prev_actions': prev_actions_batch,
            'masks': not_done_masks,
            'add_noise_to_action': True,
            'denoise_action': False,
            'need_img_extraction': need_img_extraction,  # need img embedding in model
            'depth_return_x_before_fc': depth_return_x_before_fc,
            'img_mod': self.config.model.image_encoder.rgb.img_mod,
            'proj': self.config.model.image_encoder.rgb.rgb_proj,
            'process_images': False,  # has processed in dataLoader
            'train_cls_free_guidance': self.config.model.diffusion_policy.use_cls_free_guidance,
            'sample_cls_free_guidance': False,
            'need_txt_extraction': True,
        }

        (
            noise_pred,
            dist_pred,
            rnn_states_out,
            noise,
            diffusion_output,
            progress_hat,
            denoise_action_list,
            stop_progress_pred,
        ) = model(batch)

        outputs = {
            'rnn_states_out': rnn_states_out,
            'noise': noise,
            'noise_pred': noise_pred,
            'progress_hat': progress_hat,
            'stop_progress_pred': stop_progress_pred,
        }

        # L2 loss
        masks_unsqueeze = masks.squeeze() if masks is not None else None
        if self.config.model.diffusion_policy.pred_type == 'epsilon' and self.config.model.diffusion_policy.use:
            # pred noise
            f_loss = F.mse_loss(noise_pred, noise, reduction='none')
            diffusion_loss = action_reduce(masks_unsqueeze, f_loss)
        elif self.config.model.diffusion_policy.pred_type == 'sample' or not self.config.model.diffusion_policy.use:
            # pred x_0
            f_loss = F.mse_loss(noise_pred, observations_batch['actions'], reduction='none')
            diffusion_loss = action_reduce(masks_unsqueeze, f_loss)

        # Aux loss
        pm_loss = 0
        if self.config.model.progress_monitor.use:
            progress_loss = F.mse_loss(
                progress_hat.squeeze(),
                observations_batch['progress'].to(progress_hat.device),
                reduction='none',
            )
            pm_loss = action_reduce(masks_unsqueeze, progress_loss)

        stop_pm_loss = 0
        if self.config.model.stop_progress_predictor.use:
            stop_pm_loss = F.mse_loss(
                stop_progress_pred.squeeze(),
                observations_batch['stop_progress'].to(stop_progress_pred.device).squeeze(),
                reduction='none',
            )
            stop_pm_loss = action_reduce(masks_unsqueeze, stop_pm_loss)
            stop_pm_loss *= self.config.model.stop_progress_predictor.loss_alpha

        # Total loss
        loss = diffusion_loss + pm_loss + stop_pm_loss

        outputs['loss'] = loss

        return (loss, outputs) if return_outputs else loss
