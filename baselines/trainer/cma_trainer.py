import torch
import torch.nn.functional as F

from internnav.trainer.base import BaseTrainer

from ..basemodels.LongCLIP.model import longclip
from ..modules.utils.bert_token import BertTokenizer


class CMATrainer(BaseTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.use_bert = False
        self.bert_tokenizer = None
        self.is_clip_long = False

        if self.config.model.policy_name == 'CMA_CLIP_Policy':
            self.use_clip_encoders = True
        else:
            self.use_clip_encoders = False

        if self.use_clip_encoders:
            self.use_bert = False
            self.bert_tokenizer = None
            self.is_clip_long = False
            if self.config.model.text_encoder.type == 'roberta':
                self.bert_tokenizer = BertTokenizer(
                    max_length=self.config.model.text_encoder.max_length,
                    load_model=self.config.model.text_encoder.load_model,
                    device=self.device,
                )
                self.use_bert = True
            elif self.config.model.text_encoder.type == 'clip-long':
                self.bert_tokenizer = longclip.tokenize
                self.use_bert = True
                self.is_clip_long = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        (
            observations_batch,
            prev_actions_batch,
            not_done_masks,
            corrected_actions_batch,
            weights_batch,
        ) = inputs

        observations_batch = {
            k: v.to(
                device=self.args.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            for k, v in observations_batch.items()
        }

        T, N = corrected_actions_batch.size()

        recurrent_hidden_states = torch.zeros(
            N,
            self.config.model.state_encoder.num_recurrent_layers,
            self.config.model.state_encoder.hidden_size,
            device=self.args.device,
        )

        batch = {
            'mode': 'train',
            'observations': observations_batch,
            'rnn_states': recurrent_hidden_states,
            'prev_actions': prev_actions_batch,
            'masks': not_done_masks,
        }

        if self.use_clip_encoders:
            depth_return_x_before_fc = False
            batch.update(
                {
                    'need_img_extraction': True,
                    'img_mod': self.config.model.image_encoder.rgb.img_mod,
                    'proj': self.config.model.image_encoder.rgb.rgb_proj,
                    'process_images': True,
                    'need_txt_extraction': True,
                    'depth_return_x_before_fc': depth_return_x_before_fc,
                }
            )

        logits, rnn_states_out, progress_hat = model(batch)

        outputs = {'logits': logits, 'rnn_states_out': rnn_states_out, 'progress_hat': progress_hat}

        logits = logits.view(T, N, -1)

        action_loss = F.cross_entropy(logits.permute(0, 2, 1), corrected_actions_batch, reduction='none')
        action_loss = ((weights_batch * action_loss).sum(0) / weights_batch.sum(0)).mean()

        # aux loss
        aux_loss = torch.tensor(0)
        if self.config.model.progress_monitor.use:
            progress_hat = progress_hat.view(T, N, -1).squeeze()
            progress_gt = observations_batch['progress'].view(T, N, -1).squeeze()
            progress_loss = F.mse_loss(
                progress_hat,
                progress_gt.to(progress_hat.device),
                reduction='none',
            )
            aux_loss = ((weights_batch * progress_loss).sum(0) / weights_batch.sum(0)).mean()

        loss = action_loss + aux_loss

        outputs['loss'] = loss

        return (loss, outputs) if return_outputs else loss
