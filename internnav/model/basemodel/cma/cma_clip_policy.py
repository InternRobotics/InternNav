import copy
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.model.encoder import (
    ImageEncoder,
    InstructionLongCLIPEncoder,
    LanguageEncoder,
    VisionLanguageEncoder,
)
from internnav.model.encoder.rnn_encoder import build_rnn_state_encoder

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

class CustomFixedCategorical(torch.distributions.Categorical):
    """Same as the CustomFixedCategorical in hab-lab, but renames log_probs
    to log_prob. All the torch distributions use log_prob.
    """

    def sample(self, sample_shape=torch.Size()) -> Tensor:  # noqa: B008
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class CMACLIPModelConfig(PretrainedConfig):
    model_type = 'cma-clip'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)

    @classmethod
    def from_dict(cls, config_dict):
        if 'model_cfg' in config_dict:
            config_dict['model_cfg'] = ExpCfg(**config_dict['model_cfg'])
        return super().from_dict(config_dict)


class CMA_CLIP_Net(PreTrainedModel):
    config_class = CMACLIPModelConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if config is pydantic model, convert to CMAModelConfig
        if hasattr(config, 'model_dump'):
            config = cls.config_class(model_cfg=config)

        model = cls(config)

        # Load pretrained weights
        if os.path.isdir(pretrained_model_name_or_path):
            pytorch_model_path = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
            safetensors_model_path = os.path.join(pretrained_model_name_or_path, 'model.safetensors')
            
            if _has_safetensors and os.path.exists(safetensors_model_path):
                try:
                    incompatible_keys, _ = model.load_state_dict(
                        safetensors.torch.load_file(safetensors_model_path)
                    )
                    print(f'Successfully loaded model from {safetensors_model_path}')
                except Exception as e:
                    print(f'Failed to load safetensors file: {e}')
                    if os.path.exists(pytorch_model_path):
                        incompatible_keys, _ = model.load_state_dict(
                            torch.load(pytorch_model_path)
                        )
                        print(f'Successfully loaded model from {pytorch_model_path}')
                    else:
                        raise FileNotFoundError(f'No model file found in {pretrained_model_name_or_path}')
            elif os.path.exists(pytorch_model_path):
                incompatible_keys, _ = model.load_state_dict(
                    torch.load(pytorch_model_path)
                )
                print(f'Successfully loaded model from {pytorch_model_path}')
            else:
                raise FileNotFoundError(f'No model file found in {pretrained_model_name_or_path}')
                
            if len(incompatible_keys) > 0:
                print(f'Incompatible keys: {incompatible_keys}')
        elif pretrained_model_name_or_path is None or len(pretrained_model_name_or_path) == 0:
            pass
        else:
            incompatible_keys, _ = model.load_state_dict(
                torch.load(pretrained_model_name_or_path)['state_dict'], strict=False
            )
            if len(incompatible_keys) > 0:
                print(f'Incompatible keys: {incompatible_keys}')

        return model

    def __init__(self, config: CMACLIPModelConfig) -> None:
        super().__init__(config)

        self.model_config = ModelCfg(**config.model_cfg['model'])
        self.num_actions = 4
        observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(256, 256, 1),
            dtype=np.float32,
        )

        # Init instruction encoder
        if self.model_config.text_encoder.model_name == 'clip-long':
            self.instruction_encoder = InstructionLongCLIPEncoder(self.model_config.text_encoder)
        else:
            if self.model_config.text_encoder.model_name in [
                'roberta',
            ]:
                config_name = 'roberta-base'
            else:
                config_name = self.model_config.text_encoder.model_name
            bert_config = PretrainedConfig.from_pretrained(config_name)
            # Init the instruction encoder
            text_encoder_config = copy.deepcopy(bert_config)
            for k, v in self.model_config.text_encoder.dict().items():
                setattr(text_encoder_config, k, v)

            self.instruction_encoder = LanguageEncoder(text_encoder_config)

        # Init the RGB & depth encoder
        self.image_encoder = ImageEncoder(
            self.model_config,
            self.model_config.image_encoder,
            observation_space,
        )
        self.rgb_proj_linear = nn.Linear(
            self.model_config.image_encoder.rgb.feature_dim,
            self.model_config.image_encoder.rgb.projection_dim,
        )
        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.model_config.image_encoder.rgb.projection_dim,
                self.model_config.image_encoder.rgb.projection_dim,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod((192, 4, 4)),
                self.model_config.image_encoder.rgb.projection_dim,
            ),
            nn.ReLU(True),
        )

        self.prev_action_embedding = nn.Embedding(self.num_actions + 1, 32)

        hidden_size = self.model_config.state_encoder.hidden_size
        self._hidden_size = hidden_size

        # Init the RNN state decoder
        rnn_input_size = self.model_config.image_encoder.rgb.projection_dim
        rnn_input_size += self.model_config.image_encoder.rgb.projection_dim
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=self.model_config.state_encoder.hidden_size,
            rnn_type=self.model_config.state_encoder.rnn_type,
            num_layers=1,
        )

        self._output_size = (
            self.model_config.state_encoder.hidden_size
            + self.model_config.state_encoder.hidden_size  # RGB
            + self.model_config.state_encoder.hidden_size  # DEPTH
            + self.model_config.state_encoder.hidden_size  # TEXT
        )

        # cross-attn for RGB, depth, and instruction
        bert_config = PretrainedConfig.from_pretrained('roberta-base')
        cross_modal_config = copy.deepcopy(bert_config)
        for k, v in self.model_config.cross_modal_encoder.dict().items():
            setattr(cross_modal_config, k, v)

        self.state_txt_cross_encoder = VisionLanguageEncoder(cross_modal_config)
        self.txt_rgb_cross_encoder = VisionLanguageEncoder(cross_modal_config)
        self.txt_depth_cross_encoder = VisionLanguageEncoder(cross_modal_config)
        self.depth_k_linear = nn.Linear(192, self.model_config.text_encoder.hidden_size)

        # second rnn
        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=self.model_config.state_encoder.rnn_type,
            num_layers=1,
        )
        self._output_size = self.model_config.state_encoder.hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

        # Determine
        self.action_distribution = CategoricalNet(self._output_size, self.num_actions)

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers + (self.second_state_encoder.num_recurrent_layers)

    def _init_layers(self) -> None:
        if self.model_config.progress_monitor.use:
            nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity='tanh')
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits = torch.einsum('nc, nci -> ni', q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum('ni, nci -> nc', attn, v)

    def img_embedding(
        self,
        rgb_inputs,
        depth_inputs,
        img_mod,
        depth_return_x_before_fc=False,
        proj=True,
        process_images=False,
        need_rgb_extraction=True,
    ):
        if process_images:
            rgb_inputs = self.image_encoder.process_image(rgb_inputs)
        if need_rgb_extraction:
            rgb_embeds = self.image_encoder.embed_image(rgb_inputs, img_mod=img_mod, proj=proj).squeeze(1)
        else:
            rgb_embeds = rgb_inputs

        depth_embeds = self.image_encoder.embed_depth(
            depth_inputs, return_x_before_fc=depth_return_x_before_fc
        ).squeeze(1)
        return rgb_embeds, depth_embeds

    def _forward(
        self,
        batch,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,  # [bs, 2, 512]
        prev_actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # 1. instruction embedding
        (
            instruction_embedding,
            txt_masks,
            text_cls_embeds,
        ) = self.instruction_encoder(observations['instruction'], need_txt_extraction=True)

        # 2. rgb & depth embedding
        rgb_features, depth_features = self.img_embedding(
            observations['rgb'],
            observations['depth'],
            batch['img_mod'],
            batch['depth_return_x_before_fc'],
            batch['proj'],
            batch['process_images'],
            need_rgb_extraction=True,
        )

        rgb_features = self.rgb_proj_linear(rgb_features)

        rgb_embedding = rgb_features.permute(0, 2, 1)
        depth_embedding = torch.flatten(depth_features, 2)

        prev_actions = self.prev_action_embedding(((prev_actions.float() + 1) * masks).long().view(-1))

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)

        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (state, rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],) = self.state_encoder(
            state_in,
            rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
            masks,
        )

        do_self_attn = True
        text_embedding, _ = self.state_txt_cross_encoder(
            state.unsqueeze(1),
            instruction_embedding,
            q_masks=masks,
            kv_masks=txt_masks,
            output_attentions=True,
            do_self_attn=do_self_attn,
        )

        rgb_embedding, _ = self.txt_rgb_cross_encoder(
            rgb_features,
            instruction_embedding,
            q_masks=None,
            kv_masks=txt_masks,
            output_attentions=True,
            do_self_attn=do_self_attn,
        )
        rgb_embedding = rgb_embedding[:, 0, :]

        depth_k_embedding = self.depth_k_linear(depth_embedding.permute(0, 2, 1))
        depth_embedding, _ = self.txt_depth_cross_encoder(
            depth_k_embedding,
            instruction_embedding,
            q_masks=None,
            kv_masks=txt_masks,
            output_attentions=True,
            do_self_attn=do_self_attn,
        )
        depth_embedding = depth_embedding[:, 0, :]

        text_embedding = text_embedding.squeeze(1)

        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                depth_embedding,
                prev_actions,
            ],
            dim=1,
        )
        x = self.second_state_compress(x)
        (x, rnn_states_out[:, self.state_encoder.num_recurrent_layers :],) = self.second_state_encoder(
            x,
            rnn_states[:, self.state_encoder.num_recurrent_layers :],
            masks,
        )

        progress_hat = None
        if self.model_config.progress_monitor.use:
            progress_hat = torch.tanh(self.progress_monitor(x))
        return x, rnn_states_out, progress_hat

    def build_distribution(self, observations, rnn_states, prev_actions, masks) -> CustomFixedCategorical:
        features, rnn_states = self.forward(observations, rnn_states, prev_actions, masks)
        return self.action_distribution(features)

    def forward(self, batch):
        x, rnn_states_out, progress_hat = self._forward(
            batch,
            batch['observations'],
            batch['rnn_states'],
            batch['prev_actions'],
            batch['masks'],
        )
        # distribution = self.action_distribution(x) # This would meet the error when using DataParallel during training "TypeError: 'CustomFixedCategorical' object is not iterable"
        if batch['mode'] == 'train':
            outputs = self.action_distribution(x).logits
        elif batch['mode'] == 'inference':
            outputs = self.action_distribution(x).mode()
        return outputs, rnn_states_out, progress_hat
