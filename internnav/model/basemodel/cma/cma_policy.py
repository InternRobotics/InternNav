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

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

from internnav.configs.model.base_encoders import ModelCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.model.encoder import (
    InstructionEncoder,
    InstructionLongCLIPEncoder,
    LanguageEncoder,
    resnet_encoders,
)
from internnav.model.encoder.rnn_encoder import build_rnn_state_encoder


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()) -> Tensor:
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


class CMAModelConfig(PretrainedConfig):
    model_type = 'cma'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)

    @classmethod
    def from_dict(cls, config_dict):
        if 'model_cfg' in config_dict:
            config_dict['model_cfg'] = ExpCfg(**config_dict['model_cfg'])
        return super().from_dict(config_dict)


class CMANet(PreTrainedModel):
    config_class = CMAModelConfig

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

    def __init__(self, config: CMAModelConfig):
        super().__init__(config)

        if isinstance(config, CMAModelConfig):
            self.model_config = ModelCfg(**config.model_cfg['model'])
        else:
            self.model_config = config
        self.num_actions = 4
        observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(256, 256, 1),
            dtype=np.float32,
        )
        self.model_config.instruction_encoder.final_state_only = False

        self.use_instr_bert_encoder = False
        if self.model_config.policy_name == 'CMA_CLIP_Policy':
            if self.model_config.text_encoder.model_name == 'clip-long':
                self.instruction_encoder = InstructionLongCLIPEncoder(self.model_config.text_encoder)
                self.txt_linear_512_to_256 = nn.Linear(512, 256)
                self.instruction_encoder.output_size = 256
            else:
                if self.model_config.text_encoder.model_name in ['roberta']:
                    config_name = 'roberta-base'
                else:
                    config_name = self.model_config.text_encoder.model_name
                bert_config = PretrainedConfig.from_pretrained(config_name)
                text_encoder_config = copy.deepcopy(bert_config)
                for k, v in self.model_config.text_encoder.dict().items():
                    setattr(text_encoder_config, k, v)
                self.instruction_encoder = LanguageEncoder(text_encoder_config)
            self.use_instr_bert_encoder = True
        else:
            self.instruction_encoder = InstructionEncoder(self.model_config.instruction_encoder)

        self.depth_encoder = getattr(resnet_encoders, self.model_config.depth_encoder.cnn_type)(
            observation_space,
            output_size=self.model_config.depth_encoder.output_size,
            checkpoint=self.model_config.depth_encoder.ddppo_checkpoint,
            backbone=self.model_config.depth_encoder.backbone,
            trainable=self.model_config.depth_encoder.trainable,
            spatial_output=True,
        )

        self.rgb_encoder = getattr(resnet_encoders, self.model_config.rgb_encoder.cnn_type)(
            self.model_config.rgb_encoder.output_size,
            normalize_visual_inputs=self.model_config.normalize_rgb,
            trainable=self.model_config.rgb_encoder.trainable,
            spatial_output=True,
        )

        self.prev_action_embedding = nn.Embedding(self.num_actions + 1, 32)

        hidden_size = self.model_config.state_encoder.hidden_size
        self._hidden_size = hidden_size

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.rgb_encoder.output_shape[0], self.model_config.rgb_encoder.output_size),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.depth_encoder.output_shape), self.model_config.depth_encoder.output_size),
            nn.ReLU(True),
        )

        rnn_input_size = (
            self.model_config.depth_encoder.output_size
            + self.model_config.rgb_encoder.output_size
            + self.prev_action_embedding.embedding_dim
        )

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=self.model_config.state_encoder.rnn_type,
            num_layers=1,
        )

        self._output_size = (
            hidden_size
            + self.model_config.rgb_encoder.output_size
            + self.model_config.depth_encoder.output_size
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0], hidden_size // 2 + self.model_config.rgb_encoder.output_size, 1
        )
        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0], hidden_size // 2 + self.model_config.depth_encoder.output_size, 1
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(self.instruction_encoder.output_size, hidden_size // 2, 1)
        self.text_q = nn.Linear(self.instruction_encoder.output_size, hidden_size // 2)

        self.register_buffer('_scale', torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))

        self.second_state_compress = nn.Sequential(
            nn.Linear(self._output_size + self.prev_action_embedding.embedding_dim, hidden_size),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            rnn_type=self.model_config.state_encoder.rnn_type,
            num_layers=1,
        )

        self._output_size = hidden_size

        self.progress_monitor = nn.Linear(hidden_size, 1)
        self._init_layers()

        self.action_distribution = CategoricalNet(hidden_size, self.num_actions)

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers + self.second_state_encoder.num_recurrent_layers

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

    def _forward(
        self, observations: Dict[str, Tensor], rnn_states: Tensor, prev_actions: Tensor, masks: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if self.use_instr_bert_encoder:
            instr = observations['instruction']
            instruction_embedding, txt_masks, txt_cls_embeds = self.instruction_encoder(instr)
            instruction_embedding = self.txt_linear_512_to_256(instruction_embedding)
            instruction_embedding = instruction_embedding.permute(0, 2, 1)
        else:
            instruction_embedding = self.instruction_encoder(observations)

        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)

        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)

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
        state, rnn_states_out[:, : self.state_encoder.num_recurrent_layers] = self.state_encoder(
            state_in, rnn_states[:, : self.state_encoder.num_recurrent_layers], masks
        )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(text_state_q, text_state_k, instruction_embedding, text_mask)

        rgb_k, rgb_v = torch.split(self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1)
        depth_k, depth_v = torch.split(self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1)
        text_q = self.text_q(text_embedding)

        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat([state, text_embedding, rgb_embedding, depth_embedding, prev_actions], dim=1)
        x = self.second_state_compress(x)
        x, rnn_states_out[:, self.state_encoder.num_recurrent_layers :] = self.second_state_encoder(
            x, rnn_states[:, self.state_encoder.num_recurrent_layers :], masks
        )

        progress_hat = None
        if self.model_config.progress_monitor.use:
            progress_hat = torch.tanh(self.progress_monitor(x))

        return x, rnn_states_out, progress_hat

    def build_distribution(self, observations, rnn_states, prev_actions, masks) -> CustomFixedCategorical:
        features, rnn_states, _ = self._forward(observations, rnn_states, prev_actions, masks)
        return self.action_distribution(features)

    def forward(self, batch):
        x, rnn_states_out, progress_hat = self._forward(
            batch['observations'], batch['rnn_states'], batch['prev_actions'], batch['masks']
        )
        if batch['mode'] == 'train':
            outputs = self.action_distribution(x).logits
        elif batch['mode'] == 'inference':
            outputs = self.action_distribution(x).mode()
        else:
            outputs = x
        return outputs, rnn_states_out, progress_hat
