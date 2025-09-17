import os

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.model.encoder import InstructionEncoder, resnet_encoders
from internnav.model.encoder.rnn_encoder import build_rnn_state_encoder

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


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


class Seq2SeqModelConfig(PretrainedConfig):
    model_type = 'seq2seq'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)

    @classmethod
    def from_dict(cls, config_dict):
        if 'model_cfg' in config_dict:
            config_dict['model_cfg'] = ExpCfg(**config_dict['model_cfg'])
        return super().from_dict(config_dict)


class Seq2SeqNet(PreTrainedModel):
    config_class = Seq2SeqModelConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if config is a pydantic model, convert to CMAModelConfig
        if hasattr(config, 'model_dump'):
            config = cls.config_class(model_cfg=config)

        model = cls(config)

        # load pretrained weights
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
            pass  # no pretrained model
        else:
            model.load_state_dict(torch.load(pretrained_model_name_or_path))

        return model

    def __init__(self, config: Seq2SeqModelConfig):
        super().__init__(config)
        self.model_config = ModelCfg(**config.model_cfg['model'])
        self.num_actions = 4
        observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(256, 256, 1),
            dtype=np.float32,
        )

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(self.model_config.instruction_encoder)

        # Init the depth encoder
        assert self.model_config.depth_encoder.cnn_type in ['VlnResnetDepthEncoder']
        self.depth_encoder = getattr(resnet_encoders, self.model_config.depth_encoder.cnn_type)(
            observation_space,
            output_size=self.model_config.depth_encoder.output_size,
            checkpoint=self.model_config.depth_encoder.ddppo_checkpoint,
            backbone=self.model_config.depth_encoder.backbone,
            trainable=self.model_config.depth_encoder.trainable,
        )

        # Init the RGB visual encoder
        assert self.model_config.rgb_encoder.cnn_type in [
            'TorchVisionResNet18',
            'TorchVisionResNet50',
        ]
        self.rgb_encoder = getattr(resnet_encoders, self.model_config.rgb_encoder.cnn_type)(
            self.model_config.rgb_encoder.output_size,
            normalize_visual_inputs=self.model_config.normalize_rgb,
            trainable=self.model_config.rgb_encoder.trainable,
            spatial_output=False,
        )

        if self.model_config.seq2seq.use_prev_action:
            self.prev_action_embedding = nn.Embedding(self.num_actions + 1, 32)

        # Init the RNN state decoder
        rnn_input_size = (
            self.instruction_encoder.output_size
            + self.model_config.depth_encoder.output_size
            + self.model_config.rgb_encoder.output_size
        )

        if self.model_config.seq2seq.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=self.model_config.state_encoder.hidden_size,
            rnn_type=self.model_config.state_encoder.rnn_type,
            num_layers=1,
        )

        self.progress_monitor = nn.Linear(self.model_config.state_encoder.hidden_size, 1)

        self._init_layers()

        self.train()

        # Determine
        self.action_distribution = CategoricalNet(self.output_size, self.num_actions)

    @property
    def output_size(self):
        return self.model_config.state_encoder.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _init_layers(self):
        nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity='tanh')
        nn.init.constant_(self.progress_monitor.bias, 0)

    def _forward(self, observations, rnn_states, prev_actions, masks):
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        x = torch.cat([instruction_embedding, depth_embedding, rgb_embedding], dim=1)

        if self.model_config.seq2seq.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(((prev_actions.float() + 1) * masks).long().view(-1))
            x = torch.cat([x, prev_actions_embedding], dim=1)

        x, rnn_states_out = self.state_encoder(x, rnn_states, masks)

        if self.model_config.progress_monitor.use:
            progress_hat = torch.tanh(self.progress_monitor(x))
        return x, rnn_states_out, progress_hat

    def build_distribution(self, observations, rnn_states, prev_actions, masks) -> CustomFixedCategorical:
        features, rnn_states = self.forward(observations, rnn_states, prev_actions, masks)
        return self.action_distribution(features)

    def forward(self, batch):
        x, rnn_states_out, progress_hat = self._forward(
            batch['observations'],
            batch['rnn_states'],
            batch['prev_actions'],
            batch['masks'],
        )
        if batch['mode'] == 'train':
            outputs = self.action_distribution(x).logits
        elif batch['mode'] == 'inference':
            outputs = self.action_distribution(x).mode()
        return outputs, rnn_states_out, progress_hat
