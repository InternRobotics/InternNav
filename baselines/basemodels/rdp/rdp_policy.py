import copy
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from gym import spaces
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg

from ...basemodel.diffusion_policy_modified.transformer_for_diffusion_modified import (
    TransformerForDiffusion,
)
from ...encoder import (
    DistanceNetwork,
    ImageEncoder,
    InstructionLongCLIPEncoder,
    LanguageEncoder,
    PositionalEncoding,
    VisionLanguageEncoder,
)
from ...encoder.rnn_encoder import build_rnn_state_encoder
from ...utils.utils import get_action

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

action_spaces = {
    'stop': [0],
    'go_forward': [1],
    'turn_left': [2],
    'turn_right': [3],
    'wait': [4],
}


class RDPModelConfig(PretrainedConfig):
    model_type = 'rdp'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)

    @classmethod
    def from_dict(cls, config_dict):
        if 'model_cfg' in config_dict:
            config_dict['model_cfg'] = ExpCfg(**config_dict['model_cfg'])
        return super().from_dict(config_dict)


class RDPNet(PreTrainedModel):
    config_class = RDPModelConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

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

    def __init__(self, config: RDPModelConfig):
        super().__init__(config)
        self.model_config = ModelCfg(**config.model_cfg['model'])
        self.model_config.eval = EvalCfg(**config.model_cfg['model']['eval'])
        if self.model_config.learn_angle:
            self.num_actions = 3
        else:
            self.num_actions = 2

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(256, 256, 1),
            dtype=np.float32,
        )

        if hasattr(self.model_config, 'diffusion_policy'):
            self.action_stats = {
                'min': torch.Tensor(np.asarray(self.model_config.diffusion_policy.action_stats.min)),
                'max': torch.Tensor(np.asarray(self.model_config.diffusion_policy.action_stats.max)),
            }

        self.model_config.text_encoder.final_state_only = False

        if self.model_config.text_encoder.model_name == 'clip-long':
            self.instruction_encoder = InstructionLongCLIPEncoder(self.model_config.text_encoder)
        else:
            if self.model_config.text_encoder.model_name in ['roberta']:
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
            self.observation_space,
        )

        # Init the cross-modal fusion network
        bert_config = PretrainedConfig.from_pretrained('roberta-base')
        cross_modal_config = copy.deepcopy(bert_config)
        try:
            for k, v in self.model_config.cross_modal_encoder.dict().items():
                setattr(cross_modal_config, k, v)
        except Exception as e:
            for k, v in dict(self.model_config.cross_modal_encoder).items():
                setattr(cross_modal_config, k, v)

        if self.model_config.cross_modal_encoder.txt_to_img:
            txt_to_img_cross_encoder_config = copy.deepcopy(cross_modal_config)
            txt_to_img_cross_encoder_config.num_x_layers = self.model_config.cross_modal_encoder.txt_to_img_layer
            self.txt_img_cross_encoder = VisionLanguageEncoder(txt_to_img_cross_encoder_config)
        self.img_txt_cross_encoder = VisionLanguageEncoder(cross_modal_config)

        # Init the prev action embedding
        prev_action_encoder_size = self.model_config.prev_action_encoder.encoding_size
        self.prev_action_embedding = nn.Linear(self.num_actions, prev_action_encoder_size)
        self.prev_action_embedding_dp = nn.Linear(self.num_actions, self.model_config.state_encoder.hidden_size)
        self.prev_act_ln = nn.LayerNorm(self.model_config.prev_action_encoder.encoding_size)
        self.prev_action_pos_embedding = PositionalEncoding(
            self.model_config.prev_action_encoder.encoding_size,
            self.model_config.len_traj_act,
        )

        if self.model_config.image_encoder.rgb.img_mod == 'cls':
            concat_size = self.model_config.image_encoder.rgb.projection_dim
        elif self.model_config.image_encoder.rgb.img_mod == 'multi_patches_avg_pooling':
            if self.model_config.state_encoder.rgb_depth_embed_method == 'flat':
                concat_size = (
                    self.model_config.image_encoder.rgb.projection_dim
                    * self.model_config.image_encoder.rgb.multi_patches_num
                )
            elif self.model_config.state_encoder.rgb_depth_embed_method == 'first':
                concat_size = self.model_config.image_encoder.rgb.projection_dim

        # Init the IMU encoder
        if self.model_config.imu_encoder.use:
            self.imu_linear = nn.Linear(
                self.model_config.imu_encoder.input_size,
                self.model_config.imu_encoder.encoding_size,
            )
            self.imu_linear_dp = nn.Linear(
                self.model_config.imu_encoder.input_size,
                self.model_config.state_encoder.hidden_size,
            )  # This is used to encode IMU to hidden_states as the inputs for diffusion transformer
            concat_size += self.model_config.imu_encoder.encoding_size

        concat_size += self.model_config.prev_action_encoder.encoding_size

        # Init the GRU network
        self.state_encoder = build_rnn_state_encoder(
            input_size=concat_size,
            hidden_size=self.model_config.state_encoder.hidden_size,
            rnn_type=self.model_config.state_encoder.rnn_type,
            num_layers=self.model_config.state_encoder.num_recurrent_layers,
        )

        if self.model_config.state_encoder.use_dropout:
            self.state_dropout = nn.Dropout(self.model_config.state_encoder.dropout_rate)

        # Init the diffusion policy network
        self.dp_type = self.model_config.diffusion_policy.type
        if self.model_config.diffusion_policy.type == 'transformer':
            self.use_cls_free_guidance = self.model_config.diffusion_policy.use_cls_free_guidance
            # define the length of conditions
            if self.model_config.image_encoder.rgb.img_mod == 'cls':
                vis_length = 1
            elif self.model_config.image_encoder.rgb.img_mod == 'multi_patches_avg_pooling':
                vis_length = self.model_config.image_encoder.rgb.multi_patches_num

            txt_length = 1
            rnn_length = 1
            prev_act_length = self.model_config.len_traj_act
            imu_length = 1 if self.model_config.imu_encoder.use else 0

            n_obs_steps = rnn_length + txt_length + vis_length + 1 + imu_length + prev_act_length
            self.action_dp_pred_net = TransformerForDiffusion(
                input_dim=self.num_actions,
                output_dim=self.num_actions,
                horizon=self.model_config.diffusion_policy.len_traj_pred,
                n_obs_steps=n_obs_steps,
                n_emb=self.model_config.diffusion_policy.transformer_encoding_size,
                p_drop_emb=self.model_config.diffusion_policy.transformer_p_drop_emb,
                cond_dim=self.model_config.state_encoder.hidden_size,
                causal_attn=True,
                time_as_cond=True,
                n_layer=self.model_config.diffusion_policy.transformer_n_layers,
                n_cond_layers=self.model_config.diffusion_policy.transformer_n_cond_layers,
                use_dp=self.model_config.diffusion_policy.use,
            )
            self.action_type_embeds = nn.Embedding(
                10,
                self.model_config.diffusion_policy.transformer_encoding_size,
            )

        if self.model_config.diffusion_policy.scheduler == 'DDPM':
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_config.diffusion_policy.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type=self.model_config.diffusion_policy.pred_type,
            )

        # Init the distance prediction network
        if self.model_config.distance_predictor.use:
            self.distance_pred_net = DistanceNetwork(
                embedding_dim=self.model_config.state_encoder.hidden_size,
                normalize=self.model_config.distance_predictor.normalize,
            )

        if self.model_config.progress_monitor.use:
            if self.model_config.progress_monitor.concat_state_txt:
                self.progress_monitor = DistanceNetwork(
                    embedding_dim=self.model_config.state_encoder.hidden_size * 2,
                    normalize=True,
                )  # pm_pred

            self._init_pm_layers(self.progress_monitor)

        # Init the stop progress predictor
        if self.model_config.stop_progress_predictor.use:
            if self.model_config.stop_progress_predictor.concat_state_txt:
                stop_hidden_dim = self.model_config.state_encoder.hidden_size * 2

            if self.model_config.stop_progress_predictor.type == 'continuous':
                self.stop_progress_predictor = DistanceNetwork(
                    embedding_dim=stop_hidden_dim, normalize=True
                )  # stop_progress_pred

            self._init_pm_layers(self.stop_progress_predictor)

        self._output_size = self.num_actions

        self.train()

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return False

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers

    def _init_pm_layers(self, pm_net) -> None:
        for param in pm_net.parameters():
            if param.ndim == 2:  # Typically weights are 2D
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif param.ndim == 1:  # Typically biases are 1D
                nn.init.constant_(param, 0)

    def denoise_actions(
        self,
        noisy_diffusion_output,
        lv_state,
        type_embeds,
        device,
        sample_classifier_free_guidance=False,
        cls_free_guidance_scale=4,
        cond_mask=None,
        y_cond=None,
        y_cond_mask=None,
    ):
        batch_size = noisy_diffusion_output.shape[0]
        diffusion_output = noisy_diffusion_output

        for k in self.noise_scheduler.timesteps[:]:
            noise_pred = self.action_dp_pred_net(
                sample=diffusion_output,
                timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                cond=lv_state.float(),
                type_embeds=type_embeds,
                cond_mask=cond_mask,
                y_cond=y_cond,
                y_cond_mask=y_cond_mask,
            )

            if k != 0 and sample_classifier_free_guidance:
                noise_out, noise_out_null = (
                    noise_pred[: batch_size // 2],
                    noise_pred[batch_size // 2 :],
                )
                noise_out = noise_out_null + cls_free_guidance_scale * (noise_out - noise_out_null)
                noise_pred = torch.cat([noise_out, noise_out], dim=0)

            # inverse diffusion step (remove noise)
            diffusion_output = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=diffusion_output,
            ).prev_sample

        if sample_classifier_free_guidance:
            diffusion_output = diffusion_output[: batch_size // 2]

        return diffusion_output

    def pred_actions(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
        add_noise_to_action=True,
        denoise_action=False,
        num_sample=1,
        train_classifier_free_guidance=False,
        sample_classifier_free_guidance=False,
        need_txt_extraction=True,
    ):
        device = observations['instruction'].device
        batch_size = observations['instruction'].shape[0]

        """1. Encoding text"""
        text_embeds, txt_masks, text_cls_embeds = self.instruction_encoder(
            observations['instruction'],
            need_txt_extraction=need_txt_extraction,
        )

        """2. Encoding previous actions and steps"""
        prev_actions_masks = prev_actions.float() * masks.unsqueeze(-1).float()  # [bs, act_length, 3]
        prev_action_embeds = self.prev_action_embedding(prev_actions_masks)
        prev_action_dp_embeds = self.prev_action_embedding_dp(prev_actions_masks)
        latest_prev_action_embeds = prev_action_embeds[:, 0]

        """3. Encoding images"""
        rgb_depth_embeds = self.image_encoder(
            observations['stack_rgb'],
            observations['stack_depth'],
            img_mod=self.model_config.image_encoder.rgb.img_mod,
        )

        """4. Update GRU"""
        if self.model_config.image_encoder.rgb.img_mod == 'multi_patches_avg_pooling':
            if self.model_config.state_encoder.rgb_depth_embed_method == 'flat':
                rgb_depth_for_rnn = torch.flatten(rgb_depth_embeds, 1)  # [bs, 5, h_dim] -> [bs, 5*h_dim]
            elif self.model_config.state_encoder.rgb_depth_embed_method == 'first':
                rgb_depth_for_rnn = rgb_depth_embeds[:, 0, :]
        else:
            rgb_depth_for_rnn = rgb_depth_embeds.squeeze(1)
        concat_embeds = torch.cat([rgb_depth_for_rnn, latest_prev_action_embeds], dim=1)
        if self.model_config.imu_encoder.use:
            imu_embeds = self.imu_linear(observations['imu'])
            imu_dp_embeds = self.imu_linear_dp(observations['imu'])

            # Concat GRU input features
            concat_embeds = torch.cat([concat_embeds, imu_embeds], dim=1)

        """5. Compute GRU features"""
        state, rnn_states_out = self.state_encoder(concat_embeds, rnn_states, masks.bool())
        state = state.unsqueeze(1)

        if self.model_config.state_encoder.use_dropout:
            state = self.state_dropout(state)

        """6. Encoding vision-and-language"""
        if (
            not self.model_config.image_encoder.use_stack
            and self.model_config.image_encoder.rgb.img_mod != 'multi_patches_avg_pooling'
        ):
            do_self_attn = False
        else:
            do_self_attn = True

        # 6.1 Current img features combine with the text features
        rgb_depth_his_embeds = torch.cat((rgb_depth_embeds, state), dim=1)
        try:
            img_txt_embeds, img_txt_attn_probs = self.img_txt_cross_encoder(
                rgb_depth_his_embeds,
                text_embeds,
                q_masks=masks,
                kv_masks=txt_masks,
                output_attentions=True,
                do_self_attn=do_self_attn,
            )
        except Exception as e:
            print(e)
            img_txt_embeds, img_txt_attn_probs = self.img_txt_cross_encoder(
                rgb_depth_his_embeds,
                text_embeds,
                q_masks=masks,
                kv_masks=txt_masks,
                output_attentions=True,
                do_self_attn=do_self_attn,
            )
        img_txt_attn_probs = img_txt_attn_probs[:, 0, :]

        # 6.2 Current text features combine with the current img features
        if self.model_config.cross_modal_encoder.txt_to_img:
            txt_img_embeds, txt_img_attn_probs = self.txt_img_cross_encoder(
                text_embeds,
                rgb_depth_his_embeds,
                q_masks=txt_masks,
                kv_masks=None,
                output_attentions=True,
                do_self_attn=do_self_attn,
            )
            fused_update_txt_embeds = txt_img_embeds
        else:
            fused_update_txt_embeds = text_embeds

        """7. Predict action distribution using diffusion policy"""
        # Sample a diffusion iteration for each data point
        if self.model_config.diffusion_policy.use:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()
        else:
            timesteps = None

        noise = noise_pred = diffusion_output = None
        denoise_action_list = []

        if denoise_action:
            # initialize action from Gaussian noise
            noisy_diffusion_output = torch.randn(
                (
                    batch_size,
                    self.model_config.diffusion_policy.len_traj_pred,
                    self.num_actions,
                ),
                device=device,
            )
            noise = copy.deepcopy(noisy_diffusion_output)
            diffusion_output = noisy_diffusion_output
            # predict noise
            if self.dp_type == 'transformer':
                if self.model_config.imu_encoder.use:
                    imu_dp_embeds = imu_dp_embeds.unsqueeze(1)
                prev_action_dp_embeds = prev_action_dp_embeds
                txt_dp_embeds = fused_update_txt_embeds[:, 0, :].unsqueeze(1)
                lv_state = torch.cat([img_txt_embeds, txt_dp_embeds, state], dim=1)
                if self.model_config.imu_encoder.use:
                    lv_state = torch.cat([lv_state, imu_dp_embeds], dim=1)
                lv_state = torch.cat([lv_state, prev_action_dp_embeds], dim=1)

                type_embeds = [0] * img_txt_embeds.shape[1] + [1] * txt_dp_embeds.shape[1] + [2] * state.shape[1]

                if self.model_config.imu_encoder.use:
                    type_embeds += [4] * imu_dp_embeds.shape[1]
                type_embeds += [5] * prev_action_dp_embeds.shape[1]

                type_embeds = torch.from_numpy(np.array(type_embeds)).to(device)
                type_embeds = self.action_type_embeds(type_embeds).repeat(batch_size, 1, 1)

                if (
                    sample_classifier_free_guidance
                    and self.model_config.diffusion_policy.cls_mask_method == 'mask_token'
                ):
                    uncond_mask = torch.zeros(batch_size, lv_state.shape[1]).to(device)
                    if self.model_config.diffusion_policy.random_mask_instr:
                        uncond_mask[
                            :,
                            img_txt_embeds.shape[1] : img_txt_embeds.shape[1] + txt_dp_embeds.shape[1],
                        ] = 1
                    if self.model_config.diffusion_policy.random_mask_rgb:
                        uncond_mask[:, : img_txt_embeds.shape[1]] = 1

                    t_token_mask = torch.zeros(batch_size, 1).to(device)
                    uncond_mask = torch.cat([t_token_mask, uncond_mask], dim=1)

                    cond_mask = torch.zeros_like(uncond_mask)
                    cond_mask = torch.cat([cond_mask, uncond_mask], dim=0)
                    batch_size = cond_mask.shape[0]

                    type_embeds = torch.cat([type_embeds, type_embeds], dim=0)
                    lv_state = torch.cat([lv_state, lv_state], dim=0)
                else:
                    cond_mask = None

            if self.model_config.diffusion_policy.use:  # use standard diffusion policy
                # initialize action from Gaussian noise
                if sample_classifier_free_guidance:
                    noise_bs = batch_size // 2
                    noisy_diffusion_output = torch.randn(
                        (
                            noise_bs,
                            self.model_config.diffusion_policy.len_traj_pred,
                            self.num_actions,
                        ),
                        device=device,
                    )
                    noisy_diffusion_output = torch.cat(
                        [noisy_diffusion_output, noisy_diffusion_output],
                        dim=0,
                    )
                else:
                    noise_bs = batch_size
                    noisy_diffusion_output = torch.randn(
                        (
                            noise_bs,
                            self.model_config.diffusion_policy.len_traj_pred,
                            self.num_actions,
                        ),
                        device=device,
                    )
                diffusion_output = self.denoise_actions(
                    noisy_diffusion_output,
                    lv_state,
                    type_embeds,
                    device,
                    sample_classifier_free_guidance,
                    cls_free_guidance_scale=self.model_config.diffusion_policy.cls_free_guidance_scale,
                    cond_mask=cond_mask,
                )

            else:  # directly regress the action
                noisy_action = None
                diffusion_output = self.action_dp_pred_net(
                    sample=noisy_action,
                    timestep=timesteps,
                    cond=lv_state.float(),
                    type_embeds=type_embeds,
                    cond_mask=None,
                )

        else:
            if self.model_config.diffusion_policy.use and add_noise_to_action:
                # Add noise to the clean images according to the noise magnitude at each diffusion iterationd
                # Sample noise to add to actions
                naction = observations['actions']  # which has been normalized
                noise = torch.randn(naction.shape, device=device)
                noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
            else:
                noisy_action = observations['actions']

            if self.dp_type == 'transformer':
                if self.model_config.imu_encoder.use:
                    imu_dp_embeds = imu_dp_embeds.unsqueeze(1)
                prev_action_dp_embeds = prev_action_dp_embeds
                txt_dp_embeds = fused_update_txt_embeds[:, 0, :].unsqueeze(1)
                lv_state = torch.cat([img_txt_embeds, txt_dp_embeds, state], dim=1)
                if self.model_config.imu_encoder.use:
                    lv_state = torch.cat([lv_state, imu_dp_embeds], dim=1)
                lv_state = torch.cat([lv_state, prev_action_dp_embeds], dim=1)

                type_embeds = [0] * img_txt_embeds.shape[1] + [1] * txt_dp_embeds.shape[1] + [2] * state.shape[1]

                if self.model_config.imu_encoder.use:
                    type_embeds += [4] * imu_dp_embeds.shape[1]
                type_embeds += [5] * prev_action_dp_embeds.shape[1]

                type_embeds = torch.from_numpy(np.array(type_embeds)).to(device)
                type_embeds = self.action_type_embeds(type_embeds).repeat(batch_size, 1, 1)

                cond_mask = torch.zeros(batch_size, lv_state.shape[1]).to(device)
                if (
                    train_classifier_free_guidance
                    and self.model_config.diffusion_policy.cls_mask_method == 'mask_token'
                ):
                    mask_prob = torch.rand(batch_size) < self.model_config.diffusion_policy.cls_mask_ratio
                    if self.model_config.diffusion_policy.random_mask_instr:
                        cond_mask[
                            mask_prob,
                            img_txt_embeds.shape[1] : img_txt_embeds.shape[1] + txt_dp_embeds.shape[1],
                        ] = 1
                    if self.model_config.diffusion_policy.random_mask_rgb:
                        cond_mask[mask_prob, : img_txt_embeds.shape[1]] = 1

                t_token_mask = torch.zeros(batch_size, 1).to(device)
                cond_mask = torch.cat([t_token_mask, cond_mask], dim=1)

                if not self.model_config.diffusion_policy.use:
                    cond_mask = None

                noise_pred = self.action_dp_pred_net(
                    sample=noisy_action.float(),
                    timestep=timesteps,
                    cond=lv_state.float(),
                    type_embeds=type_embeds,
                    cond_mask=cond_mask,
                )

        """8. Predict distances"""
        dist_pred = None
        if self.model_config.distance_predictor.use:
            dist_pred = self.distance_pred_net(state.squeeze(1))

        progress_pred = None
        if self.model_config.progress_monitor.use:
            if self.model_config.progress_monitor.concat_state_txt:
                progress_pred = self.progress_monitor(
                    torch.cat(
                        [state.squeeze(1), fused_update_txt_embeds[:, 0, :]],
                        dim=1,
                    )
                )
            else:
                progress_pred = self.progress_monitor(state.squeeze(1))

        stop_progress_pred = None
        if self.model_config.stop_progress_predictor.use:
            if self.model_config.stop_progress_predictor.concat_state_txt:
                stop_progress_pred = self.stop_progress_predictor(
                    torch.cat(
                        [state.squeeze(1), fused_update_txt_embeds[:, 0, :]],
                        dim=1,
                    )
                )
            else:
                stop_progress_pred = self.stop_progress_predictor(state.squeeze(1))

        return (
            noise_pred,
            dist_pred,
            rnn_states_out,
            noise,
            diffusion_output,
            progress_pred,
            denoise_action_list,
            stop_progress_pred,
        )

    def update_rnn_states(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
    ):
        """1. Encoding previous actions and steps"""
        prev_actions_masks = prev_actions.float() * masks.unsqueeze(-1).float()  # [bs, act_length, 3]
        prev_action_embeds = self.prev_action_embedding(prev_actions_masks)
        latest_prev_action_embeds = prev_action_embeds[:, 0]

        """2. Encoding images"""
        rgb_depth_embeds = self.image_encoder(
            observations['stack_rgb'],
            observations['stack_depth'],
            img_mod=self.model_config.image_encoder.rgb.img_mod,
        )

        """3. Update GRU"""
        if self.model_config.image_encoder.rgb.img_mod == 'multi_patches_avg_pooling':
            if self.model_config.state_encoder.rgb_depth_embed_method == 'flat':
                rgb_depth_for_rnn = torch.flatten(rgb_depth_embeds, 1)
            elif self.model_config.state_encoder.rgb_depth_embed_method == 'first':
                rgb_depth_for_rnn = rgb_depth_embeds[:, 0, :]
        else:
            rgb_depth_for_rnn = rgb_depth_embeds.squeeze(1)
        concat_embeds = torch.cat([rgb_depth_for_rnn, latest_prev_action_embeds], dim=1)
        if self.model_config.imu_encoder.use:
            imu_embeds = self.imu_linear(observations['imu'])

            # Concat GRU input features
            concat_embeds = torch.cat([concat_embeds, imu_embeds], dim=1)

        """4. Compute GRU features"""
        state, rnn_states_out = self.state_encoder(concat_embeds, rnn_states, masks.bool())
        return state, rnn_states_out

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

    def parse_action(
        self,
        diffusion_output,
        dist_pred,
        pm_pred=None,
        stop_mode='distance',
        steps=None,
        stop_pm_pred=None,
    ):
        cumsum = False if self.model_config.eval.action == 'discrete' else True
        if self.model_config.learn_angle:
            un_actions = get_action(diffusion_output, self.action_stats, cumsum=cumsum)
            un_actions_nocumsum = get_action(diffusion_output, self.action_stats, cumsum=False)
            actions_cumsum = get_action(diffusion_output, self.action_stats, cumsum=True)
        else:
            un_actions = diffusion_output
            un_actions_nocumsum, actions_cumsum = un_actions, None

        un_actions = un_actions.detach().cpu().numpy()

        if self.model_config.eval.action == 'discrete':
            # 0: stop, 1: move forward, 2: turn left, 3: turn right
            actions = [[] for _ in range(un_actions.shape[0])]
            for bs_idx in range(un_actions.shape[0]):
                if self.model_config.learn_angle:
                    for step_idx in range(un_actions.shape[1]):
                        if stop_mode in ['progress', 'stop_progress']:
                            stop_flag = False
                            if stop_mode == 'stop_progress':
                                stop_flag = stop_pm_pred[bs_idx].item() > self.model_config.eval.stop_progress_threshold
                            else:
                                stop_flag = pm_pred[bs_idx].item() > self.model_config.eval.pm_threshold

                            """Stop if there are N consecutive stops"""
                            M_stops = 3
                            if step_idx + M_stops < len(
                                un_actions_nocumsum[bs_idx]
                            ):  # Make sure we have enough steps ahead
                                consecutive_stops = True
                                for i in range(M_stops):  # Check current and next M steps
                                    curr_action = un_actions_nocumsum[bs_idx][step_idx + i]
                                    if not (
                                        abs(curr_action[0]) < 1e-1
                                        and abs(curr_action[1]) < 1e-1
                                        and abs(curr_action[2]) < 1e-1
                                    ):
                                        consecutive_stops = False
                                        break

                                if consecutive_stops or stop_flag:
                                    stop_flag = True

                            if stop_flag:
                                actions[bs_idx].append(action_spaces['stop'])
                                continue
                            # if bs_idx>0 and len(actions[bs_idx])>1:
                            #     if np.logical_and.reduce(np.array([actions[i][len(actions[bs_idx])-1] for i in range(bs_idx+1)]).flatten() == action_spaces['stop']):
                            #         a=1
                        if abs(un_actions[bs_idx][step_idx][0]) < 1e-1 and abs(un_actions[bs_idx][step_idx][1]) < 1e-1:
                            # turn left or turn right
                            if un_actions[bs_idx][step_idx][2] > 0:
                                actions[bs_idx].append(action_spaces['turn_left'])
                            elif un_actions[bs_idx][step_idx][2] < 0:
                                actions[bs_idx].append(action_spaces['turn_right'])
                        else:
                            # move forward
                            actions[bs_idx].append(action_spaces['go_forward'])

        return actions, actions_cumsum, un_actions_nocumsum

    def save_predicted_actions(self, un_actions, gt_actions=None, N=1, save_dir=None, step=None):
        for item_idx in range(N):
            plt.clf()
            plt.figure(figsize=(8, 8))

            ax = plt.gca()
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            # Plot predicted actions with arrows
            # X is forward, positive Y is left, negative Y is right
            plt.scatter(
                -un_actions[item_idx][:, 1],
                un_actions[item_idx][:, 0],
                label='un_actions',
                color='blue',
                alpha=0.5,
            )
            for i in range(un_actions[item_idx].shape[0]):
                # Calculate arrow direction components using yaw angle
                arrow_length = 0.2  # Adjust this value to change arrow length
                dx = arrow_length * np.cos(np.pi / 2 + un_actions[item_idx][i, 2])
                dy = arrow_length * np.sin(np.pi / 2 + un_actions[item_idx][i, 2])

                # Draw arrow
                plt.arrow(
                    -un_actions[item_idx][i, 1],
                    un_actions[item_idx][i, 0],
                    dx,
                    dy,
                    head_width=0.05,
                    head_length=0.1,
                    fc='blue',
                    ec='blue',
                    alpha=0.5,
                )

                # Add point index
                plt.text(
                    -un_actions[item_idx][i, 1],
                    un_actions[item_idx][i, 0],
                    str(i),
                    fontsize=9,
                    color='blue',
                    ha='left',
                )

            # Plot ground truth actions with arrows
            if gt_actions is not None:
                plt.scatter(
                    -gt_actions[item_idx][:, 1],
                    gt_actions[item_idx][:, 0],
                    label='gt_actions',
                    color='red',
                    alpha=0.5,
                )
                for i in range(gt_actions[item_idx].shape[0]):
                    # Calculate arrow direction components using yaw angle
                    arrow_length = 0.2  # Adjust this value to change arrow length
                    dx = arrow_length * np.cos(np.pi / 2 + gt_actions[item_idx][i, 2])
                    dy = arrow_length * np.sin(np.pi / 2 + gt_actions[item_idx][i, 2])

                    # Draw arrow
                    plt.arrow(
                        -gt_actions[item_idx][i, 1],
                        gt_actions[item_idx][i, 0],
                        dx,
                        dy,
                        head_width=0.05,
                        head_length=0.1,
                        fc='red',
                        ec='red',
                        alpha=0.5,
                    )

                    # Add point index
                    plt.text(
                        -gt_actions[item_idx][i, 1],
                        gt_actions[item_idx][i, 0],
                        str(i),
                        fontsize=9,
                        color='red',
                        ha='right',
                    )

            plt.xlabel('y', x=1.0, ha='center')
            plt.ylabel('x', y=1.0, ha='center')

            max_range = max(
                abs(plt.xlim()[0]),
                abs(plt.xlim()[1]),
                abs(plt.ylim()[0]),
                abs(plt.ylim()[1]),
            )
            plt.xlim(-max_range * 1.2, max_range * 1.2)
            plt.ylim(-max_range * 1.2, max_range * 1.2)

            plt.plot(0, 0, 'ko', markersize=5)  # origin

            plt.legend(loc='upper right')
            plt.grid(True)
            plt.axis('equal')  # Make sure the aspect ratio is equal

            if save_dir is not None:
                save_path = os.path.join(save_dir, f'model_output_step_{step}.jpg')
            else:
                save_path = f'data/images/model_output_{item_idx}_step_{step}.jpg'
            plt.savefig(save_path)
            print(f'save fig to {save_path}')

            plt.close()

    def act(self, batch):
        predicted_actions_save_dir = (
            batch['predicted_actions_save_dir'] if 'predicted_actions_save_dir' in batch else None
        )
        vis = batch['vis']
        step = batch['step']

        (
            noise_pred,
            dist_pred,
            rnn_states_out,
            noise,
            diffusion_output,
            progress_pred,
            denoise_action_list,
            stop_progress_pred,
        ) = self.pred_actions(
            batch['observations'],
            batch['rnn_states'],
            batch['prev_actions'],
            batch['masks'],
            batch['add_noise_to_action'],
            batch['denoise_action'],
            batch['num_sample'],
            sample_classifier_free_guidance=batch['sample_classifier_free_guidance'],
        )

        if vis:
            un_actions = get_action(diffusion_output, self.action_stats).cpu().detach().numpy()
            self.save_predicted_actions(
                un_actions,
                gt_actions=None,
                N=1,
                save_dir=predicted_actions_save_dir,
                step=step,
            )

        actions, actions_cumsum, un_actions_nocumsum = self.parse_action(
            diffusion_output,
            dist_pred,
            pm_pred=progress_pred,
            stop_mode=batch['stop_mode'],
            steps=step,
            stop_pm_pred=stop_progress_pred,
        )

        return (
            actions,
            rnn_states_out,
            noise_pred,
            dist_pred,
            noise,
            diffusion_output,
            un_actions_nocumsum,
            progress_pred,
            stop_progress_pred,
        )

    def forward(self, batch) -> Tuple[Tensor, Tensor]:
        mode = batch['mode']

        if mode == 'img_embedding':
            if 'depth_return_x_before_fc' not in batch:
                batch['depth_return_x_before_fc'] = False
            if 'need_img_extraction' not in batch:
                batch['need_img_extraction'] = True
            return self.img_embedding(
                batch['rgb_inputs'],
                batch['depth_inputs'],
                batch['img_mod'],
                batch['depth_return_x_before_fc'],
                batch['proj'],
                batch['process_images'],
                batch['need_img_extraction'],
            )

        elif mode == 'txt_embedding':
            text_embeds, txt_masks, text_cls_embeds = self.instruction_encoder(batch['instr_inputs'])
            return text_embeds

        elif mode == 'update_rnn':
            return self.update_rnn_states(
                batch['observations'],
                batch['rnn_states'],
                batch['prev_actions'],
                batch['masks'],
            )

        elif mode == 'pred_actions':
            device = batch['observations']['instruction'].device
            batch_size = batch['observations']['instruction'].shape[0]
            if 'num_sample' not in batch:
                batch['num_sample'] = 1
            if 'need_img_extraction' in batch and batch['need_img_extraction']:
                if 'rgb' in batch['observations'].keys():
                    input_rgb = batch['observations']['rgb']
                else:
                    input_rgb = batch['observations']['rgb_features']
                input_depth = batch['observations']['depth']

                if 'rgb_features' in batch['observations'].keys():
                    need_rgb_extraction = False
                else:
                    need_rgb_extraction = True

                if batch['train_cls_free_guidance'] and self.model_config.diffusion_policy.random_mask_rgb:
                    cls_free_mask = torch.rand(batch_size) < self.model_config.diffusion_policy.cls_mask_ratio
                    cls_free_mask = cls_free_mask.to(device)
                    input_rgb[cls_free_mask] = torch.zeros_like(input_rgb[cls_free_mask])
                    input_depth[cls_free_mask] = torch.zeros_like(input_depth[cls_free_mask])

                stack_rgb, stack_depth = self.img_embedding(
                    input_rgb,
                    input_depth,
                    batch['img_mod'],
                    batch['depth_return_x_before_fc'],
                    batch['proj'],
                    batch['process_images'],
                    need_rgb_extraction,
                )
                if len(stack_rgb.shape) == 2:
                    batch['observations']['stack_rgb'] = stack_rgb.unsqueeze(1)
                    batch['observations']['stack_depth'] = stack_depth.unsqueeze(1)
                else:
                    batch['observations']['stack_rgb'] = stack_rgb
                    batch['observations']['stack_depth'] = stack_depth
            else:
                if (
                    batch['train_cls_free_guidance']
                    and self.model_config.diffusion_policy.random_mask_rgb
                    and self.model_config.diffusion_policy.cls_mask_method == 'mask_inputs'
                ):
                    cls_free_mask = torch.rand(batch_size) < self.model_config.diffusion_policy.cls_mask_ratio
                    cls_free_mask = cls_free_mask.to(device)
                    batch['observations']['stack_rgb'][cls_free_mask] = torch.zeros_like(
                        batch['observations']['stack_rgb'][cls_free_mask]
                    )
                    batch['observations']['stack_depth'][cls_free_mask] = torch.zeros_like(
                        batch['observations']['stack_depth'][cls_free_mask]
                    )

            return self.pred_actions(
                batch['observations'],
                batch['rnn_states'],
                batch['prev_actions'],
                batch['masks'],
                batch['add_noise_to_action'],
                batch['denoise_action'],
                batch['num_sample'],
                batch['train_cls_free_guidance'],
                batch['sample_cls_free_guidance'],
                batch['need_txt_extraction'],
            )

        elif mode == 'act':
            return self.act(batch)
