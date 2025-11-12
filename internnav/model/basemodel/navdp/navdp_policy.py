import os

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import PretrainedConfig, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.model.encoder.navdp_backbone import (
    ImageGoalBackbone,
    LearnablePositionalEncoding,
    PixelGoalBackbone,
    RGBDBackbone,
    SinusoidalPosEmb,
)


class NavDPModelConfig(PretrainedConfig):
    model_type = 'navdp'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # pass in navdp_exp_cfg
        self.model_cfg = kwargs.get('model_cfg', None)

    @classmethod
    def from_dict(cls, config_dict):
        if 'model_cfg' in config_dict:
            config_dict['model_cfg'] = ExpCfg(**config_dict['model_cfg'])
        return super().from_dict(config_dict)


class NavDPNet(PreTrainedModel):
    config_class = NavDPModelConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)  # navdp_exp_cfg_dict_NavDPModelConfig
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if config is a pydantic model, convert to NavDPModelConfig
        if hasattr(config, 'model_dump'):
            config = cls.config_class(model_cfg=config)

        model = cls(config)
        model.to(model._device)

        # load pretrained weights
        if os.path.isdir(pretrained_model_name_or_path):
            incompatible_keys, _ = model.load_state_dict(
                torch.load(os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin'))
            )
            if len(incompatible_keys) > 0:
                print(f'Incompatible keys: {incompatible_keys}')
        elif pretrained_model_name_or_path is None or len(pretrained_model_name_or_path) == 0:
            pass
        else:
            incompatible_keys, _ = model.load_state_dict(torch.load(pretrained_model_name_or_path), strict=False)
            if len(incompatible_keys) > 0:
                print(f'Incompatible keys: {incompatible_keys}')

        return model

    def __init__(self, config: NavDPModelConfig):
        super().__init__(config)

        if isinstance(config, NavDPModelConfig):
            self.model_config = ModelCfg(**config.model_cfg['model'])
        else:
            self.model_config = config
        self.config.model_cfg['il']
        self._device = torch.device(f"cuda:{config.model_cfg['local_rank']}")
        self.image_size = self.config.model_cfg['il']['image_size']
        self.memory_size = self.config.model_cfg['il']['memory_size']
        self.predict_size = self.config.model_cfg['il']['predict_size']
        self.pixel_channel = self.config.model_cfg['il']['pixel_channel']
        self.temporal_depth = self.config.model_cfg['il']['temporal_depth']
        self.attention_heads = self.config.model_cfg['il']['heads']
        self.input_channels = self.config.model_cfg['il']['channels']
        self.dropout = self.config.model_cfg['il']['dropout']
        self.token_dim = self.config.model_cfg['il']['token_dim']
        self.scratch = self.config.model_cfg['il']['scratch']
        self.finetune = self.config.model_cfg['il']['finetune']
        self.rgbd_encoder = RGBDBackbone(
            self.image_size, self.token_dim, memory_size=self.memory_size, finetune=self.finetune, device=self._device
        )
        self.pixel_encoder = PixelGoalBackbone(
            self.image_size, self.token_dim, pixel_channel=self.pixel_channel, device=self._device
        )
        self.image_encoder = ImageGoalBackbone(self.image_size, self.token_dim, device=self._device)
        self.point_encoder = nn.Linear(3, self.token_dim)

        if not self.finetune:
            for p in self.rgbd_encoder.rgb_model.parameters():
                p.requires_grad = False
            self.rgbd_encoder.rgb_model.eval()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.token_dim,
            nhead=self.attention_heads,
            dim_feedforward=4 * self.token_dim,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.temporal_depth)
        self.input_embed = nn.Linear(3, self.token_dim)

        self.cond_pos_embed = LearnablePositionalEncoding(self.token_dim, self.memory_size * 16 + 4)
        self.out_pos_embed = LearnablePositionalEncoding(self.token_dim, self.predict_size)
        self.drop = nn.Dropout(self.dropout)
        self.time_emb = SinusoidalPosEmb(self.token_dim)
        self.layernorm = nn.LayerNorm(self.token_dim)
        self.action_head = nn.Linear(self.token_dim, 3)
        self.critic_head = nn.Linear(self.token_dim, 1)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=10, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon'
        )
        self.tgt_mask = (torch.triu(torch.ones(self.predict_size, self.predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = (
            self.tgt_mask.float()
            .masked_fill(self.tgt_mask == 0, float('-inf'))
            .masked_fill(self.tgt_mask == 1, float(0.0))
        )
        self.tgt_mask = self.tgt_mask.to(self._device)

        self.cond_critic_mask = torch.zeros((self.predict_size, 4 + self.memory_size * 16))
        self.cond_critic_mask[:, 0:4] = float('-inf')

        self.pixel_aux_head = nn.Linear(self.token_dim, 3)
        self.image_aux_head = nn.Linear(self.token_dim, 3)

    def to(self, device, *args, **kwargs):
        # first call the to method of the parent class
        self = super().to(device, *args, **kwargs)

        # ensure the buffer is on the correct device
        self.cond_critic_mask = self.cond_critic_mask.to(device)

        # update device attribute
        self._device = device

        return self

    def sample_noise(self, action):
        device = action.device
        noise = torch.randn(action.shape, device=device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (action.shape[0],), device=device
        ).long()
        time_embeds = self.time_emb(timesteps).unsqueeze(1)
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        noisy_action_embed = self.input_embed(noisy_action)
        return noise, time_embeds, noisy_action_embed

    def predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self._device)).unsqueeze(1)
        cond_embedding = torch.cat(
            [time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1
        ) + self.cond_pos_embed(torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1))
        cond_embedding = cond_embedding.repeat(action_embeds.shape[0], 1, 1)
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt=input_embedding, memory=cond_embedding, tgt_mask=self.tgt_mask.to(self._device))
        output = self.layernorm(output)
        output = self.action_head(output)
        return output

    def predict_critic(self, predict_trajectory, rgbd_embed):
        repeat_rgbd_embed = rgbd_embed.repeat(predict_trajectory.shape[0], 1, 1)
        nogoal_embed = torch.zeros_like(repeat_rgbd_embed[:, 0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat(
            [nogoal_embed, nogoal_embed, nogoal_embed, nogoal_embed, repeat_rgbd_embed], dim=1
        ) + self.cond_pos_embed(
            torch.cat([nogoal_embed, nogoal_embed, nogoal_embed, nogoal_embed, repeat_rgbd_embed], dim=1)
        )
        critic_output = self.decoder(tgt=action_embeddings, memory=cond_embeddings, memory_mask=self.cond_critic_mask)
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:, 0]
        return critic_output

    def forward(self, goal_point, goal_image, goal_pixel, input_images, input_depths, output_actions, augment_actions):
        device = next(self.parameters()).device

        assert input_images.shape[1] == self.memory_size
        tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32).to(device)
        tensor_label_actions = torch.as_tensor(output_actions, dtype=torch.float32).to(device)
        tensor_augment_actions = torch.as_tensor(augment_actions, dtype=torch.float32).to(device)
        input_images = input_images.to(device)
        input_depths = input_depths.to(device)

        ng_noise, ng_time_embed, ng_noisy_action_embed = self.sample_noise(tensor_label_actions)
        mg_noise, mg_time_embed, mg_noisy_action_embed = self.sample_noise(tensor_label_actions)

        rgbd_embed = self.rgbd_encoder(input_images, input_depths)
        pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)
        nogoal_embed = torch.zeros_like(pointgoal_embed)
        imagegoal_embed = self.image_encoder(goal_image).unsqueeze(1)
        pixelgoal_embed = self.pixel_encoder(goal_pixel).unsqueeze(1)

        imagegoal_aux_pred = self.image_aux_head(imagegoal_embed[:, 0])
        pixelgoal_aux_pred = self.pixel_aux_head(pixelgoal_embed[:, 0])

        label_embed = self.input_embed(tensor_label_actions).detach()
        augment_embed = self.input_embed(tensor_augment_actions).detach()

        cond_pos_embed = self.cond_pos_embed(
            torch.cat([ng_time_embed, nogoal_embed, imagegoal_embed, pixelgoal_embed, rgbd_embed], dim=1)
        )
        ng_cond_embeddings = self.drop(
            torch.cat([ng_time_embed, nogoal_embed, nogoal_embed, nogoal_embed, rgbd_embed], dim=1) + cond_pos_embed
        )

        cand_goal_embed = [pointgoal_embed, imagegoal_embed, pixelgoal_embed]
        batch_size = pointgoal_embed.shape[0]

        # Generate deterministic selections for each sample in the batch using vectorized operations
        batch_indices = torch.arange(batch_size, device=pointgoal_embed.device)
        pattern_indices = batch_indices % 27  # 3^3 = 27 possible combinations
        selections_0 = pattern_indices % 3
        selections_1 = (pattern_indices // 3) % 3
        selections_2 = (pattern_indices // 9) % 3
        goal_embeds = torch.stack(cand_goal_embed, dim=0)  # [3, batch_size, 1, token_dim]
        selected_goals_0 = goal_embeds[selections_0, torch.arange(batch_size), :, :]  # [batch_size, 1, token_dim]
        selected_goals_1 = goal_embeds[selections_1, torch.arange(batch_size), :, :]
        selected_goals_2 = goal_embeds[selections_2, torch.arange(batch_size), :, :]
        mg_cond_embed_tensor = torch.cat(
            [mg_time_embed, selected_goals_0, selected_goals_1, selected_goals_2, rgbd_embed], dim=1
        )
        mg_cond_embeddings = self.drop(mg_cond_embed_tensor + cond_pos_embed)

        out_pos_embed = self.out_pos_embed(ng_noisy_action_embed)
        ng_action_embeddings = self.drop(ng_noisy_action_embed + out_pos_embed)
        mg_action_embeddings = self.drop(mg_noisy_action_embed + out_pos_embed)
        label_action_embeddings = self.drop(label_embed + out_pos_embed)
        augment_action_embeddings = self.drop(augment_embed + out_pos_embed)

        ng_output = self.decoder(tgt=ng_action_embeddings, memory=ng_cond_embeddings, tgt_mask=self.tgt_mask)
        ng_output = self.layernorm(ng_output)
        noise_pred_ng = self.action_head(ng_output)

        mg_output = self.decoder(
            tgt=mg_action_embeddings, memory=mg_cond_embeddings, tgt_mask=self.tgt_mask.to(ng_action_embeddings.device)
        )
        mg_output = self.layernorm(mg_output)
        noise_pred_mg = self.action_head(mg_output)

        cr_label_output = self.decoder(
            tgt=label_action_embeddings, memory=ng_cond_embeddings, memory_mask=self.cond_critic_mask.to(self._device)
        )
        cr_label_output = self.layernorm(cr_label_output)
        cr_label_pred = self.critic_head(cr_label_output.mean(dim=1))[:, 0]

        cr_augment_output = self.decoder(
            tgt=augment_action_embeddings, memory=ng_cond_embeddings, memory_mask=self.cond_critic_mask.to(self._device)
        )
        cr_augment_output = self.layernorm(cr_augment_output)
        cr_augment_pred = self.critic_head(cr_augment_output.mean(dim=1))[:, 0]
        return (
            noise_pred_ng,
            noise_pred_mg,
            cr_label_pred,
            cr_augment_pred,
            [ng_noise, mg_noise],
            [imagegoal_aux_pred, pixelgoal_aux_pred],
        )

    def _get_device(self):
        """Safe get device information"""
        # try to get device through model parameters
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass

        # try to get device through buffer
        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass

        # try to get device through submodule
        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue

        # finally revert to default device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_pointgoal_batch_action_vel(self, goal_point, input_images, input_depths, sample_num=32):
        with torch.no_grad():
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self._device)
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)

            noisy_action = torch.randn(
                (sample_num * pointgoal_embed.shape[0], self.predict_size, 3), device=self._device
            )
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.to(self._device).unsqueeze(0), pointgoal_embed, rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            critic_values = self.predict_critic(naction, rgbd_embed)

            negative_trajectory = torch.cumsum(naction / 4.0, dim=1)[(critic_values).argsort()[0:8]]
            positive_trajectory = torch.cumsum(naction / 4.0, dim=1)[(-critic_values).argsort()[0:8]]
            return negative_trajectory, positive_trajectory

    def predict_nogoal_batch_action_vel(self, input_images, input_depths, sample_num=32):
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            nogoal_embed = torch.zeros_like(rgbd_embed[:, 0:1])

            noisy_action = torch.randn((sample_num * nogoal_embed.shape[0], self.predict_size, 3), device=self._device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), nogoal_embed, rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            critic_values = self.predict_critic(naction, rgbd_embed)

            negative_trajectory = torch.cumsum(naction / 4.0, dim=1)[(critic_values).argsort()[0:8]]
            positive_trajectory = torch.cumsum(naction / 4.0, dim=1)[(-critic_values).argsort()[0:8]]
            return negative_trajectory, positive_trajectory
