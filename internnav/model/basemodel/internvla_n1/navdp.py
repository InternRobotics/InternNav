import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

from internnav.model.encoder.navdp_backbone import *  # noqa: F403

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class NavDP_Policy_DPT_CriticSum_DAT(nn.Module):
    def __init__(
        self,
        image_size=224,
        memory_size=2,
        predict_size=32,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        vlm_token_dim=3584,
        channels=3,
        dropout=0.1,
        scratch=False,
        finetune=False,
        use_critic=False,
        input_dtype="bf16",
        navdp_pretrained=None,
        navdp_version=0.0,
        device='cuda:0',
    ):
        super().__init__()
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.dropout = dropout

        self.use_critic = use_critic
        self.token_dim = token_dim
        self.vlm_token_dim = vlm_token_dim
        if input_dtype == "bf16":
            self.input_dtype = torch.bfloat16
        else:
            self.input_dtype = torch.float32

        self.rgbd_encoder = DAT_RGBD_Patch_Backbone(  # noqa: F405
            image_size, token_dim, memory_size=memory_size, finetune=finetune, version=navdp_version
        )
        self.point_encoder = nn.Linear(3, self.token_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=self.temporal_depth)

        self.input_embed = nn.Linear(3, token_dim)
        self.cond_pos_embed = nn.Parameter(torch.zeros((1, memory_size * 16 + 2, token_dim), dtype=self.input_dtype))
        self.out_pos_embed = nn.Parameter(torch.zeros((1, predict_size, token_dim), dtype=self.input_dtype))
        self.drop = nn.Dropout(dropout)
        self.time_emb = SinusoidalPosEmb(token_dim)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=20, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon'
        )

        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)

        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = (
            self.tgt_mask.float()
            .masked_fill(self.tgt_mask == 0, float('-inf'))
            .masked_fill(self.tgt_mask == 1, float(0.0))
        )
        self.tgt_mask = self.tgt_mask.to(dtype=self.input_dtype)

        self.cond_critic_mask = torch.zeros((predict_size, 2 + memory_size * 16))
        self.cond_critic_mask[:, 0:2] = float('-inf')
        self.cond_critic_mask = self.cond_critic_mask.to(dtype=self.input_dtype)

        self.vlm_embed_mlp = nn.Sequential(
            nn.Linear(vlm_token_dim, vlm_token_dim // 4),
            nn.ReLU(),
            nn.Linear(vlm_token_dim // 4, vlm_token_dim // 8),
            nn.ReLU(),
            nn.Linear(vlm_token_dim // 8, token_dim),
        )

        self.goal_compressor = TokenCompressor(token_dim, 8, 1)  # noqa: F405

        self.pg_embed_mlp = nn.Sequential(nn.Linear(2, token_dim // 2), nn.ReLU(), nn.Linear(token_dim // 2, token_dim))
        self.pg_pred_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Linear(token_dim // 2, token_dim // 4),
            nn.ReLU(),
            nn.Linear(token_dim // 4, 2),
        )

        self.model_name = "NavDP_Policy_DPT_CriticSum_DAT"
        self.navdp_pretrained = navdp_pretrained

    def load_model(self):
        rank0_print(f"Loading navdp model: {self.model_name}")
        rank0_print(f"Pretrained: {self.navdp_pretrained}")

        if self.navdp_pretrained is None:
            rank0_print("No pretrained weights provided, initializing randomly.")
            return

        try:
            pretrained_dict = torch.load(self.navdp_pretrained)

            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']

            model_dict = self.state_dict()

            matched_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()
            }

            unmatched_pretrained = [k for k in pretrained_dict if k not in matched_dict]
            unmatched_model = [k for k in model_dict if k not in pretrained_dict]

            model_dict.update(matched_dict)
            self.load_state_dict(model_dict)

            rank0_print(f"Successfully loaded pretrained weights from {self.navdp_pretrained}")
            rank0_print(f"Loaded {len(matched_dict)}/{len(model_dict)} layers")

            if unmatched_pretrained:
                rank0_print("\nParameters in pretrained but NOT loaded:")
                for k in unmatched_pretrained:
                    if k in model_dict:
                        reason = (
                            f"size mismatch (pretrained: {pretrained_dict[k].size()}, model: {model_dict[k].size()})"
                        )
                    else:
                        reason = "not in model"
                    rank0_print(f"  - {k} ({reason})")

            if unmatched_model:
                rank0_print("\nParameters in model but NOT in pretrained:")
                for k in unmatched_model:
                    rank0_print(f"  - {k}")

        except Exception as e:
            rank0_print(f"Error loading pretrained weights: {str(e)}")
            rank0_print("Continuing with random initialization.")

    def sample_noise(self, action):
        noise = torch.randn(action.shape, dtype=action.dtype).to(action.device)
        timesteps = (
            torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (action.shape[0],))
            .long()
            .to(action.device)
        )
        time_embeds = self.time_emb(timesteps).unsqueeze(1).to(dtype=self.input_dtype)
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        noisy_action_embed = self.input_embed(noisy_action)
        return noise, time_embeds, noisy_action_embed

    def predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed=None):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(last_actions.device)).unsqueeze(1).to(dtype=last_actions.dtype)

        if rgbd_embed is not None:
            cond_embedding = (
                torch.cat([time_embeds, goal_embed, rgbd_embed], dim=1)
                + self.cond_pos_embed[:, : self.memory_size * 16 + 2, :]
            )
        else:
            cond_embedding = torch.cat([time_embeds, goal_embed], dim=1) + self.cond_pos_embed[:, :2, :]

        cond_embedding = cond_embedding.repeat(action_embeds.shape[0], 1, 1)
        input_embedding = action_embeds + self.out_pos_embed[:, : self.predict_size, :]

        output = self.decoder(tgt=input_embedding, memory=cond_embedding, tgt_mask=self.tgt_mask)
        output = self.layernorm(output)
        output = self.action_head(output)
        return output

    def predict_pointgoal_action_async(
        self, vlm_tokens, input_images=None, input_depths=None, vlm_mask=None, sample_num=32
    ):
        """
        Predict action sequence for point goal navigation using diffusion-based approach.

        This method generates a sequence of actions to reach a target point using
        vision-language model (VLM) embeddings and RGB-D sensory inputs, leveraging
        a diffusion model to denoise action predictions.

        Args:
            vlm_tokens (Tensor): Token embeddings from vision-language model,
                shape (batch_size, token_numbers, 3584)
            input_images (Tensor, optional): Input RGB images including memory frames,
                shape (batch_size, memory_frames, 224, 224, 3).
                Defaults to None.
            input_depths (Tensor, optional): Input depth maps,
                shape (batch_size, memory_frames, 224, 224, 1).
                Defaults to None.
            vlm_mask (Tensor, optional): Mask for VLM tokens indicating valid positions,
                shape (batch_size, token_numbers).
                Defaults to None.
            sample_num (int, optional): Number of action sequences to sample through diffusion.
                Defaults to 32.

        Returns:
            Tensor: Predicted action trajectories after diffusion denoising,
                shape (sample_num * batch_size, prediction_size, 3)
        """
        with torch.no_grad():
            bs = vlm_tokens.shape[0]
            if bs != 1:
                vlm_tokens = vlm_tokens[0:1]
                vlm_mask = vlm_mask[0:1]
                bs = 1

            if vlm_mask is not None:
                vlm_mask_ = vlm_mask.bool()
                vlm_mask = ~vlm_mask_

            vlm_tokens = self.vlm_embed_mlp(vlm_tokens)
            vlm_embed = self.goal_compressor(vlm_tokens, vlm_mask)

            rgbd_embed = self.rgbd_encoder(input_images, input_depths)

            noisy_action = torch.randn((sample_num * bs, self.predict_size, 3), dtype=vlm_embed.dtype).to(
                vlm_embed.device
            )
            naction = noisy_action

            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), vlm_embed, rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            current_trajectory = naction
            return current_trajectory

    def predict_pointgoal_action(self, vlm_tokens, input_images=None, input_depths=None, vlm_mask=None, sample_num=32):
        """
        Args:
            vlm_tokens: bs*sel_num, token_nums, 3584
            input_images: bs*sel_num, memory+1, 224, 224, 3
            input_depths: bs*sel_num, 1, 224, 224, 3
            vlm_mask: bs*sel_num, token_nums
        """
        with torch.no_grad():
            bs = vlm_tokens.shape[0]
            if bs != 1:
                vlm_tokens = vlm_tokens[0:1]
                vlm_mask = vlm_mask[0:1]
                bs = 1

            if vlm_mask is not None:
                vlm_mask_ = vlm_mask.bool()
                # mask==True parts will be ignored by transformer, but now vlm valid parts have mask==True!
                vlm_mask = ~vlm_mask_

            vlm_tokens = self.vlm_embed_mlp(vlm_tokens)
            vlm_embed = torch.mean(vlm_tokens, dim=1).unsqueeze(1)

            noisy_action = torch.randn((sample_num * bs, self.predict_size, 3), dtype=vlm_embed.dtype).to(
                vlm_embed.device
            )
            naction = noisy_action

            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), vlm_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            current_trajectory = naction
            return current_trajectory

    def forward_vlm_traj(
        self, vlm_tokens, input_images, input_depths, tensor_label_actions, tensor_augment_actions=None
    ):
        vlm_tokens = self.vlm_embed_mlp(vlm_tokens)
        vlm_embed = self.goal_compressor(vlm_tokens)  # （bs,1,384)

        tensor_label_actions = tensor_label_actions.flatten(0, 1)

        # sample noise and actions
        pg_noise, pg_time_embed, pg_noisy_action_embed = self.sample_noise(tensor_label_actions)

        rgbd_embed = self.rgbd_encoder(input_images, input_depths)  # [64, 32, 384]

        cond_embed = torch.cat([pg_time_embed, vlm_embed, rgbd_embed], dim=1)
        pg_cond_embeddings = self.drop(cond_embed + self.cond_pos_embed[:, : cond_embed.size(1)])

        pg_action_embeddings = self.drop(pg_noisy_action_embed + self.out_pos_embed[:, : self.predict_size, :])

        pg_output = self.decoder(tgt=pg_action_embeddings, memory=pg_cond_embeddings, tgt_mask=self.tgt_mask)
        pg_output = self.layernorm(pg_output)
        noise_pred_pg = self.action_head(pg_output)
        return noise_pred_pg, pg_noise
