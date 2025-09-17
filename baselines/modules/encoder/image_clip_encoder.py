import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from internnav.configs.model.base_encoders import ImageEncoder as ImageEncoderCfg

from ..basemodel.LongCLIP.model import longclip
from . import resnet_encoders
from .bert_backbone import PositionalEncoding
import torch.nn.functional as F

class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        full_config,
        config: ImageEncoderCfg,
        observation_space,
        test=False,
        analysis_time=False,
    ):
        super().__init__()

        self.config: ImageEncoderCfg = config
        self.analysis_time = analysis_time

        # RGB image model
        self.is_clip_long = False
        if config.rgb.model_name == 'clip-long':
            self.image_transformer, self.image_processor = longclip.load(config.rgb.model_path)
            # del text part
            del self.image_transformer.token_embedding
            del self.image_transformer.transformer
            del self.image_transformer.positional_embedding
            del self.image_transformer.ln_final
            self.is_clip_long = True
            self.to_pil = ToPILImage()

        else:
            self.image_transformer_config = CLIPVisionConfig.from_pretrained(config.rgb.model_path)
            self.image_transformer = CLIPVisionModel(self.image_transformer_config)

            self.image_processor = CLIPImageProcessor.from_pretrained(config.rgb.model_path)
        self.image_feature_dim = config.rgb.feature_dim
        self.image_projection_dim = config.rgb.projection_dim
        self.image_fc = torch.nn.Linear(self.image_feature_dim, self.image_projection_dim, bias=False)

        # Depth model
        if config.depth.bottleneck == 'resnet':
            self.depth_encoder = getattr(resnet_encoders, config.depth.cnn_type)(
                observation_space,
                output_size=config.depth.output_size,
                checkpoint=config.depth.ddppo_checkpoint,
                backbone=config.depth.backbone,
                trainable=config.depth.update_depth_encoder,
                spatial_output=True,
                analysis_time=analysis_time,
            )
            self.depth_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.depth_encoder.output_shape),
                    config.depth.feature_dim,
                ),
                nn.ReLU(True),
            )

        # position embedding
        if config.rgb.img_mod == 'multi_patches_avg_pooling':
            self.pos_embedding = PositionalEncoding(
                config.rgb.projection_dim * config.rgb.multi_patches_num,
                max_seq_len=config.img_stack_nums,
            )
        else:
            self.pos_embedding = PositionalEncoding(config.rgb.projection_dim, max_seq_len=config.img_stack_nums)

        self.layernorm = nn.LayerNorm(config.rgb.projection_dim)

        # image & depth linear
        self.img_learnable_linear = nn.Linear(config.rgb.feature_dim, config.rgb.projection_dim)
        self.img_ln = nn.LayerNorm(config.rgb.projection_dim)
        self.depth_learnable_linear = nn.Linear(config.depth.feature_dim, config.depth.projection_dim)
        self.depth_ln = nn.LayerNorm(config.depth.projection_dim)

        # Dropout layers
        self.env_drop = nn.Dropout(config.env_drop)
        self.dropout = nn.Dropout(config.dropout)

        # Set trainable
        for param in self.image_transformer.parameters():
            param.requires_grad_(config.rgb.update_rgb_encoder)

        # Init params
        self.init_param()

    def process_image(self, image_inputs):
        if len(image_inputs.shape) == 5:
            # bs, stack_num, 224, 224, 3
            image_size = image_inputs.shape[2]
            # image_inputs = image_inputs.reshape(-1, 3, image_size, image_size)
            image_inputs = image_inputs.reshape(
                -1,
                image_inputs.shape[-3],
                image_inputs.shape[-2],
                image_inputs.shape[-1],
            )
        if self.is_clip_long:
            if len(image_inputs.shape) == 3 and image_inputs.shape[-1] == 3:
                # convert H,W,C to C,H,W
                image_inputs = image_inputs.permute(2, 0, 1)
                image_Image = self.to_pil(image_inputs)
                image_feat = np.array(self.image_processor(image_Image))
                image_feat = torch.from_numpy(np.array(image_feat)).to(image_inputs.device)

            elif len(image_inputs.shape) == 4 and image_inputs.shape[-1] == 3:
                # convert B,H,W,C to B,C,H,W
                image_inputs = image_inputs.permute(0, 3, 1, 2)
                image_feat = []
                for image in image_inputs:
                    image = self.to_pil(image)
                    image_feat.append(np.array(self.image_processor(image)))
                image_feat = np.array(image_feat)
                image_feat = torch.from_numpy(image_feat).to(image_inputs.device)

        else:
            image_feat = self.image_processor(
                image_inputs,
                do_resize=False,
                do_center_crop=False,
                return_tensors='pt',
            ).pixel_values

        if len(image_inputs.shape) == 5:
            image_feat = image_feat.reshape(image_inputs.shape[0], image_inputs.shape[1], -1)
        return image_feat

    def _normalize(self, depth_batch):
        """Simplified process function"""
        if isinstance(depth_batch, np.ndarray):
            # Handle NumPy array
            depth_mean = self.depth_mean_numpy
            depth_std = self.depth_std_numpy

            if depth_batch.shape[-1] == 1:
                depth_batch = np.repeat(depth_batch, repeats=3, axis=-1)  # Expand to last dimension with 3 copies
            depth_batch = (depth_batch - depth_mean) / depth_std

            if len(depth_batch.shape) == 4:
                depth_batch = np.transpose(depth_batch, (0, 3, 1, 2))  # Permute dimensions for NumPy
            elif len(depth_batch.shape) == 3:
                depth_batch = np.transpose(depth_batch, (2, 0, 1))  # Permute dimensions for NumPy

        elif isinstance(depth_batch, torch.Tensor):
            # Handle PyTorch tensor
            device = depth_batch.device
            depth_mean = self.depth_mean.to(device)
            depth_std = self.depth_std.to(device)

            if len(depth_batch.shape) == 4:
                if depth_batch.shape[-1] == 1:
                    depth_batch = depth_batch.expand(-1, -1, -1, 3)  # Expand to last dimension with 3
                depth_batch = (depth_batch - depth_mean) / depth_std
                depth_batch = depth_batch.permute(0, 3, 1, 2)  # Permute dimensions for PyTorch
            elif len(depth_batch.shape) == 3:
                if depth_batch.shape[-1] == 1:
                    depth_batch = depth_batch.expand(-1, -1, 3)  # Expand to last dimension with 3
                depth_batch = (depth_batch - depth_mean) / depth_std
                depth_batch = depth_batch.permute(2, 0, 1)  # Permute dimensions for PyTorch

        assert depth_batch.shape[-1] == 224
        # depth_batch = self.resize_trans(depth_batch) # Here is a bug that needs the Image class to handle..

        return depth_batch

    def process_depth(self, depth_inputs):
        if len(depth_inputs.shape) == 2:
            depth_inputs = np.expand_dims(depth_inputs, axis=2).repeat(3, axis=2)
        # depth_feat = self.depth_processor(depth_inputs, return_tensors="pt").pixel_values
        depth_feat = self._normalize(depth_inputs)
        return depth_feat

    def encode_image(self, batch_subset: torch.Tensor, proj=True):
        visual = self.image_transformer.visual
        try:
            data_type = visual.conv1.weight.dtype
        except Exception:
            data_type = self.image_transformer.transformer.resblocks[0].mlp.c_fc.weight.dtype
        x = batch_subset.type(data_type)

        x = visual.conv1(x)  # shape = [*, width, grid, grid] # [bs, 3, 224, 224]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD # [bs, 197, 768] (w//16, H//16)

        x = visual.ln_post(x[:, 0, :])

        if proj and visual.proj is not None:
            x = x @ visual.proj

        return x

    def encode_image_multi_patches(self, batch_subset):
        visual = self.image_transformer.visual

        try:
            data_type = visual.conv1.weight.dtype
        except Exception:
            data_type = self.image_transformer.transformer.resblocks[0].mlp.c_fc.weight.dtype
        x = batch_subset.type(data_type)

        """Combine multiple patches into 4 patches and return the full and the multiple patches features"""
        x = visual.conv1(x)  # shape = [*, width, grid, grid] # [bs, 3, 224, 224]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD # [bs, 197, 768] (W//16, H//16)

        grid_num = int(np.sqrt(x.shape[1]))
        patch_features = x[:, 1:, :].reshape(x.shape[0], grid_num, grid_num, -1)  # [bs, 196, 768] -> [bs, 14, 14, 768]
        grid_num_down = grid_num // 2
        patch_features_permute = patch_features.permute(0, 3, 1, 2)
        avg_pool_features = F.avg_pool2d(
            patch_features_permute,
            kernel_size=(grid_num_down, grid_num_down),
            stride=(grid_num_down, grid_num_down),
        )
        avg_pool_features = avg_pool_features.reshape(x.shape[0], -1, 4).permute(0, 2, 1)

        outputs = torch.cat([x[:, 0, :].unsqueeze(1), avg_pool_features], dim=1)

        return outputs

    def embed_image(
        self,
        image_batch,
        fc=False,
        max_batch_size=400,
        img_mod='cls',
        proj=True,
    ):
        """Embed a batch of image."""
        if len(image_batch.shape) == 3:
            image_batch = image_batch.unsqueeze(0)

        BS = image_batch.shape[0]
        reshape_flag = False
        if len(image_batch.shape) == 5:
            stack_num = image_batch.shape[1]
            image_batch = image_batch.reshape(
                -1, 3, image_batch.shape[3], image_batch.shape[4]
            )  # [BS, T, 224, 224, 3] -> [BS*T, 3, 224, 224]
            reshape_flag = True

        embeddings = []
        # Process in chunks if the batch size exceeds the limit
        for i in range(0, image_batch.shape[0], max_batch_size):
            batch_subset = image_batch[i : i + max_batch_size]
            if self.is_clip_long:
                if img_mod == 'cls':
                    # return [bs, 768]
                    outputs = self.encode_image(batch_subset, proj=proj)
                elif img_mod == 'multi_patches_avg_pooling':
                    # return [bs, 5, 768]. 0 is CLS token, and the last 4 are average pooling tokens.
                    outputs = self.encode_image_multi_patches(batch_subset)
            else:
                outputs = self.image_transformer(pixel_values=batch_subset).pooler_output
            embeddings.append(outputs)

        # Concatenate outputs from all the chunks
        if img_mod == 'cls':
            outputs = torch.cat(embeddings, dim=0).reshape(BS, -1, outputs.shape[-1])
        elif img_mod == 'multi_patches_avg_pooling':
            outputs = torch.cat(embeddings, dim=0).float()  # convert float16 -> 32

        if fc:
            outputs = self.image_fc(outputs)
        if reshape_flag:
            outputs = outputs.reshape(BS, stack_num, *outputs.shape[1:])
        return outputs

    def embed_depth(self, input, return_x_before_fc=False):
        if self.analysis_time:
            start_time = time.time()
        if self.config.depth.bottleneck == 'resnet':
            outputs = self.embed_depth_resnet(input, return_x_before_fc=return_x_before_fc)
            if return_x_before_fc:
                outputs = outputs[0]  # [bs, 128, 4, 4]. Otherwise, [bs, 192, 4, 4]
        if self.analysis_time:
            end_time = time.time()
            print(f'MODEL depth embedding time: {end_time - start_time}')
        return outputs

    def embed_depth_resnet(self, depth, return_x_before_fc=False):
        # set return_x_before_fc to be True when collect dataset (the same as CMA)
        BS = depth.shape[0]
        reshape_flag = False
        if len(depth.shape) == 5:
            # stack depth: [BS, T, 224, 224, 1]
            depth = depth.flatten(0, 1)
            reshape_flag = True
        if self.analysis_time:
            start_time = time.time()
        batch = {'depth': depth}
        outputs = self.depth_encoder(batch, return_x_before_fc=return_x_before_fc)
        if self.analysis_time:
            end_time = time.time()
            print(f'MODEL embed_depth_resnet time: {end_time - start_time}')
            start_time = time.time()
        if return_x_before_fc:
            if reshape_flag:
                outputs0 = outputs[0].reshape(BS, -1, *outputs[0].shape[1:])
                outputs1 = outputs[1].reshape(BS, -1, *outputs[1].shape[1:])
            else:
                outputs0 = outputs[0]
                outputs1 = outputs[1]
            if self.analysis_time:
                end_time = time.time()
                print(f'MODEL embed_depth_resnet reshape_flag time: {end_time - start_time}')
            return [outputs0, outputs1]
        else:
            return outputs

    def init_param(self):
        pass

    def clamp_param(self):
        self.temperature.data.clamp_(-2, 5)
        self.time_scale.data.clamp_(1 / 20, 1)

    def forward(
        self,
        rgb_inputs,
        depth_inputs,
        fc=False,
        do_process=False,
        do_embeds=False,
        img_mod='cls',
    ):
        batch_size = rgb_inputs.shape[0]
        if do_process:
            rgb_inputs = self.process_image(rgb_inputs)
            depth_inputs = self.process_depth(depth_inputs)
        if do_embeds:
            image_embeddings = self.embed_image(rgb_inputs, fc=fc)
            depth_embeddings = self.embed_depth(depth_inputs, fc=fc)
        else:
            image_embeddings = rgb_inputs
            depth_embeddings = depth_inputs

        if len(image_embeddings.shape) == 4:
            image_embeddings = image_embeddings[:, 0, :]
            depth_embeddings = depth_embeddings[:, 0, :]

        if self.config.use_env_drop:
            # directly use dropout on the raw features
            image_embeddings = self.env_drop(image_embeddings)
            # depth_embeddings = self.env_drop(depth_embeddings)

        if self.config.depth.bottleneck == 'resnet':
            depth_resnet_inputs = {'depth_features': depth_embeddings}
            depth_embeds = self.depth_encoder(depth_resnet_inputs)  # [bs,128,4,4]
            depth_embeds = torch.flatten(depth_embeds, 2)  # [bs, 192, 16]
            depth_embeds = self.depth_linear(depth_embeds)

        image_map_embeds = self.dropout(self.img_learnable_linear(image_embeddings))
        depth_map_embeds = self.dropout(self.depth_learnable_linear(depth_embeds))

        if img_mod == 'cls':
            img_depth_embeds = image_map_embeds + depth_map_embeds
            img_depth_embeds = self.layernorm(img_depth_embeds)

        elif img_mod == 'multi_patches_avg_pooling':
            # combine the depth with the full rgb embeds at the 0-pth location.
            ## 0-th location: full depth+img. 2~5: semantic rgb.
            image_map_embeds[:, 0, :] = image_map_embeds[:, 0, :] + depth_map_embeds
            img_depth_embeds = image_map_embeds

        if img_mod == 'cls':
            return img_depth_embeds.unsqueeze(1)
        elif img_mod == 'multi_patches_avg_pooling':
            return img_depth_embeds
