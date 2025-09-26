from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .navdp import NavDP_Policy_DPT_CriticSum_DAT


def build_navdp(navdp_cfg):
    navdp_version = getattr(navdp_cfg, "navdp_version", 0.0)
    if navdp_version > 0.0:
        memory_size = 2
    else:
        memory_size = 3

    navdp = NavDP_Policy_DPT_CriticSum_DAT(
        memory_size=memory_size, navdp_pretrained=navdp_cfg.navdp_pretrained, navdp_version=navdp_version
    )
    navdp.load_model()
    return navdp


class InternVLAN1MetaModel:
    def __init__(self, config):
        super(InternVLAN1MetaModel, self).__init__(config)
        if hasattr(config, "navdp"):
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
            self.navdp = build_navdp(config)

    def initialize_vision_modules(self, model_args):
        if getattr(self, 'navdp', None) is None:
            self.config.navdp = model_args.navdp
            self.config.navdp_pretrained = model_args.navdp_pretrained
            self.navdp = build_navdp(model_args)

        self.config.n_query = model_args.n_query
        if getattr(self, 'latent_queries', None) is None:
            print("random initiation the latent_queries !!!")
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))


class InternVLAN1MetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_navdp(self):
        return self.get_model().navdp

    def get_mm_projector(self):
        return self.get_model().mm_projector

    def get_n_query(self):
        return self.get_model().config.n_query


TRAJ_START_TOKEN_INDEX = 151665
IMAGE_TOKEN_INDEX = 151655
TRAJ_TOKEN_INDEX = 151667


class InternVLAN1ModelConfig(Qwen2_5_VLConfig):
    model_type = "internvla_n1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)


class InternVLAN1Model(InternVLAN1MetaModel, Qwen2_5_VLModel):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(InternVLAN1Model, self).__init__(config)


class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type == "internvla_n1"

        self.model = InternVLAN1Model(config)
        self.rope_deltas = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )
        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        # add for QwenVL kv cache
        model_inputs["pixel_values"] = pixel_values
        model_inputs["pixel_values_videos"] = pixel_values_videos

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        t_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        raw_input_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            if pixel_values is not None and n_image_tokens > 0:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds = image_embeds[-n_image_tokens:]
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            n_traj_tokens = (input_ids == TRAJ_TOKEN_INDEX).sum().item()
            traj_idx = input_ids == TRAJ_TOKEN_INDEX
            latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
            H = latent_queries.shape[-1]
            latent_queries = latent_queries.contiguous().view(-1, H)
            if n_traj_tokens != 0:
                inputs_embeds[traj_idx] = latent_queries

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            elif n_image_tokens > 0:  # using only for kv cache
                attention_mask = attention_mask[:, : raw_input_ids.shape[1]]
                position_ids, rope_deltas = self.get_rope_index(
                    raw_input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = position_ids[:, :, -input_ids.shape[1] :]
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate_latents(self, input_ids, pixel_values, image_grid_thw):
        input_ids.to(self.get_model().device)
        input_ids = torch.cat([input_ids, torch.tensor([[TRAJ_START_TOKEN_INDEX]]).to(input_ids.device)], dim=1)
        text_embeds = self.get_model().embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
        image_idx = input_ids == IMAGE_TOKEN_INDEX
        N_QUERY = self.get_n_query()
        input_ids = torch.cat([input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY]).to(input_ids.device)], dim=1)

        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).unsqueeze(0)

        text_embeds[image_idx] = image_embeds.to(text_embeds.device)[: image_idx.sum(), :]

        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)

        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)
        outputs = self.model(
            inputs_embeds=text_embeds,
            position_ids=position_ids,
            # attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1][:, -N_QUERY:, :]
        return hidden_states

    def generate_traj(self, traj_latents, images_dp=None, depths_dp=None, use_async=False):
        if use_async:
            all_trajs = self.model.navdp.predict_pointgoal_action_async(
                traj_latents.to(self.get_model().device), images_dp, depths_dp, vlm_mask=None
            )
        else:
            all_trajs = self.model.navdp.predict_pointgoal_action(
                traj_latents.to(self.get_model().device), vlm_mask=None
            )
        return all_trajs
