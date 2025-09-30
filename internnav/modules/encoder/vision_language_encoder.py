import torch.nn as nn

from .bert_backbone import CrossmodalEncoder


class VisionLanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_modal_encoder = CrossmodalEncoder(config)

    def forward(
        self,
        q_embeds,
        kv_embeds,
        kv_masks,
        q_masks=None,
        output_attentions=False,
        do_self_attn=True,
    ):
        outputs = self.cross_modal_encoder(
            q_embeds=q_embeds,
            q_masks=q_masks,
            kv_embeds=kv_embeds,
            kv_masks=kv_masks,
            output_attentions=output_attentions,
            do_self_attn=do_self_attn,
        )
        if output_attentions:
            outputs, attentions = outputs
            return outputs, attentions

        return outputs
