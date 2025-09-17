import torch
import torch.nn as nn
import torch.nn.functional as F

from internnav.configs.model.base_encoders import TextEncoder

from ..basemodel.LongCLIP.model import longclip


class InstructionLongCLIPEncoder(nn.Module):
    def __init__(self, config: TextEncoder):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_text_encoder

        self.text_transformer, _ = longclip.load(config.model_path)

        # del visual part
        del self.text_transformer.visual

        if not self.update_lang_bert:
            for name, param in self.text_transformer.named_parameters():
                param.requires_grad = False

    def encode_text(self, text, return_full):
        text_transformer = self.text_transformer
        try:
            data_type = text_transformer.visual.conv1.weight.dtype
        except Exception:
            data_type = text_transformer.transformer.resblocks[0].mlp.c_fc.weight.dtype
        x = text_transformer.token_embedding(text).type(data_type)  # [batch_size, n_ctx, d_model]

        x = (
            x
            + (text_transformer.positional_embedding.to(x.device) * text_transformer.mask1.to(x.device)).type(data_type).to(x.device)
            + (text_transformer.positional_embedding_res.to(x.device) * text_transformer.mask2.to(x.device)).type(data_type).to(x.device)
        )

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = text_transformer.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = text_transformer.ln_final(x).type(data_type)

        if return_full:
            x_eot = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ text_transformer.text_projection
            return x_eot, x

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ text_transformer.text_projection

        return x

    def forward(
        self,
        txt_inputs,
        txt_masks=None,
        need_txt_extraction=True,
    ):
        if need_txt_extraction:
            txt_inputs = txt_inputs.long()
            # padding the length of text to 248
            if txt_inputs.size(1) < 248:
                txt_inputs = F.pad(txt_inputs, (0, 248 - txt_inputs.size(1)), value=0)
            if txt_masks is None:
                txt_masks = (txt_inputs != 0).to(txt_inputs.device)

            (
                txt_cls_embeds,
                txt_full_embeds,
            ) = self.encode_text(txt_inputs, return_full=True)
            txt_cls_embeds = txt_cls_embeds.type(torch.float32)
            txt_full_embeds = txt_full_embeds.type(torch.float32)
        else:
            txt_full_embeds = txt_inputs
            txt_masks = None
            txt_cls_embeds = None

        return txt_full_embeds, txt_masks, txt_cls_embeds
