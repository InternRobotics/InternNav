import torch.nn as nn

from .bert_backbone import RobertaEmbeddings, RobertaLayer, extend_neg_masks


class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_text_encoder

        self.embeddings = RobertaEmbeddings(config)

        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(self.num_l_layers)])
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_inputs, txt_masks=None):
        txt_inputs = txt_inputs.long()
        if txt_masks is None:
            txt_masks = (txt_inputs != 1).to(txt_inputs.device)
        txt_embeds = self.embeddings(txt_inputs)

        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()

        return txt_embeds, txt_masks, txt_embeds[:, 0, :]
