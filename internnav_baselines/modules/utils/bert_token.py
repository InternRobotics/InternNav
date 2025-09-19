import torch
from transformers import RobertaModel, RobertaTokenizer


class BertTokenizer:
    def __init__(self, max_length=80, load_model=False, device='cuda'):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length
        self.padding = True
        if load_model:
            self.model = RobertaModel.from_pretrained('roberta-base').to(device)

    def text_token(self, text, max_length=None):
        max_length = max_length if max_length is not None else self.max_length

        encoded_input = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )  # input_ids, attention_mask
        return encoded_input

    def text_embeddings(self, text):
        encoded_input = self.text_token(text)
        output = self.model(**encoded_input)
        return output

    def convert_tokens_to_string(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        decoded_text = self.tokenizer.convert_tokens_to_string(tokens)
        return decoded_text
