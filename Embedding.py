# Simpler Tokeniser, nothing like Charformer (american spellings :( )
from transformers import AutoTokenizer
import torch
from config import (model_name, embedding_dim, text, tokenizer, vocab_size)
import numpy as np


class Tokenizer:

    def __init__(self):
        self.tokenizer = tokenizer
        self.text = text.strip()  # Remove white spaces around text

    # Everything except the first word → target text
    def shift_text_for_target(self):
        words = self.text.split()
        return " ".join(words[1:])

    # Everything except the last word → input text
    def shift_text_for_input(self):
        words = self.text.split()
        return " ".join(words[:-1])

    # Decoder
    def decode_word(self, token_id):
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    # Encoder for inputs into transformer
    def encode_input(self):
        input_text = self.shift_text_for_input()
        tokens = self.tokenizer(input_text, return_tensors="pt")
        ids = self.tokenizer.encode(input_text)
        return tokens, ids

    # Encoder for target tokens, for loss calculation
    def encode_target(self):
        target_text = self.shift_text_for_target()
        tokens = self.tokenizer(target_text, return_tensors="pt")
        ids = self.tokenizer.encode(target_text)
        return tokens, ids


class Embedding:
    def __init__(self):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # all possible embedding dims
        self.W_e = torch.randn(self.vocab_size, self.embedding_dim, requires_grad=True)

        # For each id in ids, fetches its embedding vector

    def get_embedding_vector(self, ids):
        return self.W_e[ids]  # shape [len(ids), embedding_dim]
