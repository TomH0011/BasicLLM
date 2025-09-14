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
        self.last_ids = None

        # all possible embedding dims
        self.W_e = torch.randn(self.vocab_size, self.embedding_dim, requires_grad=True)

    # fetches embedding vector for a given id
    def get_embedding_vector(self, ids):
        # Always convert to list if it's a single int
        if isinstance(ids, int):
            self.last_ids = [ids]
        else:
            self.last_ids = list(ids)  # make sure it's a list
        return self.W_e[ids]

    def backward(self, grad_embeddings):
        if self.W_e.grad is None:
            self.W_e.grad = torch.zeros_like(self.W_e)
        # accumulate gradients into W_e
        for i, idx in enumerate(self.last_ids):
            # Update corresponding row in W_e
            if self.W_e.grad is None:
                self.W_e.grad = torch.zeros_like(self.W_e)
            self.W_e.grad[idx] += grad_embeddings[i]
