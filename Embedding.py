# Simpler Tokeniser, nothing like Charformer (american spellings :( )
from transformers import AutoTokenizer
import torch
from config import (model_name, embedding_dim, text, tokenizer, vocab_size)
import numpy as np


class Tokenizer:

    def __init__(self, prompt):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.text = text

    def decode_word(self, token_id):
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    def encode_prompt(self):
        tokens = self.tokenizer(self.text, return_tensors='pt')
        print(f'Number of tokens: {len(tokens[0])}')
        id = self.tokenizer.encode(self.text)
        return tokens, id


class Embedding:
    def __init__(self):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding_matrix = np.random.randn(self.vocab_size, self.embedding_dim)  # all possible embedding dims

    # For each id in ids, fetches its embedding vector
    def get_embedding_vector(self, ids):
        return self.embedding_matrix[ids]  # shape [len(ids), embedding_dim]

