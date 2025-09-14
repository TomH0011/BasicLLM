import numpy as np
from config import vocab_size, embedding_dim
import torch


class Unembed:
    def __init__(self):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_u = torch.randn(self.embedding_dim, self.vocab_size, requires_grad=True) * (1.0 / self.embedding_dim ** 0.5)

    def unembed(self, vec_E):
        return vec_E @ self.W_u
