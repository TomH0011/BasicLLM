import numpy as np
from config import vocab_size, embedding_dim, device
import torch


class Unembed:
    def __init__(self):
        self.last_input = None
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_u = torch.randn(self.embedding_dim, self.vocab_size, requires_grad=True, device=device) * (1.0 / self.embedding_dim ** 0.5)

    def unembed(self, vec_E):
        self.last_input = vec_E  # store for backward
        logits = vec_E @ self.W_u
        return logits

    def backward(self, grad_logits):
        # Gradients wrt W_u
        grad_W_u = self.last_input.T @ grad_logits
        self.W_u.grad = grad_W_u if self.W_u.grad is None else self.W_u.grad + grad_W_u

        # Gradients wrt input embeddings
        grad_E = grad_logits @ self.W_u.T
        return grad_E
