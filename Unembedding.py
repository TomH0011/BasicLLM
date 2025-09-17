import numpy as np
from config import vocab_size, embedding_dim, device
import torch


class Unembed:
    def __init__(self):
        self.last_input = None
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Don't use requires_grad=True for manual gradient computation
        self.W_u = torch.randn(self.embedding_dim, self.vocab_size, device=device) * (1.0 / self.embedding_dim ** 0.5)
        self.W_u.grad = None

    def unembed(self, vec_E):
        self.last_input = vec_E  # store for backward
        logits = vec_E @ self.W_u
        return logits

    def backward(self, grad_logits):
        # Detach to prevent graph buildup
        grad_logits = grad_logits.detach()

        # Gradients wrt W_u - detach before accumulating
        grad_W_u = (self.last_input.T @ grad_logits).detach()

        if self.W_u.grad is None:
            self.W_u.grad = grad_W_u
        else:
            self.W_u.grad = self.W_u.grad.detach() + grad_W_u

        # Gradients wrt input embeddings
        grad_E = grad_logits @ self.W_u.T

        return grad_E.detach()