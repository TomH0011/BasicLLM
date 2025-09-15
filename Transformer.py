import math

import torch
from config import (embedding_dim, d_head, attn_heads, hidden_dimension)

class SelfAttention:
    def __init__(self, vec_E):
        self.embedding_dim = embedding_dim
        self.d_head = d_head
        self.attn_heads = attn_heads

        self.W_Q = torch.randn(self.embedding_dim, self.d_head, requires_grad=True) * (1.0 / self.embedding_dim ** 0.5)
        self.W_K = torch.randn(self.embedding_dim, self.d_head, requires_grad=True) * (1.0 / self.embedding_dim ** 0.5)
        self.W_V = torch.randn(self.embedding_dim, self.d_head, requires_grad=True) * (1.0 / self.embedding_dim ** 0.5)
        self.W_O = torch.randn(self.d_head, self.embedding_dim, requires_grad=True) * (1.0 / math.sqrt(self.d_head))

        self.embedding_vectors = vec_E

    def attention(self):
        self.Q = self.embedding_vectors @ self.W_Q
        self.K = self.embedding_vectors @ self.W_K
        self.V = self.embedding_vectors @ self.W_V

        self.scores = (self.Q @ self.K.T) / math.sqrt(self.d_head)
        self.A = torch.softmax(self.scores, dim=-1)
        self.attn_out = self.A @ self.V
        self.output = self.attn_out @ self.W_O

        return self.output

    def backward(self, grad_output):
        """
        grad_output: ∂L/∂output, shape (seq_len, embedding_dim)
        """
        # Through W_O
        grad_W_O = self.attn_out.T @ grad_output
        self.W_O.grad = grad_W_O if self.W_O.grad is None else self.W_O.grad + grad_W_O
        grad_attn_out = grad_output @ self.W_O.T  # (seq_len, d_head)

        # Through attn_out = A @ V
        grad_A = grad_attn_out @ self.V.T
        grad_V = self.A.T @ grad_attn_out

        # Through softmax
        grad_scores = self.A * (grad_A - (grad_A * self.A).sum(dim=-1, keepdim=True))

        # Through scores = Q @ K.T / sqrt(d_head)
        scale = 1.0 / math.sqrt(self.d_head)
        grad_Q = grad_scores @ self.K * scale
        grad_K = grad_scores.T @ self.Q * scale

        # Through Q, K, V linear projections
        grad_W_Q = self.embedding_vectors.T @ grad_Q
        grad_W_K = self.embedding_vectors.T @ grad_K
        grad_W_V = self.embedding_vectors.T @ grad_V

        self.W_Q.grad = grad_W_Q if self.W_Q.grad is None else self.W_Q.grad + grad_W_Q
        self.W_K.grad = grad_W_K if self.W_K.grad is None else self.W_K.grad + grad_W_K
        self.W_V.grad = grad_W_V if self.W_V.grad is None else self.W_V.grad + grad_W_V

        # Gradients wrt input embeddings
        grad_E_Q = grad_Q @ self.W_Q.T
        grad_E_K = grad_K @ self.W_K.T
        grad_E_V = grad_V @ self.W_V.T

        grad_E = grad_E_Q + grad_E_K + grad_E_V
        return grad_E


class MLP:
    def __init__(self):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dimension

        # Parameters (raw tensors, requires_grad not used because we compute grads manually)
        scale_up = 1.0 / (self.embedding_dim ** 0.5)
        scale_down = 1.0 / (self.hidden_dim ** 0.5)
        self.W_up = torch.randn(self.embedding_dim, self.hidden_dim) * scale_up
        self.W_down = torch.randn(self.hidden_dim, self.embedding_dim) * scale_down
        self.b_up = torch.zeros(self.hidden_dim)
        self.b_down = torch.zeros(self.embedding_dim)

        # initialize grads to zeros (same shape as params)
        self.W_up_grad = torch.zeros_like(self.W_up)
        self.W_down_grad = torch.zeros_like(self.W_down)
        self.b_up_grad = torch.zeros_like(self.b_up)
        self.b_down_grad = torch.zeros_like(self.b_down)

    def forward(self, embedding_vectors):
        # embedding_vectors: [N, embedding_dim]  (N = seq_len or batch_size)
        self.E = embedding_vectors                  # save input for backward
        self.Z_up = embedding_vectors @ self.W_up + self.b_up  # [N, hidden]
        self.H = torch.relu(self.Z_up)               # [N, hidden]
        mlp_output = self.H @ self.W_down + self.b_down       # [N, embedding_dim]
        return mlp_output

    def backward(self, grad_output):
        # grad_output: ∂L/∂output, shape [N, embedding_dim]
        # N = grad_output.shape[0]

        # grads for W_down and b_down
        grad_W_down = self.H.T @ grad_output         # [hidden, embedding_dim]
        grad_b_down = grad_output.sum(dim=0)         # [embedding_dim]

        # backprop into H
        grad_H = grad_output @ self.W_down.T         # [N, hidden]

        # backprop through ReLU (Z_up > 0)
        relu_mask = (self.Z_up > 0).to(dtype=grad_H.dtype)  # [N, hidden]
        grad_Z_up = grad_H * relu_mask                # [N, hidden]

        # grads for W_up and b_up
        grad_W_up = self.E.T @ grad_Z_up             # [embedding_dim, hidden]
        grad_b_up = grad_Z_up.sum(dim=0)             # [hidden]

        # gradient wrt input E (to pass to previous layer)
        grad_E = grad_Z_up @ self.W_up.T             # [N, embedding_dim]

        # store/accumulate parameter gradients (optionally average by N)
        # invN = 1.0 / float(N)
        self.W_down_grad += grad_W_down
        self.b_down_grad += grad_b_down
        self.W_up_grad += grad_W_up
        self.b_up_grad += grad_b_up

        return grad_E


