import math
import torch
from config import (embedding_dim, d_head, attn_heads, hidden_dimension, device)


class SelfAttention:
    def __init__(self, vec_E):
        self.embedding_dim = embedding_dim
        self.d_head = d_head
        self.attn_heads = attn_heads

        # Don't use requires_grad=True for manual gradient computation
        self.W_Q = torch.randn(self.embedding_dim, self.d_head, device=device) * (1.0 / self.embedding_dim ** 0.5)
        self.W_K = torch.randn(self.embedding_dim, self.d_head, device=device) * (1.0 / self.embedding_dim ** 0.5)
        self.W_V = torch.randn(self.embedding_dim, self.d_head, device=device) * (1.0 / self.embedding_dim ** 0.5)
        self.W_O = torch.randn(self.d_head, self.embedding_dim, device=device) * (1.0 / math.sqrt(self.d_head))

        # Initialize gradients
        self.W_Q.grad = None
        self.W_K.grad = None
        self.W_V.grad = None
        self.W_O.grad = None

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
        # Detach grad_output to prevent graph buildup
        grad_output = grad_output.detach()

        # Through W_O
        grad_W_O = (self.attn_out.T @ grad_output).detach()
        if self.W_O.grad is None:
            self.W_O.grad = grad_W_O
        else:
            self.W_O.grad = self.W_O.grad.detach() + grad_W_O

        grad_attn_out = grad_output @ self.W_O.T

        # Through attn_out = A @ V
        grad_A = grad_attn_out @ self.V.T
        grad_V = self.A.T @ grad_attn_out

        # Through softmax
        grad_scores = self.A * (grad_A - (grad_A * self.A).sum(dim=-1, keepdim=True))

        # Through scores = Q @ K.T / sqrt(d_head)
        scale = 1.0 / math.sqrt(self.d_head)
        grad_Q = grad_scores @ self.K * scale
        grad_K = grad_scores.T @ self.Q * scale

        # Through Q, K, V linear projections - DETACH before accumulating
        grad_W_Q = (self.embedding_vectors.T @ grad_Q).detach()
        grad_W_K = (self.embedding_vectors.T @ grad_K).detach()
        grad_W_V = (self.embedding_vectors.T @ grad_V).detach()

        if self.W_Q.grad is None:
            self.W_Q.grad = grad_W_Q
        else:
            self.W_Q.grad = self.W_Q.grad.detach() + grad_W_Q

        if self.W_K.grad is None:
            self.W_K.grad = grad_W_K
        else:
            self.W_K.grad = self.W_K.grad.detach() + grad_W_K

        if self.W_V.grad is None:
            self.W_V.grad = grad_W_V
        else:
            self.W_V.grad = self.W_V.grad.detach() + grad_W_V

        # Gradients wrt input embeddings
        grad_E_Q = grad_Q @ self.W_Q.T
        grad_E_K = grad_K @ self.W_K.T
        grad_E_V = grad_V @ self.W_V.T
        grad_E = grad_E_Q + grad_E_K + grad_E_V

        return grad_E.detach()


class MLP:
    def __init__(self):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dimension

        # Parameters
        scale_up = 1.0 / (self.embedding_dim ** 0.5)
        scale_down = 1.0 / (self.hidden_dim ** 0.5)
        self.W_up = torch.randn(self.embedding_dim, self.hidden_dim, device=device) * scale_up
        self.W_down = torch.randn(self.hidden_dim, self.embedding_dim, device=device) * scale_down
        self.b_up = torch.zeros(self.hidden_dim, device=device)
        self.b_down = torch.zeros(self.embedding_dim, device=device)

        # Initialize gradients as attributes (not separate tensors)
        self.W_up.grad = None
        self.W_down.grad = None
        self.b_up.grad = None
        self.b_down.grad = None

    def forward(self, embedding_vectors):
        self.E = embedding_vectors
        self.Z_up = embedding_vectors @ self.W_up + self.b_up
        self.H = torch.relu(self.Z_up)
        mlp_output = self.H @ self.W_down + self.b_down
        return mlp_output

    def backward(self, grad_output):
        # Detach to prevent graph buildup
        grad_output = grad_output.detach()

        # Grads for W_down and b_down
        grad_W_down = (self.H.T @ grad_output).detach()
        grad_b_down = grad_output.sum(dim=0).detach()

        # Backprop into H
        grad_H = grad_output @ self.W_down.T

        # Backprop through ReLU
        relu_mask = (self.Z_up > 0).to(dtype=grad_H.dtype)
        grad_Z_up = grad_H * relu_mask

        # Grads for W_up and b_up
        grad_W_up = (self.E.T @ grad_Z_up).detach()
        grad_b_up = grad_Z_up.sum(dim=0).detach()

        # Gradient wrt input
        grad_E = grad_Z_up @ self.W_up.T

        # Accumulate gradients - use detached gradients
        if self.W_down.grad is None:
            self.W_down.grad = grad_W_down
        else:
            self.W_down.grad = self.W_down.grad.detach() + grad_W_down

        if self.b_down.grad is None:
            self.b_down.grad = grad_b_down
        else:
            self.b_down.grad = self.b_down.grad.detach() + grad_b_down

        if self.W_up.grad is None:
            self.W_up.grad = grad_W_up
        else:
            self.W_up.grad = self.W_up.grad.detach() + grad_W_up

        if self.b_up.grad is None:
            self.b_up.grad = grad_b_up
        else:
            self.b_up.grad = self.b_up.grad.detach() + grad_b_up

        return grad_E.detach()