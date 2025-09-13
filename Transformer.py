import math

import torch
from config import (embedding_dim, d_head, attn_heads, hidden_dimesnion)

class SelfAttention:
    def __init__(self, vec_E):
        self.embedding_dim = embedding_dim
        self.d_head = d_head
        self.attn_heads = attn_heads

        # Initialise the Weight matrices
        self.W_Q = torch.randn(self.embedding_dim, self.d_head) * (1.0 / self.embedding_dim ** 0.5)
        self.W_K = torch.randn(self.embedding_dim, self.d_head) * (1.0 / self.embedding_dim ** 0.5)
        self.W_V = torch.randn(self.embedding_dim, self.d_head) * (1.0 / self.embedding_dim ** 0.5)

        # For up-scaling output from 64 upto 128
        self.W_O = torch.randn(self.d_head, self.embedding_dim) * (1.0 / math.sqrt(self.d_head))

        # E_i = W_e[i]
        self.vec_E = vec_E

        # Query, Key, Value Vectors
        self.vec_Q = self.vec_E @ self.W_Q
        self.vec_K = self.vec_E @ self.W_K
        self.vec_V = self.vec_E @ self.W_V

    # def query_vector(self):
    #     # vec_Q = vec_E_i @ W_Q
    #     vec_Q = self.vec_E @ self.W_Q
    #     return vec_Q
    #
    # def key_vector(self):
    #     # vec_K = vec_E @ W_K
    #     vec_K = self.vec_E @ self.W_K
    #     return vec_K
    #
    # def value_vector(self):
    #     # vec_V = vec_E_i @ W_V
    #     vec_V = self.vec_E @ self.W_V
    #     return vec_V

    def attention(self):
        # Attention(vec_Q, vec_K, vec_V) = Softmax((vec_K_tr vec_Q) / sqrt(d_k)) vec_V
        scores = (self.vec_Q @ self.vec_K.T) / math.sqrt(self.d_head)  # (seq_len, seq_len)
        print(f'shape of scores: {scores.shape}')
        weights = torch.softmax(scores, dim=-1)  # (seq_len, seq_len)
        print(f'shape of weights: {weights.shape}')
        output = weights @ self.vec_V  # (seq_len, d_head)
        print(f'shape of attn_out: {output.shape}')
        output = output @ self.W_O
        print(f'shape of attn_out after up-scaling dimensions: {output.shape}')
        return output


class MLP:
    def __init__(self):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dimesnion

        # Weights
        self.W_up = torch.randn(self.embedding_dim, self.hidden_dim) * (1.0 / self.embedding_dim ** 0.5)
        self.W_down = torch.randn(self.hidden_dim, self.embedding_dim) * (1.0 / self.hidden_dim ** 0.5)

        # Biases
        self.b_up = torch.zeros(self.hidden_dim)
        self.b_down = torch.zeros(self.embedding_dim)

    def forward(self, E_i):
        # Linear-up -> Non-linear -> linear-down
        # W_down f(x @ W_up _+ bias_1) + bias_2
        # x = vec_E

        hidden = torch.relu(E_i @ self.W_up + self.b_up)
        output = hidden @ self.W_down + self.b_down

        return output
