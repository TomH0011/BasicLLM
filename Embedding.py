# Fixed Embedding.py with proper memory management and token caching
import gc
import os
from transformers import AutoTokenizer
import torch
from config import (model_name, embedding_dim, text, tokenizer, vocab_size, device, tokenized_data_path)
import numpy as np


class Tokenizer:
    def __init__(self):
        self.tokenizer = tokenizer
        self.text = text.strip()

    def encode_text(self):
        """Encode the full text and return input_ids and target_ids properly aligned"""

        # Check if tokenized data already exists
        if os.path.exists(tokenized_data_path):
            print(f'Loading cached tokenized data from {tokenized_data_path}...')
            try:
                cached_data = torch.load(tokenized_data_path, map_location='cpu')
                input_ids = cached_data['input_ids']
                target_ids = cached_data['target_ids']

                # Clear text from memory immediately
                del self.text
                gc.collect()

                print('Cached tokenized data loaded successfully!')
                return input_ids, target_ids

            except Exception as e:
                print(f'Error loading cached data: {e}')
                print('Proceeding with fresh tokenization...')

        # Fresh tokenization if no cache exists or cache failed to load
        print('Starting to tokenize text, this may take a while...')
        full_ids = self.tokenizer.encode(self.text)
        print('Text tokenized!')

        # Input: all tokens except the last one
        input_ids = full_ids[:-1]
        # Target: all tokens except the first one (shifted by 1)
        target_ids = full_ids[1:]

        # Save tokenized data for future use
        print(f'Saving tokenized data to {tokenized_data_path}...')
        try:
            torch.save({
                'input_ids': input_ids,
                'target_ids': target_ids,
                'vocab_size': self.tokenizer.vocab_size,
                'model_name': model_name
            }, tokenized_data_path)
            print('Tokenized data saved successfully!')
        except Exception as e:
            print(f'Warning: Could not save tokenized data: {e}')

        # Clear text and intermediate data from memory
        del self.text, full_ids
        gc.collect()

        print('Text cleared from memory.')
        return input_ids, target_ids

    def decode_word(self, token_id):
        return self.tokenizer.decode([token_id], skip_special_tokens=True)


class Embedding:
    def __init__(self):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.last_ids = None

        # Initialize embedding matrix - no requires_grad for manual gradients
        self.W_e = torch.randn(self.vocab_size, self.embedding_dim, device=device) * 0.02
        self.W_e.grad = None

    def get_embedding_vector(self, ids):
        """Fetch embedding vectors for given token ids"""
        # Store the entire tensor for the backward pass
        self.last_ids = ids
        # Perform the vectorised lookup
        return self.W_e[ids]

    def backward(self, grad_embeddings):
        """Accumulate gradients for embedding matrix"""
        # Detach to prevent graph buildup
        grad_embeddings = grad_embeddings.detach()

        if self.W_e.grad is None:
            self.W_e.grad = torch.zeros_like(self.W_e)

        # Use index_add_ for efficient gradient accumulation
        # This is more memory efficient than a loop
        self.W_e.grad.index_add_(0, self.last_ids, grad_embeddings)