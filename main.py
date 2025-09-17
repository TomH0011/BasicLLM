# Optimized main.py with better memory management
from Embedding import Tokenizer, Embedding
import torch
from Transformer import SelfAttention, MLP
from Unembedding import Unembed
from config import num_layers, device
from Loss import CrossEntropyLoss
from optimiser import SGDWithMomentum
import gc  # For explicit garbage collection


class Main:
    def __init__(self):
        self.num_layers = num_layers
        self.tokenizer = Tokenizer()
        self.embedding = Embedding()
        self.perceptron_layers = [MLP() for _ in range(num_layers)]
        self.attention_layers = []
        self.unembed = Unembed()
        self.loss_fn = CrossEntropyLoss()
        self.params = None
        self.optimiser = None

    def forward_pass(self, input_ids):
        """Run forward pass through the model"""
        # Get embeddings with a single, vectorized call
        embedding_vectors = self.embedding.get_embedding_vector(input_ids)

        # Store attention layers for backward pass
        self.attention_layers = []

        # Forward through transformer layers
        embeddings = embedding_vectors
        for layer_idx in range(self.num_layers):
            # Self-Attention
            attention = SelfAttention(embeddings)
            self.attention_layers.append(attention)
            attn_out = attention.attention()
            embeddings = embeddings + attn_out  # residual connection

            # MLP
            mlp = self.perceptron_layers[layer_idx]
            mlp_out = mlp.forward(embeddings)
            embeddings = embeddings + mlp_out  # residual connection

        # Unembed to get logits
        logits = self.unembed.unembed(embeddings)

        return logits, embeddings

    def backward_pass(self, grad_logits):
        """Run backward pass through the model"""
        # Ensure grad_logits is detached
        grad_logits = grad_logits.detach()

        # Backprop through unembedding
        grad_embeddings = self.unembed.backward(grad_logits)

        # Loop backward through transformer layers
        for layer_idx in reversed(range(self.num_layers)):
            # Backprop through MLP (with residual)
            mlp = self.perceptron_layers[layer_idx]
            grad_from_mlp = mlp.backward(grad_embeddings)

            # Backprop through Attention (with residual)
            attention = self.attention_layers[layer_idx]
            grad_from_attn = attention.backward(grad_embeddings)

            # Combine gradients (both paths contribute due to residual connections)
            grad_embeddings = grad_from_mlp + grad_from_attn

        # Backprop into embedding layer
        self.embedding.backward(grad_embeddings)

    def collect_params(self):
        """Collect all parameters for optimizer"""
        params = [self.embedding.W_e, self.unembed.W_u]

        # MLP params
        for mlp in self.perceptron_layers:
            params.extend([mlp.W_up, mlp.b_up, mlp.W_down, mlp.b_down])

        # Attention params (collected after forward pass)
        for attn in self.attention_layers:
            params.extend([attn.W_Q, attn.W_K, attn.W_V, attn.W_O])

        return params

    def train_step(self, input_ids, target_ids):
        """Single training step with memory management"""
        # Forward pass
        with torch.no_grad():  # Ensure no autograd graph is created
            logits, _ = self.forward_pass(input_ids)

        # Calculate loss
        loss = self.loss_fn.calculate_cross_entropy_loss_from_logits(logits, target_ids)

        # Backward pass
        grad_logits = self.loss_fn.backward()
        self.backward_pass(grad_logits)

        # Optimizer step
        if self.optimiser is None:
            self.params = self.collect_params()
            self.optimiser = SGDWithMomentum(params=self.params)

        self.optimiser.step()
        self.optimiser.zero_grad()

        # Clean up intermediate tensors from attention layers
        for attn in self.attention_layers:
            del attn.Q, attn.K, attn.V, attn.scores, attn.A, attn.attn_out, attn.output

        # Clean up MLP intermediate tensors
        for mlp in self.perceptron_layers:
            if hasattr(mlp, 'E'):
                del mlp.E, mlp.Z_up, mlp.H

        return loss.item()

    def train(self, num_epochs=100):
        """Main training loop with memory management"""
        # Get training data
        input_ids, target_ids = self.tokenizer.encode_text()
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)

        print(f"Input shape: {input_ids.shape}")
        print(f"Target shape: {target_ids.shape}")
        print(f"Starting training for {num_epochs} epochs...")

        losses = []
        for epoch in range(num_epochs):
            loss = self.train_step(input_ids, target_ids)
            losses.append(loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

            # Periodic garbage collection to free memory
            if (epoch + 1) % 20 == 0:
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()

        return losses

    def generate_next_token(self, prompt_text=None):
        """Generate next token given a prompt"""
        if prompt_text is None:
            # Use the training text as prompt
            input_ids, _ = self.tokenizer.encode_text()
        else:
            # Tokenize the prompt
            input_ids = self.tokenizer.tokenizer.encode(prompt_text)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)

        # Forward pass with no_grad to save memory
        with torch.no_grad():
            logits, _ = self.forward_pass(input_ids)

        # Get probabilities for last position
        final_logits = logits[-1]
        probs = torch.softmax(final_logits, dim=-1)

        # Sample next token
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_token = self.tokenizer.decode_word(next_id)

        return next_token, probs

    def show_top_predictions(self, probs, topk=10):
        """Show top k predictions"""
        values, indices = torch.topk(probs, k=topk)
        print(f"\nTop {topk} predictions:")
        for i in range(topk):
            token_id = indices[i].item()
            token = self.tokenizer.decode_word(token_id)
            prob = values[i].item()
            print(f"{token:15s}  {prob:.4f}")


if __name__ == '__main__':
    print('Initializing model...')
    model = Main()

    # Train the model
    losses = model.train(num_epochs=200)  # Can now handle more epochs!

    print('\nTraining finished!')
    print(f'Final loss: {losses[-1]:.4f}')

    # Generate next token
    next_token, probs = model.generate_next_token()
    print(f'\nPredicted next token: "{next_token}"')

    # Show top predictions
    model.show_top_predictions(probs, topk=10)