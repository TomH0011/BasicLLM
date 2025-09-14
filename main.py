from Embedding import Tokenizer, Embedding
import torch
from Transformer import SelfAttention, MLP
from Unembedding import Unembed
from config import num_layers
from Loss import CrossEntropyLoss
from optimiser import SGDWithMomentum


class Main:

    def __init__(self):
        self.embedding_vectors = []
        self.num_layers = num_layers
        self.tokenizer = Tokenizer()
        self.embedding = Embedding()
        self.perceptron_layers = [MLP() for _ in range(num_layers)]
        self.attention_layers = []
        self.unembed = Unembed()
        self.loss = CrossEntropyLoss()
        self.final_probabilities = None
        self.params = None

    def main(self):
        tokens, input_ids = self.tokenizer.encode_input()

        for id in input_ids:
            vec = self.embedding.get_embedding_vector(id)
            self.embedding_vectors.append(vec)
        self.embedding_vectors = torch.stack(
            self.embedding_vectors)  # Turn into tensor with shape (seq_len, embedding_dim)

        print(f'number of embedding vectors in weights_e: {len(self.embedding_vectors)}')
        print(f'W_e shape: {self.embedding_vectors.shape}')

        embeddings = self.embedding_vectors
        self.attention_layers = []
        mlp_outputs = []

        # -------------------------- forward pass --------------------------
        embeddings = self.embedding_vectors  # Running variable
        for layer_idx in range(self.num_layers):
            # Self-Attention
            attention = SelfAttention(embeddings)
            self.attention_layers.append(attention)
            attn_out = attention.attention()
            embeddings = embeddings + attn_out  # residual

            # MLP
            mlp = self.perceptron_layers[layer_idx]
            mlp_out = mlp.forward(embeddings)
            mlp_outputs.append(mlp_out)
            embeddings = embeddings + mlp_out  # residual

        print(f'mlp.res shape: {embeddings.shape}')

        self.params = self.collect_params()
        self.optimiser = SGDWithMomentum(params=self.params)

        # Unembedding
        logits = self.unembed.unembed(embeddings)
        print(f'logits shape: {logits.shape}')

        # Calculate Loss
        _, target_ids = self.tokenizer.encode_target()
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        loss = self.loss.calulate_cross_entropy_loss_from_logits(logits, target_ids)
        print(f'The calculated loss is: {loss}')

        # -------------------------- Backward pass/Back propagation --------------------------
        grad_logits = self.loss.backward()
        grad_embeddings = self.unembed.backward(grad_logits)

        # Loop backward through layers
        for layer_idx in reversed(range(self.num_layers)):
            # Backprop through MLP
            mlp = self.perceptron_layers[layer_idx]
            grad_MLP_input = grad_embeddings  # residual gradient
            grad_embeddings = mlp.backward(grad_MLP_input)

            # Backprop through Attention
            attention = self.attention_layers[layer_idx]
            grad_attn_input = grad_embeddings  # residual gradient
            grad_embeddings = attention.backward(grad_attn_input)

        # Backprop into embeddings
        self.embedding.backward(grad_embeddings)

        # -------------------------- optimiser --------------------------

        self.optimiser.step()
        self.optimiser.zero_grad()

        # -------------------------- Make Predictions --------------------------
        # Want probs to sum to 1
        probabilities = torch.softmax(logits, dim=-1)
        self.final_probabilities = probabilities[-1]
        next_id = torch.multinomial(self.final_probabilities, num_samples=1).item()
        next_token = self.tokenizer.decode_word(next_id)
        print("Next token:", next_token)

    def show_top_10_predictions(self, topk=10):
        if self.final_probabilities is None:
            raise ValueError("Run main() first!")
        probs, indices = torch.topk(self.final_probabilities, k=topk)
        print("\nTop 10 predictions:")
        for i in range(topk):
            token_id = indices[i].item()
            token = self.tokenizer.decode_word(token_id)
            prob = probs[i].item()
            print(f"{token:15s}  {prob:.4f}")

    def collect_params(self):
        params = []

        # embedding param
        params.append(self.embedding.W_e)

        # unembedding param
        params.append(self.unembed.W_u)

        # MLP params
        for mlp_param in self.perceptron_layers:
            params.extend([mlp_param.W_up, mlp_param.b_up, mlp_param.W_down, mlp_param.b_down])

        # Attention (all layers)
        for attn in self.attention_layers:
            params.extend([attn.W_Q, attn.W_K, attn.W_V, attn.W_O])

        return params

if __name__ == '__main__':
    model = Main()
    model.main()
    model.show_top_10_predictions()
