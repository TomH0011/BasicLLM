from Embedding import Tokenizer, Embedding
import torch
from Transformer import SelfAttention, MLP
from Unembedding import Unembed
from config import num_layers
from Loss import CrossEntropyLoss


class Main:

    def __init__(self):
        self.W_e = []
        self.num_layers = num_layers
        self.tokenizer = Tokenizer()
        self.embedding = Embedding()
        self.perceptron = MLP()
        self.unembed = Unembed()
        self.loss = CrossEntropyLoss()

    def main(self):
        tokens, input_ids = self.tokenizer.encode_input()
        # print(f'tokens: {tokens} id: {id}')

        for id in input_ids:
            vec = self.embedding.get_embedding_vector(id)
            vec = torch.tensor(vec, dtype=torch.float32)  # convert NumPy → Tensor
            self.W_e.append(vec)

        self.W_e = torch.stack(self.W_e)  # Turn into tensor with shape (seq_len, embedding_dim)
        print(f'number of embedding vectors in weights_e: {len(self.W_e)}')
        print(f'W_e shape: {self.W_e.shape}')

        # ---------------------------------------------------------------
        # Actual training loop Begins here
        # ---------------------------------------------------------------
        x = self.W_e  # Running variable
        for _ in range(self.num_layers):

            attention = SelfAttention(x)
            attn_out = attention.attention()
            x = x + attn_out  # residual connection

            mlp_out = self.perceptron.forward(x)
            x = x + mlp_out  # residual connection

        # ---------------------------------------------------------------
        # Training loop ends here
        # ---------------------------------------------------------------
        print(f'mlp.res shape: {x.shape}')

        # What we will make predictions on
        # final_vec = x[-1]
        # print(f'final vector shape: {final_vec.shape}')
        # logits = self.unembed.unembed(final_vec)
        logits = self.unembed.unembed(x)
        print(f'logits shape: {logits.shape}')

        # Calculate Loss
        _, target_ids = self.tokenizer.encode_target()
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        loss = self.loss.calulate_cross_entropy_loss_from_logits(logits, target_ids)
        print(f'The calculated loss is: {loss}')
        # Want probs to sum to 1
        probabilities = torch.softmax(logits, dim=-1)
        final_probabilities = probabilities[-1]
        print(f'probs shape: {probabilities.shape}')

        next_id = torch.multinomial(final_probabilities, num_samples=1).item()
        next_token = self.tokenizer.decode_word(next_id)

        print("next token:", next_token)

        topk = 10
        probs, indices = torch.topk(final_probabilities, k=topk)

        print("\nTop 10 predictions:")
        for i in range(topk):
            token_id = indices[i].item()
            token = self.tokenizer.decode_word(token_id)
            prob = probs[i].item()
            print(f"{token:15s}  {prob:.4f}")


if __name__ == '__main__':
    Main = Main()
    Main.main()
