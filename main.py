from Embedding import Tokenizer, Embedding
import torch
from Transformer import SelfAttention, MLP
from Predictions import Unembed
from config import num_layers


class Main:

    def __init__(self):
        self.W_e = []
        self.num_layers = num_layers
        self.tokenizer = Tokenizer()
        self.embedding = Embedding()
        self.perceptron = MLP()
        self.unembed = Unembed()

    def main(self):
        tokens, ids = self.tokenizer.encode_prompt()
        # print(f'tokens: {tokens} id: {id}')

        for id in ids:
            vec = self.embedding.get_embedding_vector(id)
            vec = torch.tensor(vec, dtype=torch.float32)  # convert NumPy → Tensor
            self.W_e.append(vec)

        self.W_e = torch.stack(self.W_e)  # Turn into tensor with shape (seq_len, embedding_dim)
        print(f'number of embedding vectors in weights_e: {len(self.W_e)}')
        print(f'W_e shape: {self.W_e.shape}')

        # ---------------------------------------------------------------
        # Actual training loop Begins here
        # ---------------------------------------------------------------
        x = self.W_e # Running variable
        for _ in range(self.num_layers):
            # attention = SelfAttention(self.W_e)
            # attn_out = attention.attention()
            # attn_res = self.W_e + attn_out
            #
            # mlp_out = self.perceptron.forward(attn_res)
            # mlp_res = self.W_e + mlp_out

            attention = SelfAttention(x)
            attn_out = attention.attention()
            x = x + attn_out  # residual connection

            mlp_out = self.perceptron.forward(x)
            x = x + mlp_out  # residual connection


        # ---------------------------------------------------------------
        # Training loop ends here
        # ---------------------------------------------------------------
        print(f'mlp.res shape: {x.shape}')

        # What we will make predicitons on
        final_vec = x[-1]
        print(f'final vector shape: {final_vec.shape}')
        logits = self.unembed.unembed(final_vec)
        print(f'logits shape: {logits.shape}')
        # Want probs to sum to 1
        probabilities = torch.softmax(logits, dim=-1)
        print(f'probs shape: {probabilities.shape}')

        next_id = torch.multinomial(probabilities, num_samples=1).item()

        next_token = self.tokenizer.decode_word(next_id)

        print("next token:", next_token)


if __name__ == '__main__':
    Main = Main()
    Main.main()
