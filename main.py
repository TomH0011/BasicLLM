from Embedding import Tokenizer, Embedding
import torch
from Transformer import SelfAttention, MLP
from Predictions import Unembed


def main():
    tokenizer = Tokenizer('.')
    tokens, ids = tokenizer.encode_prompt()
    # print(f'tokens: {tokens} id: {id}')

    embedding = Embedding()
    W_e = []
    for id in ids:
        vec = embedding.get_embedding_vector(id)
        vec = torch.tensor(vec, dtype=torch.float32)  # convert NumPy → Tensor
        W_e.append(vec)

    W_e = torch.stack(W_e)  # Turn into tensor with shape (seq_len, embedding_dim)
    print(f'number of embedding vectors in weights_e: {len(W_e)}')
    print(f'W_e shape: {W_e.shape}')

    attention = SelfAttention(W_e)
    attn_out = attention.attention()
    attn_res = W_e + attn_out

    perceptron = MLP()
    mlp_out = perceptron.forward(attn_res)
    mlp_res = W_e + mlp_out

    print(f'mlp.res shape: {mlp_res.shape}')

    # What we will make predicitons on
    unembed = Unembed()
    final_vec = mlp_res[-1]
    print(f'final vector shape: {final_vec.shape}')
    logits = unembed.unembed(final_vec)
    print(f'logits shape: {logits.shape}')
    # Want probs to sum to 1
    probabilities = torch.softmax(logits, dim=-1)
    print(f'probs shape: {probabilities.shape}')

    next_id = torch.multinomial(probabilities, num_samples=1).item()

    next_token = tokenizer.decode_word(next_id)

    print("next token:", next_token)




if __name__ == '__main__':
    main()
