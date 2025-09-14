from transformers import AutoTokenizer

model_name = "gpt2"  # Only a small model https://huggingface.co/models?num_parameters=min:0,max:1B&sort=trending

embedding_dim = 128     # Self-explanatory
d_head = 64             # size of each attention head
attn_heads = 2          # Number of attention heads
hidden_dimension = 512  # Hidden dimension for when up-scaling inside the MLP
num_layers = 16          # number of times the loop goes Attention -> MLP -> Attention

tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = tokenizer.vocab_size
text = 'The quick brown fox jumped over the lazy'