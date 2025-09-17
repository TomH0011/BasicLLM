import torch.cuda
from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "gpt2"  # Only a small model https://huggingface.co/models?num_parameters=min:0,max:1B&sort=trending
data_file_path = 'Sample_Text.txt'  # Where my data is stored
tokenized_data_path = 'tokenized_data.pt'  # Where ill save data too
embedding_dim = 128  # Self-explanatory
d_head = 64  # size of each attention head
attn_heads = 2  # Number of attention heads
hidden_dimension = 512  # Hidden dimension for when up-scaling inside the MLP
num_layers = 16  # number of times the loop goes Attention -> MLP -> Attention
learning_rate = 0.00001  # constant of proportionality in optimiser
momentum = 0.9  # momentum in optimiser


def delete_blank_lines(dirty_text):  # Very dirty text
    clean_text = []
    for char in dirty_text:
        if char != ' ' or char != chr(0x2588):
            clean_text.append(char)
        elif char == ' ':
            if clean_text[-1]:
                continue
            else:
                clean_text.append(char)
        else:
            continue
    return ''.join(clean_text)


try:
    with open(data_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.strip()
        text = delete_blank_lines(text)
except Exception as e:
    print(f'Data file {data_file_path} not found, or in invalid format')
    raise FileNotFoundError

tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = tokenizer.vocab_size
# text = text  # import your own data and set text to a string containing utf-8 text
