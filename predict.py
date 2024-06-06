import torch
import torch.nn.functional as F

from models import GPT_1


def tok_k_logits(logits, k):
    v, ix = torch.topk(logits,k)
    out = logits.clone()
    out[out < v[:,[-1]]] = -float('inf')
    return out


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = 'data/input.txt'

    block_size = 128
    n_embed = 768
    n_heads = 12
    n_layers = 2
    dropout_ratio = 0.1
    topk = 10

    # load vocab
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read(-1)
    characters = sorted(list(set(text)))
    vocab_size = len(characters)
    stoi = {ch: i for i, ch in enumerate(characters)}

    # load model
    model = GPT_1(vocab_size, block_size, n_embed, n_heads, n_layers, dropout_ratio)
    model.load_state_dict(torch.load('weights/epoch_2_loss_0.84.pt', map_location='cpu'))
    model.eval()
    model.to(device)

    # input sentence
    input_sentence = 'Excuse me!'
    print(input_sentence, end='')
    indexes = [stoi[s] for s in input_sentence]
    # Start decoding
    for _ in range(512):
        input_token = torch.tensor([indexes]).to(device)
        input_token = input_token if input_token.size(1) <= block_size else input_token[:, -block_size:]
        # Decoder forward pass
        logits = model(input_token)
        logits = logits[:, -1, :]
        logits = tok_k_logits(logits, topk)
        # Forward to linear classify token in vocab and Softmax
        probs = F.softmax(logits, dim=-1)

        index = torch.multinomial(probs, num_samples=1)
        indexes.append(index.item())
        print(characters[index.item()], end='')

if __name__ == "__main__":
    main()