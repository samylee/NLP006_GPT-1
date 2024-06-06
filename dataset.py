import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, data, characters, block_size):
        super(ShakespeareDataset, self).__init__()
        # char to idx mapping and vice-versa
        self.stoi = {ch: i for i, ch in enumerate(characters)}
        self.itos = {i: ch for i, ch in enumerate(characters)}

        self.block_size = block_size
        self.data = data

    def __getitem__(self, idx):
        # take a chunk of data from the given index from the dataset
        chunk = self.data[idx: idx + self.block_size + 1]

        # convert the chunk to integers
        data = [self.stoi[ch] for ch in chunk]

        x = torch.tensor(data[:-1], dtype=torch.long)
        y = torch.tensor(data[1:], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data) - self.block_size


def load_datasets(data_path, block_size):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read(-1)
    characters = sorted(list(set(text)))
    data_size, vocab_size = len(text), len(characters)
    print(f"Dataset has {data_size} characters. {vocab_size} of characters are unique.")

    dataset = ShakespeareDataset(text, characters, block_size)
    return dataset, vocab_size