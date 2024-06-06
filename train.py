import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

from models import GPT_1
from dataset import load_datasets


def train(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    losses = torch.zeros(len(train_loader))
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        logits = logits.reshape(-1, logits.shape[-1])

        loss = criterion(logits, y.view(-1))
        losses[i] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 == 0:
            print("epoch: {}, iter: {}/{}, train loss: {:.2f}".format(epoch + 1, i, len(train_loader), loss.item()))

    return torch.mean(losses)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = 'data/input.txt'

    batch_size = 256
    epochs = 2
    lr = 1e-4

    block_size = 128
    n_embed = 768
    n_heads = 12
    n_layers = 2
    dropout_ratio = 0.1

    checkpoint_dir = 'weights/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # load dataset
    dataset, vocab_size = load_datasets(data_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load model
    model = GPT_1(vocab_size, block_size, n_embed, n_heads, n_layers, dropout_ratio)
    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # training loop
    for epoch in range(epochs):
        train_loss = train(dataloader, model, criterion, optimizer, device, epoch)
        print("epoch: {}, train avg loss: {:.2f}".format(epoch + 1, train_loss))
        torch.save(model.state_dict(), checkpoint_dir + "epoch_{}_loss_{:.2f}.pt".format(epoch + 1, train_loss))


if __name__ == "__main__":
    main()
