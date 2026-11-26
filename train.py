import random

import torch
import torch.nn as nn

from model import GPT
from tokenizer import Tokenizer

CONTEXT_LEN = 8
D_MODEL = 64
N_HEADS = 4
EPOCHS = 100
BATCH_SIZE = 64
LR = 3e-3


def load_data(tokenizer):
    sentences = open("data/corpus.txt").read().splitlines()
    random.Random(42).shuffle(sentences)
    rows = []
    for s in sentences:
        ids = tokenizer.encode(s) + [tokenizer.eos_id]
        ids += [tokenizer.pad_id] * (CONTEXT_LEN - len(ids))
        rows.append(ids)
    data = torch.tensor(rows)
    x, y = data[:, :-1], data[:, 1:]
    n_val = len(data) // 10
    return x[n_val:], y[n_val:], x[:n_val], y[:n_val]


def train():
    torch.manual_seed(42)
    tokenizer = Tokenizer.from_file("data/corpus.txt")
    x_train, y_train, x_val, y_val = load_data(tokenizer)
    model = GPT(tokenizer.vocab_size, D_MODEL, CONTEXT_LEN, N_HEADS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(x_train))
        total = 0
        for i in range(0, len(x_train), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            logits = model(x_train[idx])
            loss = loss_fn(logits.flatten(0, 1), y_train[idx].flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        if epoch % 10 == 0:
            with torch.no_grad():
                val_loss = loss_fn(model(x_val).flatten(0, 1), y_val.flatten()).item()
            train_loss = total / (len(x_train) / BATCH_SIZE)
            print(f"epoch {epoch}  train loss {train_loss:.4f}  val loss {val_loss:.4f}")

    torch.save({"model": model.state_dict(), "vocab": tokenizer.itos}, "model.pt")


def load_model():
    checkpoint = torch.load("model.pt")
    tokenizer = Tokenizer(checkpoint["vocab"])
    model = GPT(tokenizer.vocab_size, D_MODEL, CONTEXT_LEN, N_HEADS)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    train()
