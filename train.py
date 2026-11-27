import os

import torch
import torch.nn as nn

from bpe import BPE
from model import GPT

CONTEXT_LEN = 128
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
BATCH_SIZE = 32
STEPS = 5000
LR = 1e-3
EVAL_EVERY = 250
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_tokens(bpe):
    if not os.path.exists("data/tokens.pt"):
        ids = bpe.encode(open("data/stories.txt").read())
        torch.save(torch.tensor(ids), "data/tokens.pt")
    data = torch.load("data/tokens.pt")
    n_val = len(data) // 10
    return data[n_val:], data[:n_val]


def get_batch(data):
    starts = torch.randint(len(data) - CONTEXT_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + CONTEXT_LEN] for i in starts])
    y = torch.stack([data[i + 1:i + CONTEXT_LEN + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def val_loss(model, data, loss_fn, batches=20):
    model.eval()
    total = 0
    for _ in range(batches):
        x, y = get_batch(data)
        total += loss_fn(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return total / batches


def train():
    torch.manual_seed(42)
    bpe = BPE.load("bpe.json")
    train_data, val_data = load_tokens(bpe)
    model = GPT(bpe.vocab_size, D_MODEL, CONTEXT_LEN, N_HEADS, N_LAYERS).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"tokens: {len(train_data):,} train, {len(val_data):,} val, device: {DEVICE}")

    losses = []
    for step in range(1, STEPS + 1):
        x, y = get_batch(train_data)
        loss = loss_fn(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % EVAL_EVERY == 0:
            train_loss = sum(losses) / len(losses)
            losses = []
            print(f"step {step}  train loss {train_loss:.4f}  val loss {val_loss(model, val_data, loss_fn):.4f}")

    torch.save(model.state_dict(), "model.pt")


def load_model():
    bpe = BPE.load("bpe.json")
    model = GPT(bpe.vocab_size, D_MODEL, CONTEXT_LEN, N_HEADS, N_LAYERS)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model, bpe


if __name__ == "__main__":
    train()
