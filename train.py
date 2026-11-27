import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from bpe import BPE, EOS
from model import GPT

CONTEXT_LEN = 256
D_MODEL = 384
N_HEADS = 6
N_LAYERS = 6
BATCH_SIZE = 32
STEPS = 10000
LR = 1e-3
WARMUP = 200
EVAL_EVERY = 500
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def tokenize(bpe, name):
    path = f"data/{name}.npy"
    if not os.path.exists(path):
        text = open(f"data/{name}.txt").read()
        parts, start = [], 0
        while start < len(text):
            end = text.find(EOS, start + 10_000_000)
            end = len(text) if end == -1 else end
            parts.append(np.array(bpe.encode(text[start:end]), dtype=np.int16))
            start = end
        np.save(path, np.concatenate(parts))
    return torch.from_numpy(np.load(path))


def get_batch(data):
    starts = torch.randint(len(data) - CONTEXT_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + CONTEXT_LEN] for i in starts]).long()
    y = torch.stack([data[i + 1:i + CONTEXT_LEN + 1] for i in starts]).long()
    return x.to(DEVICE), y.to(DEVICE)


def learning_rate(step):
    if step < WARMUP:
        return LR * step / WARMUP
    progress = (step - WARMUP) / (STEPS - WARMUP)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def val_loss(model, data, loss_fn, batches=40):
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
    train_data, val_data = tokenize(bpe, "train"), tokenize(bpe, "val")
    model = GPT(bpe.vocab_size, D_MODEL, CONTEXT_LEN, N_HEADS, N_LAYERS).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"tokens: {len(train_data):,} train, {len(val_data):,} val, device: {DEVICE}")

    best, losses, start = float("inf"), [], time.time()
    for step in range(1, STEPS + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        x, y = get_batch(train_data)
        loss = loss_fn(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % EVAL_EVERY == 0:
            val = val_loss(model, val_data, loss_fn)
            if val < best:
                best = val
                torch.save(model.state_dict(), "model.pt")
            minutes = (time.time() - start) / 60
            print(f"step {step}  train loss {sum(losses) / len(losses):.4f}  val loss {val:.4f}  "
                  f"lr {learning_rate(step):.5f}  {minutes:.1f} min", flush=True)
            losses = []


def load_model():
    bpe = BPE.load("bpe.json")
    model = GPT(bpe.vocab_size, D_MODEL, CONTEXT_LEN, N_HEADS, N_LAYERS)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model, bpe


if __name__ == "__main__":
    train()
