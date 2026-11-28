import math
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

from bpe import BPE, EOS
from model import GPT

CONTEXT_LEN = 512
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 8
BATCH_SIZE = 16
STEPS = 60000
LR = 6e-4
WARMUP = 1000
EVAL_EVERY = 1000
SOURCES = ["stories", "web", "code"]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def forward_loss(model, loss_fn, x, y):
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "mps"):
        return loss_fn(model(x).flatten(0, 1).float(), y.flatten())


def encode_piece(text):
    return np.array(BPE.load("bpe.json").encode(text), dtype=np.int16)


def tokenize(name):
    path = f"data/{name}.npy"
    if not os.path.exists(path):
        text = open(f"data/{name}.txt").read()
        pieces, start = [], 0
        while start < len(text):
            end = text.find(EOS, start + 10_000_000)
            end = len(text) if end == -1 else end
            pieces.append(text[start:end])
            start = end
        with Pool() as pool:
            np.save(path, np.concatenate(pool.map(encode_piece, pieces)))
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
def val_losses(model, val, loss_fn, batches=20):
    model.eval()
    losses = {}
    for name, data in val.items():
        total = 0
        for _ in range(batches):
            x, y = get_batch(data)
            total += forward_loss(model, loss_fn, x, y).item()
        losses[name] = total / batches
    model.train()
    return losses


def train():
    torch.manual_seed(42)
    bpe = BPE.load("bpe.json")
    train_data = tokenize("train")
    val = {name: tokenize(f"val_{name}") for name in SOURCES}
    model = GPT(bpe.vocab_size, D_MODEL, CONTEXT_LEN, N_HEADS, N_LAYERS).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    first_step, best = 1, float("inf")
    if os.path.exists("checkpoint.pt"):
        checkpoint = torch.load("checkpoint.pt", map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_step, best = checkpoint["step"] + 1, checkpoint["best"]
        print(f"resuming from step {checkpoint['step']}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"tokens: {len(train_data):,} train, device: {DEVICE}")

    losses, start = [], time.time()
    for step in range(first_step, STEPS + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        x, y = get_batch(train_data)
        loss = forward_loss(model, loss_fn, x, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % EVAL_EVERY == 0:
            val_loss = val_losses(model, val, loss_fn)
            mean = sum(val_loss.values()) / len(val_loss)
            if mean < best:
                best = mean
                torch.save(model.state_dict(), "model.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "best": best}, "checkpoint.pt")
            hours = (time.time() - start) / 3600
            per_source = "  ".join(f"{name} {v:.3f}" for name, v in val_loss.items())
            print(f"step {step}  train {sum(losses) / len(losses):.4f}  val {mean:.4f} ({per_source})  "
                  f"lr {learning_rate(step):.6f}  {hours:.2f} h", flush=True)
            losses = []


def load_model():
    bpe = BPE.load("bpe.json")
    model = GPT(bpe.vocab_size, D_MODEL, CONTEXT_LEN, N_HEADS, N_LAYERS)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model, bpe


if __name__ == "__main__":
    train()
