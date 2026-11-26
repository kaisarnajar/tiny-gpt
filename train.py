"""Train the Step 4b model (embeddings + multi-head attention + residual) with AdamW and cross-entropy.

Each sentence is turned into next-word questions:
    tokens : the  cat  eats  fish  .  <eos> <pad> <pad>      (padded to CONTEXT_LEN)
    input  : the  cat  eats  fish  .  <eos> <pad>            tokens[:-1]
    target : cat  eats fish  .   <eos> <pad> <pad>           tokens[1:]
Positions whose target is <pad> are ignored by the loss.
"""

import math
import random
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn

from model import GPT, count_parameters
from tokenizer import Tokenizer

CORPUS_PATH = "data/corpus.txt"
MODEL_PATH = "model.pt"

D_MODEL = 64
N_HEADS = 4  # attention heads: 4 × 16 numbers (1 = the Step 3 model)
RESIDUAL = True  # x = x + attention(x) instead of x = attention(x) (False = Step 4a)
CONTEXT_LEN = 8
EPOCHS = 100
BATCH_SIZE = 64
LR = 3e-3
WEIGHT_DECAY = 0.0
VAL_FRACTION = 0.1
LOG_EVERY = 10
SEED = 42


def load_data(tokenizer: Tokenizer):
    """Return (X_train, Y_train, X_val, Y_val), each of shape (N, CONTEXT_LEN - 1)."""
    sentences = open(CORPUS_PATH).read().splitlines()
    random.Random(SEED).shuffle(sentences)

    rows = []
    for s in sentences:
        ids = tokenizer.encode(s, add_eos=True)
        assert len(ids) <= CONTEXT_LEN, f"Too long for context_len: {s!r}"
        rows.append(ids + [tokenizer.pad_id] * (CONTEXT_LEN - len(ids)))
    data = torch.tensor(rows)  # (N, CONTEXT_LEN)

    X, Y = data[:, :-1], data[:, 1:]  # input / target, shifted by one
    n_val = int(len(data) * VAL_FRACTION)
    return X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]


def best_possible_loss(X: torch.Tensor, Y: torch.Tensor, pad_id: int, use_pos: bool = True,
                       prefix: bool = False) -> float:
    """Lowest loss ANY model can reach on this data, given what it is allowed to see.

    prefix=False: only (current token, position)  -> the limit of the Step 2 model
    prefix=True : all words so far                -> the limit of a model with attention

    This is the uncertainty left in the data: e.g. after "the" at position 0 there
    are many valid subjects, so even a perfect model can't get the loss to 0.
    """
    counts = defaultdict(Counter)
    for x_row, y_row in zip(X.tolist(), Y.tolist()):
        for pos, (x, y) in enumerate(zip(x_row, y_row)):
            if y != pad_id:
                key = tuple(x_row[: pos + 1]) if prefix else (x, pos) if use_pos else x
                counts[key][y] += 1
    total = sum(sum(c.values()) for c in counts.values())
    loss = 0.0
    for c in counts.values():
        n = sum(c.values())
        loss -= sum(k * math.log(k / n) for k in c.values())
    return loss / total


@torch.no_grad()
def evaluate(model: GPT, X: torch.Tensor, Y: torch.Tensor, loss_fn) -> float:
    model.eval()
    logits = model(X)
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))
    model.train()
    return loss.item()


def train(use_pos: bool = True, use_attn: bool = True, n_heads: int = N_HEADS, residual: bool = RESIDUAL,
          epochs: int = EPOCHS, verbose: bool = True, save_path: str | None = MODEL_PATH):
    """Train a model and return (model, tokenizer, info)."""
    torch.manual_seed(SEED)
    tokenizer = Tokenizer.from_corpus(CORPUS_PATH)
    X_train, Y_train, X_val, Y_val = load_data(tokenizer)
    V = tokenizer.vocab_size

    model = GPT(V, d_model=D_MODEL, context_len=CONTEXT_LEN, use_pos=use_pos, use_attn=use_attn, n_heads=n_heads,
                residual=residual)
    # Takes RAW logits: it applies log-softmax internally. Don't softmax before it.
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_no_attn = best_possible_loss(X_train, Y_train, tokenizer.pad_id, use_pos)
    best_prefix = best_possible_loss(X_train, Y_train, tokenizer.pad_id, prefix=True)
    best = best_prefix if use_attn else best_no_attn
    if verbose:
        print(f"Data      : {len(X_train):,} train / {len(X_val):,} val sentences, vocab {V}")
        print("Parameters: " + ", ".join(f"{k} {v:,}" for k, v in count_parameters(model).items()))
        if use_attn:
            print(f"Attention : {n_heads} heads × {D_MODEL // n_heads} numbers, "
                  f"{'x = x + attention(x) (residual)' if residual else 'x = attention(x)'}")
        print(f"Steps     : {math.ceil(len(X_train) / BATCH_SIZE)} per epoch x {epochs} epochs")
        print(f"Uniform-guess loss ln({V}) = {math.log(V):.3f}")
        print(f"Best possible loss: {best_no_attn:.3f} seeing only (word, position) [Step 2] | "
              f"{best_prefix:.3f} seeing all words so far [attention]\n")
        print(f"start        | val loss {evaluate(model, X_val, Y_val, loss_fn):.4f}")

    start = time.time()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X_train))  # new batch order every epoch
        total, n_batches = 0.0, 0
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb, yb = X_train[idx], Y_train[idx]  # (B, 7), (B, 7)

            logits = model(xb)  # 1. forward      (B, 7, V)
            loss = loss_fn(logits.reshape(-1, V), yb.reshape(-1))  # 2. loss
            optimizer.zero_grad()
            loss.backward()  # 3. backward
            optimizer.step()  # 4. update

            total += loss.item()
            n_batches += 1

        if verbose and (epoch == 1 or epoch % LOG_EVERY == 0 or epoch == epochs):
            val = evaluate(model, X_val, Y_val, loss_fn)
            print(f"epoch {epoch:3d}/{epochs} | train loss {total / n_batches:.4f} | val loss {val:.4f}")

    info = {"val_loss": evaluate(model, X_val, Y_val, loss_fn), "best_possible": best}
    if verbose:
        print(f"\nDone in {time.time() - start:.0f}s. Final val loss {info['val_loss']:.4f} "
              f"(best possible {best:.4f})")

    if save_path:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {"vocab_size": V, "d_model": D_MODEL, "context_len": CONTEXT_LEN, "use_pos": use_pos,
                           "use_attn": use_attn, "n_heads": n_heads,
                           "residual": residual},
                "vocab": tokenizer.itos,
            },
            save_path,
        )
        if verbose:
            print(f"Saved model to {save_path}")
    return model, tokenizer, info


def load_model(path: str = MODEL_PATH) -> tuple[GPT, Tokenizer]:
    checkpoint = torch.load(path)
    model = GPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, Tokenizer(checkpoint["vocab"])


if __name__ == "__main__":
    train()
