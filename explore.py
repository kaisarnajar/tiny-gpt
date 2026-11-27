import math
import sys

import torch
import torch.nn as nn

import train
from generate import generate

PROMPTS = ["Once upon a time", "Lily found a small box under her bed.", "Tom was sad because"]


def settings(model, bpe):
    n_heads = model.blocks[0].attn.n_heads
    d_model = model.token_emb.weight.shape[1]
    print("Model")
    print(f"  parameters      {sum(p.numel() for p in model.parameters()):,}")
    print(f"  vocabulary      {bpe.vocab_size:,} tokens")
    print(f"  context length  {model.context_len} tokens")
    print(f"  d_model         {d_model}")
    print(f"  heads           {n_heads} × {d_model // n_heads}")
    print(f"  blocks          {len(model.blocks)}")


def validation(model, bpe, batches=100):
    torch.manual_seed(0)
    data = train.tokenize(bpe, "val")
    total = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = train.get_batch(data)
            total += nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    loss = total / batches
    print("\nValidation (same 100 batches every time, so models can be compared)")
    print(f"  loss        {loss:.4f}")
    print(f"  perplexity  {math.exp(loss):.1f}   (the model is as unsure as choosing among this many tokens)")


def stories(model, bpe):
    print("\nStories (temperature 0: the same text every time)")
    for prompt in PROMPTS:
        print(f"\n  {generate(model, bpe, prompt, temperature=0, max_new_tokens=80)}")


def next_tokens(model, bpe, text, k=8):
    with torch.no_grad():
        logits = model(torch.tensor([bpe.encode(text)[-model.context_len:]]))[0, -1]
    probs, ids = torch.softmax(logits, dim=-1).topk(k)
    print(f"\nNext token after {text!r}")
    for p, i in zip(probs.tolist(), ids.tolist()):
        print(f"  {bpe.decode([i])!r:14} {p:.3f}  {'█' * round(p * 40)}")


if __name__ == "__main__":
    model, bpe = train.load_model()
    train.DEVICE = "cpu"
    if len(sys.argv) > 1:
        next_tokens(model, bpe, " ".join(sys.argv[1:]))
    else:
        settings(model, bpe)
        validation(model, bpe)
        stories(model, bpe)
        next_tokens(model, bpe, "Lily was very")
