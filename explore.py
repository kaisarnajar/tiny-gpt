import math
import sys

import torch

import train
from generate import generate

PROMPTS = ["Once upon a time", "The water cycle is", "def is_prime(n):"]


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


@torch.no_grad()
def validation(model, batches=50):
    model.to(train.DEVICE)
    print("\nValidation (the same batches every time, so models can be compared)")
    print(f"  {'source':8} {'loss':>6} {'perplexity':>11} {'top-1 correct':>14}")
    for name in train.SOURCES:
        torch.manual_seed(0)
        data = train.tokenize(f"val_{name}")
        loss, correct = 0, 0
        for _ in range(batches):
            x, y = train.get_batch(data)
            logits = model(x).float()
            loss += torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
            correct += (logits.argmax(-1) == y).float().mean().item()
        loss, correct = loss / batches, correct / batches
        print(f"  {name:8} {loss:6.3f} {math.exp(loss):11.1f} {correct:13.1%}")
    model.to("cpu")


def samples(model, bpe):
    print("\nSamples (temperature 0: the same text every time)")
    for prompt in PROMPTS:
        print(f"\n{generate(model, bpe, prompt, temperature=0, max_new_tokens=80)}")


def next_tokens(model, bpe, text, k=8):
    with torch.no_grad():
        logits = model(torch.tensor([bpe.encode(text)[-model.context_len:]]))[0, -1]
    probs, ids = torch.softmax(logits, dim=-1).topk(k)
    print(f"\nNext token after {text!r}")
    for p, i in zip(probs.tolist(), ids.tolist()):
        print(f"  {bpe.decode([i])!r:14} {p:.3f}  {'█' * round(p * 40)}")


if __name__ == "__main__":
    model, bpe = train.load_model()
    if len(sys.argv) > 1:
        next_tokens(model, bpe, " ".join(sys.argv[1:]))
    else:
        settings(model, bpe)
        validation(model)
        samples(model, bpe)
        next_tokens(model, bpe, "Lily was very")
        next_tokens(model, bpe, "for i in")
