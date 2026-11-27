import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train import load_model

model, bpe = load_model()


def next_tokens(prompt, k=5):
    with torch.no_grad():
        logits = model(torch.tensor([bpe.encode(prompt)]))[0, -1]
    probs, ids = torch.softmax(logits, dim=-1).topk(k)
    return ", ".join(f"{bpe.decode([i])!r} {p:.2f}" for p, i in zip(probs.tolist(), ids.tolist()))


print("Parameters")
parts = {}
for name, p in model.named_parameters():
    part = ".".join(name.split(".")[:2]) if name.startswith("blocks") else name.split(".")[0]
    parts[part] = parts.get(part, 0) + p.numel()
for part, n in parts.items():
    print(f"  {part:12} {n:>10,}")
print(f"  {'total':12} {sum(parts.values()):>10,}")

print("\nNext token")
for prompt in ["Once upon a time, there was a", "Lily was very", "The dog ran to the", "Tom and Sam were best"]:
    print(f"  {prompt:32} -> {next_tokens(prompt)}")

text = "Lily had a red ball. She threw the ball to her"
ids = bpe.encode(text)
tokens = [bpe.decode([i]) for i in ids]
with torch.no_grad():
    model(torch.tensor([ids]))
n_heads = len(model.blocks[0].attn.weights[0])
fig, axes = plt.subplots(len(model.blocks), n_heads, figsize=(3.5 * n_heads, 3.5 * len(model.blocks)), squeeze=False)
print(f"\nAttention for {text!r} (saved to attention.png): what the last token {tokens[-1]!r} looks at most")
for b, block in enumerate(model.blocks):
    weights = block.attn.weights[0]
    best = []
    for h in range(n_heads):
        ax = axes[b][h]
        ax.imshow(weights[h].numpy(), cmap="Purples", vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tokens)), tokens, fontsize=6)
        ax.set_title(f"block {b + 1}, head {h + 1}", fontsize=8)
        j = weights[h, -1].argmax().item()
        best.append(f"head {h + 1}: {tokens[j]!r} {weights[h, -1, j]:.2f}")
    print(f"  block {b + 1}: " + " | ".join(best))
fig.tight_layout()
fig.savefig("attention.png")
