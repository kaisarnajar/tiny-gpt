import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from train import load_model

model, tokenizer = load_model()


def next_words(prompt, k=4):
    with torch.no_grad():
        logits = model(torch.tensor([tokenizer.encode(prompt)]))[0, -1]
    probs, ids = torch.softmax(logits, dim=-1).topk(k)
    return ", ".join(f"{tokenizer.itos[i]} {p:.2f}" for p, i in zip(probs.tolist(), ids.tolist()))


print("Parameters")
for name, p in model.named_parameters():
    print(f"  {name:20} {str(tuple(p.shape)):10} {p.numel():>6,}")
print(f"  {'total':31} {sum(p.numel() for p in model.parameters()):>6,}")

print("\nNext word")
for prompt in ["he is a", "she is a", "the cow lives on the", "the monkey lives in the"]:
    print(f"  {prompt:24} -> {next_words(prompt)}")

print("\nSimilar words")
emb = F.normalize(model.token_emb.weight.detach(), dim=1)
for word in ["cat", "lion", "grass", "mother", "barks"]:
    similarity = emb @ emb[tokenizer.stoi[word]]
    best = similarity.topk(6).indices[1:]
    print(f"  {word:8} -> {', '.join(tokenizer.itos[i] for i in best)}")

print("\nAttention for 'he is a doctor .' (saved to attention.png)")
words = "he is a doctor .".split()
with torch.no_grad():
    model(torch.tensor([tokenizer.encode(" ".join(words))]))
weights = model.attn.weights[0]
fig, axes = plt.subplots(1, len(weights), figsize=(4 * len(weights), 4))
for h, ax in enumerate(axes):
    ax.imshow(weights[h].numpy(), cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(len(words)), words)
    ax.set_yticks(range(len(words)), words)
    ax.set_title(f"head {h + 1}")
    looks_at = ", ".join(f"{words[j]} {weights[h, 2, j]:.2f}" for j in range(3))
    print(f"  head {h + 1}: 'a' looks at {looks_at}")
fig.tight_layout()
fig.savefig("attention.png")
