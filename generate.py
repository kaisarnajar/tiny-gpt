import sys

import torch

from train import load_model


def generate(model, bpe, prompt, temperature=1.0, top_k=0, top_p=0.9, max_new_tokens=200):
    ids = bpe.encode(prompt)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(torch.tensor([ids[-model.context_len:]]))[0, -1]

        if temperature == 0:
            next_id = logits.argmax().item()
        else:
            logits = logits / temperature
            if top_k > 0:
                kth_best = logits.topk(top_k).values[-1]
                logits[logits < kth_best] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, order = probs.sort(descending=True)
                remove = sorted_probs.cumsum(0) - sorted_probs >= top_p
                probs[order[remove]] = 0
                probs = probs / probs.sum()
            next_id = torch.multinomial(probs, 1).item()

        if next_id == bpe.eos_id:
            break
        ids.append(next_id)
    return bpe.decode(ids)


if __name__ == "__main__":
    model, bpe = load_model()
    prompts = sys.argv[1:] or ["Once upon a time", "The little dog"]
    for prompt in prompts:
        print(generate(model, bpe, prompt), "\n")
