import sys

import torch

from train import load_model


def generate(model, tokenizer, prompt, temperature=1.0, top_k=0, top_p=0.9):
    ids = tokenizer.encode(prompt)
    while len(ids) < model.context_len:
        with torch.no_grad():
            logits = model(torch.tensor([ids]))[0, -1]

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

        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ids)


if __name__ == "__main__":
    model, tokenizer = load_model()
    prompts = sys.argv[1:] or ["the dog", "the cow eats", "he is a", "she is a"]
    for prompt in prompts:
        print(generate(model, tokenizer, prompt))
