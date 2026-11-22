"""Test the trained model: continue a prompt one word at a time.

    python generate.py                       # a few default prompts
    python generate.py "the dog" "she is"    # your own prompts

At each step: logits for the LAST position -> softmax -> show the top 3 words
-> append the most likely one (greedy) -> stop at <eos> or context_len tokens.
"""

import sys

import torch

from train import load_model

TOP_K = 3


@torch.no_grad()
def generate(model, tokenizer, prompt: str, verbose: bool = True) -> str:
    ids = tokenizer.encode(prompt)
    while len(ids) < model.context_len:
        logits = model(torch.tensor([ids]))[0, -1]  # (V,) scores for the next word
        probs = torch.softmax(logits, dim=-1)  # (V,) probabilities, sum to 1
        top_p, top_i = probs.topk(TOP_K)
        if verbose:
            options = " | ".join(f"{tokenizer.itos[i]} {p:.2f}" for p, i in zip(top_p.tolist(), top_i.tolist()))
            print(f"  pos {len(ids) - 1} {tokenizer.itos[ids[-1]]!r:>10} -> {options}")
        next_id = top_i[0].item()  # greedy: most likely word
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ids)


def main():
    model, tokenizer = load_model()
    prompts = sys.argv[1:] or ["the", "the dog", "the cow eats", "the owl", "my mother", "he is"]
    for prompt in prompts:
        print(f"prompt: {prompt!r}")
        print(f"=> {generate(model, tokenizer, prompt)}\n")


if __name__ == "__main__":
    main()
