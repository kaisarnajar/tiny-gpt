"""Test the trained model: continue a prompt one word at a time.

    python generate.py                                  # a few default prompts, sampled
    python generate.py "he is a" "she is"               # your own prompts
    python generate.py "he is a" --temperature 0        # greedy (always the same answer)
    python generate.py "he is a" --top-k 3 --top-p 1    # only top-k
    python generate.py "he is a" --samples 5            # 5 different completions

At each step:
    logits for the LAST position
    -> / temperature       (T < 1 sharper, T > 1 flatter, T = 0 greedy)
    -> top-k filter        (keep the k most likely words; 0 = off)
    -> top-p filter        (keep the smallest set whose probabilities add up to p; 1.0 = off)
    -> softmax -> sample one word -> append it
    -> stop at <eos> or context_len tokens.
"""

import argparse

import torch

from train import load_model

SHOW_TOP = 3  # how many candidates to print per step (display only)

# Defaults used by the CLI and the web UI.
TEMPERATURE = 1.0
TOP_K = 0  # 0 = off
TOP_P = 0.9  # 1.0 = off


def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Set the logits of words we must not pick to -inf (softmax turns -inf into probability 0)."""
    logits = logits.clone()

    # Top-k: everything below the k-th highest score is removed.
    if top_k > 0:
        kth_best = logits.topk(min(top_k, logits.numel())).values[-1]
        logits[logits < kth_best] = float("-inf")

    # Top-p (nucleus): sort by probability, keep words until the running total reaches p.
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Remove a word if the words BEFORE it already reach p (so the word that crosses p stays,
        # and the most likely word always stays).
        remove = (cumulative - torch.softmax(sorted_logits, dim=-1)) >= top_p
        logits[sorted_idx[remove]] = float("-inf")

    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    verbose: bool = True,
) -> str:
    ids = tokenizer.encode(prompt)
    while len(ids) < model.context_len:
        logits = model(torch.tensor([ids]))[0, -1]  # (V,) scores for the next word

        if temperature == 0:  # greedy: always the most likely word
            probs = torch.softmax(logits, dim=-1)
            next_id = probs.argmax().item()
        else:
            logits = filter_logits(logits / temperature, top_k, top_p)
            probs = torch.softmax(logits, dim=-1)  # (V,) filtered words get probability 0
            next_id = torch.multinomial(probs, num_samples=1).item()  # random pick, weighted by probs

        if verbose:
            top_p_vals, top_i = probs.topk(SHOW_TOP)
            options = " | ".join(
                f"{tokenizer.itos[i]} {p:.2f}" for p, i in zip(top_p_vals.tolist(), top_i.tolist()) if p > 0
            )
            kept = int((probs > 0).sum())
            print(f"  pos {len(ids) - 1} {tokenizer.itos[ids[-1]]!r:>10} -> {options}"
                  f"  ({kept} candidates) picked {tokenizer.itos[next_id]!r}")

        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts", nargs="*", default=["the", "the dog", "the cow eats", "the owl", "my mother", "he is"])
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="0 = off")
    parser.add_argument("--top-p", type=float, default=TOP_P, help="1.0 = off")
    parser.add_argument("--samples", type=int, default=1, help="completions per prompt")
    parser.add_argument("--seed", type=int, default=None, help="fix the randomness")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer = load_model()
    print(f"temperature={args.temperature} top_k={args.top_k} top_p={args.top_p}\n")
    for prompt in args.prompts:
        print(f"prompt: {prompt!r}")
        for _ in range(args.samples):
            sentence = generate(model, tokenizer, prompt, args.temperature, args.top_k, args.top_p,
                                verbose=args.samples == 1)
            print(f"=> {sentence}")
        print()


if __name__ == "__main__":
    main()
