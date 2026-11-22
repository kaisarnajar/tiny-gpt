"""Predict the next word: "He" -> "He boy"."""

import sys

import torch

from model import GPT
from tokenizer import Tokenizer


def predict_next(model: GPT, tokenizer: Tokenizer, text: str) -> str:
    last_id = tokenizer.encode(text)[-1]
    with torch.no_grad():
        y = model(torch.tensor([[float(last_id)]])).item()
    next_id = min(max(round(y), 0), tokenizer.vocab_size - 1)  # nearest valid ID
    return f"{text} {tokenizer.decode([next_id])}"


def main():
    tokenizer = Tokenizer()
    model = GPT()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    inputs = sys.argv[1:] or ["He", "She"]
    for text in inputs:
        print(f"{text!r} -> {predict_next(model, tokenizer, text)!r}")


if __name__ == "__main__":
    main()
