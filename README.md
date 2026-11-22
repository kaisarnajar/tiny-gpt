# tiny-gpt — LLM From Scratch

Building a GPT step by step.

## Step 1: the dummy model

```
text --Tokenizer--> ID --GPT (Linear: y = w·x + b)--> ID --Tokenizer--> text
"He" -> 0 -> 1·0 + 2 = 2 -> "boy"
```

- `tokenizer.py`: fixed vocab `{He: 0, She: 1, boy: 2, girl: 3}`, no weights
- `model.py`: `GPT` = one `nn.Linear(1, 1)`, w and b start at 0
- `train.py`: AdamW + MSE loss over epochs, logs loss, w, b
- `generate.py`: predicts the next word

```bash
pip install -r requirements.txt
python train.py      # learns w ≈ 1, b ≈ 2
python generate.py   # He -> He boy, She -> She girl
```

Next steps: embeddings, cross-entropy over the vocab, attention, feed-forward.
