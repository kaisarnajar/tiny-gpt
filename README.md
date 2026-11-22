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

## Step 2 data: synthetic corpus

`data.py` generates `data/corpus.txt`: 50,000 lowercase sentences from templates (fixed seed, reproducible).

```
the {animal} {eats|likes|wants} the {food} .      the cat eats the fish .
the {animal} {chases|finds|sees} the {object} .   the dog chases the ball .
the {animal} is {adjective} .                     the lion is happy .
he is a {boy|man|king|prince} .                   she is a {girl|woman|queen|princess} .
```

- Same-category words share contexts, so their embeddings should cluster.
- "the" at position 0 is followed by an animal; at position 3 by a food/object, so position matters.
- 50 words, max 6 words per sentence (7 tokens with `<eos>`), fits `context_len = 8`.

```bash
python data.py           # 50,000 sentences
python data.py 100000    # custom size
```
