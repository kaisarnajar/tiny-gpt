# tiny-gpt — LLM From Scratch

Building a GPT step by step.

## Step 2: token + positional embeddings (current)

```
token IDs (B, T)
   ├─ token_emb = Embedding(225, 64)      "what the word is"
   └─ pos_emb   = Embedding(8, 64)        "where it is"
x      = token_emb + pos_emb              (B, T, 64)
logits = lm_head(x)  Linear(64, 225)      (B, T, 225)  one score per word
probs  = softmax(logits)                  probabilities, sum to 1
loss   = cross_entropy(logits, targets)   -log(prob of the true next word)
```

- `tokenizer.py`: vocabulary built from the corpus, `<pad>` = 0, `<eos>` = 1
- `model.py`: `GPT` (29,537 parameters), plus `softmax_manual` / `cross_entropy_manual`
- `train.py`: AdamW + cross-entropy, 100 epochs, batch 64, 90/10 train/val split; saves `model.pt`
- `generate.py`: continues a prompt word by word, showing the top-3 probabilities
- `explore.py`: shapes, softmax and cross-entropy walkthroughs, nearest words, PCA plot (`embeddings.png`), with vs without position, and the no-attention limitation

```bash
pip install -r requirements.txt
python data.py                  # data/corpus.txt (already committed)
python train.py                 # ~20s on CPU, val loss ≈ 1.27 (best possible ≈ 1.26)
python generate.py "the dog"    # the dog barks loudly .
python explore.py
```

No attention yet: each prediction sees only the current word and its position,
so "he is a" and "she is a" get the same next-word probabilities.

## Step 1: the dummy model (commit 50c8ef1)

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

`data.py` generates `data/corpus.txt`: 50,000 natural-sounding lowercase sentences (fixed seed, reproducible).
Each sentence is built from facts that are true for its subject, not random slot-filling:

```
the cow eats grass .            the owl hoots at night .
the dog chases the ball .       the mouse hides from the cat .
the cat sleeps on the sofa .    my father walks the dog .
the lion is very strong .       she is a doctor .
```

- 16 animals (sound, food, what they chase, adjectives, home, where they sleep, actions), people and their activities, and "he/she is ..." sentences.
- 223 words, at most 7 words per sentence (8 tokens with `<eos>`), fits `context_len = 8`.

```bash
python data.py           # 50,000 sentences
python data.py 100000    # custom size
```
