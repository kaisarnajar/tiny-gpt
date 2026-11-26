# tiny-gpt — LLM From Scratch

Building a GPT step by step.

## Step 4b: residual connection (current)

```
x = token_emb + pos_emb
x = x + attention(x)        # was: x = attention(x)
logits = lm_head(x)
```

The word keeps its own vector and ADDS what attention gathered, instead of being replaced by it.
No new parameters (still 45,921). `GPT(residual=True)`, `RESIDUAL = True` in `train.py`.

| | val loss | epoch 1 val loss |
|---|---|---|
| Step 3 (1 head) | 0.904 | 1.008 |
| Step 4a (4 heads) | 0.906 | 0.950 |
| **Step 4b (4 heads + residual)** | **0.901** | **0.944** |

`explore.py` section 9 shows what each part tells the output layer, e.g. for "the cow lives on the":
x alone → house / jungle / forest (a place comes next), attention alone → farm 0.74, x + attention → farm 1.00.

## Step 4a: multi-head attention

```
Q, K, V = x·W_q, x·W_k, x·W_v             (B, T, 64) each, same 64 × 64 matrices as Step 3
split   : 4 heads × 16 columns            (B, 4, T, 16)
scores  = Q·Kᵀ / √16                      (B, 4, T, T)  every head has its own attention pattern
x       = join(softmax(scores) · V) · W_o (B, T, 64)    heads side by side, then mixed by W_o
```

- `model.py`: `CausalSelfAttention(n_heads=...)`, `GPT(n_heads=...)`; `n_heads=1` is exactly Step 3
- `train.py`: `N_HEADS = 4`, saved in `model.pt`
- `explore.py`: section 8 draws one attention map per head (`attention.png`)
- Parameters unchanged: 45,921 (the columns are only cut into groups)

| | val loss | epoch 1 val loss |
|---|---|---|
| Step 3 (1 head) | 0.904 | 1.008 |
| **Step 4a (4 heads)** | **0.906** | **0.950** |

Same final loss, faster start. The heads specialise: for "a" in "he is a doctor ." head 4 looks at "he" (1.00),
head 3 at "is"; for the second "the" in "the cow lives on the farm ." head 1 looks at "cow" (1.00).
On its own 4a gains little; it pays off together with the next steps (4b residual, 4c LayerNorm, 4d FFN, 4e blocks).

## Step 3: causal self-attention

```
x      = token_emb + pos_emb              (B, T, 64)
Q, K, V = x·W_q, x·W_k, x·W_v             (B, T, 64) each, W = 64 × 64
scores = Q·Kᵀ / √64                       (B, T, T)  how well word i matches word j
scores = -inf above the diagonal          causal mask: no looking at future words
x      = softmax(scores) · V · W_o        (B, T, 64) each word mixes in earlier words
logits = lm_head(x)                       (B, T, 225)
```

- `model.py`: `CausalSelfAttention` (one head, written out by hand); `GPT(use_attn=False)` is the Step 2 model.
  45,921 parameters = 29,537 (Step 2) + 4 × 64 × 64 (W_q, W_k, W_v, W_o)
- `train.py`: also prints the best possible loss when the model can see all earlier words
- `explore.py`: section 7 compares Step 2 vs Step 3, section 8 saves the attention map `attention.png`

| | val loss | best possible | "he is a" / "she is a" | "the cow lives on the" / "the monkey lives in the" |
|---|---|---|---|---|
| Step 2 | 1.27 | 1.26 (sees word + position) | identical predictions | farm 0.14 / farm 0.14 |
| **Step 3** | **0.90** | 0.89 (sees all earlier words) | student, doctor, man / woman, girl | **farm 1.00 / trees 0.73** |

```bash
python train.py      # ~1 min on CPU
python explore.py    # attention.png: who looks at whom
```

Next (Step 4, one piece at a time): 4a multi-head attention (done), 4b residual connection (done), 4c LayerNorm,
4d feed-forward network, 4e stacking blocks.

## Step 2: token + positional embeddings

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

### Web UI

Type the start of a sentence in the browser; the model completes it.

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python server.py        # → http://localhost:8000
```

`The Dog` → `the dog barks loudly .`  ·  `the unicorn` → `Unknown word: "unicorn"`

- `server.py`: FastAPI, `POST /api/complete {"prompt": "..."}` → `{"sentence": "..."}`, serves the built page
- `frontend/`: Vite + React + TypeScript (`src/App.tsx`); `npm run dev` for live reload (proxies `/api` to port 8000)

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
