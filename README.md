# tiny-gpt — LLM From Scratch

Building a GPT step by step.

## The model now (Step 4e)

```
x = token_emb + pos_emb             (B, T, 64)

for block in blocks:                2 blocks, each with its own weights
    x = x + attention(ln1(x))       LayerNorm, then 4 heads × 16, causal mask, residual
    x = x + ffn(ln2(x))             LayerNorm, then FFN 64 → 256 → GELU → 64, residual

logits = lm_head(ln_f(x))           final LayerNorm, then (B, T, 225)
```

129,121 parameters · val loss 0.894 (best possible on this data: 0.89)

## Files

| File | What it does |
|---|---|
| `data.py` | makes `data/corpus.txt`: 50,000 simple sentences ("the cow eats grass .") |
| `tokenizer.py` | word ↔ number, `<pad>` = 0, `<eos>` = 1 (used by the current model) |
| `bpe.py` | byte-level BPE tokenizer for code: `python bpe.py` trains `bpe.json` (2,048 tokens) |
| `model.py` | `Attention`, `FFN`, `Block` and `GPT` |
| `train.py` | trains for 100 epochs and saves `model.pt` |
| `generate.py` | finishes a sentence (temperature, top-k, top-p) |
| `explore.py` | parameters, next-word guesses, similar words, `attention.png` |
| `server.py` + `frontend/` | chat in the browser |

## Run

```bash
pip install -r requirements.txt
python train.py                  # ~1 min
python generate.py "he is a"     # he is a student .
python explore.py
```

Web chat:

```bash
cd frontend && npm install && npm run build && cd ..
python server.py                 # http://localhost:8000
```

## Steps

| Step | What | Val loss |
|---|---|---|
| 1 | dummy model: one `Linear` layer | – |
| 2 | token + position embeddings | 1.27 |
| 3 | attention (1 head) | 0.904 |
| 4a | multi-head attention (4 × 16) | 0.906 |
| 4b | residual connection: `x = x + attention(x)` | 0.901 |
| 4c | LayerNorm before attention and before the output layer | 0.899 |
| 4d | feed-forward network: 64 → 256 → GELU → 64 | 0.895 |
| 4e | 2 stacked blocks (`N_LAYERS = 2`) | 0.894 |
| 5 | BPE tokenizer (`bpe.py`) | done |
| 5b | train on code with BPE | next |

## BPE tokenizer

Start with the 256 possible bytes, then repeat: find the most common pair of neighbouring tokens and merge it
into a new token. Trained on the Python standard library on this Mac (4.6 MB of code), 1,792 merges:

```
first merges : "  ", "    ", "   ", "\n    ", "se", "in", "re", ...      (indentation comes first)
'for i in range(10):'  ->  ['for', ' i', ' in', ' range', '(', '10', '):']      19 characters -> 7 tokens
```

About 3 characters per token on code. Any text works (unknown words fall back to single bytes), and
`decode(encode(text)) == text` always.
