# tiny-gpt — LLM From Scratch

Building a GPT step by step.

## The model now (Step 4c)

```
x      = token_emb + pos_emb        (B, T, 64)
x      = x + attention(ln1(x))      LayerNorm, then 4 heads × 16, causal mask, residual
logits = lm_head(ln_f(x))           final LayerNorm, then (B, T, 225)
```

46,177 parameters · val loss 0.899 (best possible on this data: 0.89)

## Files

| File | What it does |
|---|---|
| `data.py` | makes `data/corpus.txt`: 50,000 simple sentences ("the cow eats grass .") |
| `tokenizer.py` | word ↔ number, `<pad>` = 0, `<eos>` = 1 |
| `model.py` | `Attention` and `GPT` (with LayerNorm) |
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
| 4d | feed-forward network | next |
| 4e | stacked blocks | |
