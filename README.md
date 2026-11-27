# tiny-gpt

A small GPT language model written from scratch in PyTorch. It is trained on 27,630 short children's stories
(TinyStories) and continues the story you start:

> **Once upon a time**, there was a little girl named Lily. She loved to play with her toys and eat yummy
> treats. One day, she found a big, red ball in her room. She was very happy and wanted to play with it.

## Model

```
x = token_emb + pos_emb

for block in blocks:
    x = x + attention(ln1(x))
    x = x + ffn(ln2(x))

logits = lm_head(ln_f(x))
```

| Setting | Value |
|---|---|
| tokenizer | byte-level BPE, 4,096 tokens (including `<|endoftext|>`) |
| context length | 128 tokens |
| numbers per token (d_model) | 128 |
| attention heads | 4 × 32 |
| FFN hidden size | 512 |
| blocks | 4 |

## Parameters: 1,860,352

| Layer | Shape | Parameters |
|---|---|---|
| Token embedding | 4,096 × 128 | 524,288 |
| Position embedding | 128 × 128 | 16,384 |
| **Block** (× 4) | | **197,760** each |
| &nbsp;&nbsp;LayerNorm 1 | 128 + 128 | 256 |
| &nbsp;&nbsp;Attention W_q, W_k, W_v, W_o | 4 × 128 × 128 | 65,536 |
| &nbsp;&nbsp;LayerNorm 2 | 128 + 128 | 256 |
| &nbsp;&nbsp;FFN layer 1 | 128 × 512 + 512 | 66,048 |
| &nbsp;&nbsp;FFN layer 2 | 512 × 128 + 128 | 65,664 |
| 4 blocks | 4 × 197,760 | 791,040 |
| Final LayerNorm | 128 + 128 | 256 |
| Output layer | 128 × 4,096 + 4,096 | 528,384 |
| **Total** | | **1,860,352** |

More than half of the parameters (57%) are in the token embedding and the output layer, because every one of
the 4,096 tokens needs its own row in both.

## Training

| | |
|---|---|
| data | `data/stories.txt`: 22.5 MB, 27,630 stories → 5.7 million tokens (90% train, 10% validation) |
| batches | 32 random windows of 128 tokens; the target is the same window shifted by one token |
| steps | 5,000 (about 20 million tokens, ~4 passes over the data) |
| optimizer | AdamW, learning rate 0.001 |
| time | about 2 minutes on the Mac's GPU (MPS) |
| loss | 8.3 at the start (random guess among 4,096 tokens) → 2.38 validation loss |

## Files

| File | What it does |
|---|---|
| `data.py` | downloads TinyStories into `data/stories.txt` |
| `bpe.py` | byte-level BPE tokenizer; `python bpe.py` trains `bpe.json` on the stories |
| `model.py` | the model: `Attention`, `FFN`, `Block` and `GPT` |
| `train.py` | trains the model and saves `model.pt` |
| `generate.py` | continues a story, with temperature, top-k and top-p sampling |
| `explore.py` | prints the parameters and next-token guesses, saves `attention.png` |
| `server.py` | web server for the chat page |
| `frontend/` | the chat page (React) |

## Run

```bash
pip install -r requirements.txt
python train.py
python generate.py "Once upon a time"
python explore.py
```

`data/stories.txt` and `bpe.json` are in the repository, so `data.py` and `bpe.py` are only needed to rebuild them.

Chat in the browser:

```bash
cd frontend && npm install && npm run build && cd ..
python server.py
```

Then open http://localhost:8000.
