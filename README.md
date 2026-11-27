# tiny-gpt

A small GPT language model written from scratch in PyTorch. It is trained on 50,000 simple sentences and
finishes the sentence you start: `the cow eats` → `the cow eats grass .`

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
| vocabulary | 225 words |
| context length | 8 tokens |
| numbers per token (d_model) | 64 |
| attention heads | 4 × 16 |
| FFN hidden size | 256 |
| blocks | 2 |

Validation loss 0.894 (the best possible on this data is 0.892).

## Parameters: 129,121

| Layer | Shape | Parameters |
|---|---|---|
| Token embedding | 225 × 64 | 14,400 |
| Position embedding | 8 × 64 | 512 |
| **Block** (× 2) | | **49,728** each |
| &nbsp;&nbsp;LayerNorm 1 | 64 + 64 | 128 |
| &nbsp;&nbsp;Attention W_q, W_k, W_v, W_o | 4 × 64 × 64 | 16,384 |
| &nbsp;&nbsp;LayerNorm 2 | 64 + 64 | 128 |
| &nbsp;&nbsp;FFN layer 1 | 64 × 256 + 256 | 16,640 |
| &nbsp;&nbsp;FFN layer 2 | 256 × 64 + 64 | 16,448 |
| 2 blocks | 2 × 49,728 | 99,456 |
| Final LayerNorm | 64 + 64 | 128 |
| Output layer | 64 × 225 + 225 | 14,625 |
| **Total** | | **129,121** |

## Files

| File | What it does |
|---|---|
| `data.py` | writes `data/corpus.txt`: 50,000 simple sentences about animals and people |
| `tokenizer.py` | turns words into numbers and back (`<pad>` = 0, `<eos>` = 1) |
| `bpe.py` | a byte-level BPE tokenizer for code, trained into `bpe.json` (2,048 tokens) |
| `model.py` | the model: `Attention`, `FFN`, `Block` and `GPT` |
| `train.py` | trains the model for 100 epochs and saves `model.pt` |
| `generate.py` | finishes a sentence, with temperature, top-k and top-p sampling |
| `explore.py` | prints the parameters and next-word guesses, saves `attention.png` |
| `server.py` | web server for the chat page |
| `frontend/` | the chat page (React) |

## Run

```bash
pip install -r requirements.txt
python train.py
python generate.py "he is a"
python explore.py
```

Chat in the browser:

```bash
cd frontend && npm install && npm run build && cd ..
python server.py
```

Then open http://localhost:8000.
