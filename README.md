# tiny-gpt

A small GPT language model written from scratch in PyTorch. It is trained on 639,691 short children's stories
(TinyStories) and continues the story you start:

> **Once upon a time**, there was a little girl named Lily. She loved to play with her toys and make them laugh.
> One day, she found a big, red ball in the park. She wanted to play with it, but she was too small.
> Lily asked her mom, "Can I play with the ball, please?" Her mom said, "No, but be careful. Don't go too far."

## Model

```
x = token_emb + pos_emb

for block in blocks:
    x = x + attention(ln1(x))
    x = x + ffn(ln2(x))

logits = lm_head(ln_f(x))           lm_head shares its weights with token_emb
```

| Setting | Value |
|---|---|
| tokenizer | byte-level BPE, 4,096 tokens (including `<|endoftext|>`) |
| context length | 256 tokens |
| numbers per token (d_model) | 384 |
| attention heads | 6 × 64 |
| FFN hidden size | 1,536 |
| blocks | 6 |

## Parameters: 12,313,600

| Layer | Shape | Parameters |
|---|---|---|
| Token embedding | 4,096 × 384 | 1,572,864 |
| Position embedding | 256 × 384 | 98,304 |
| **Block** (× 6) | | **1,772,928** each |
| &nbsp;&nbsp;LayerNorm 1 | 384 + 384 | 768 |
| &nbsp;&nbsp;Attention W_q, W_k, W_v, W_o | 4 × 384 × 384 | 589,824 |
| &nbsp;&nbsp;LayerNorm 2 | 384 + 384 | 768 |
| &nbsp;&nbsp;FFN layer 1 | 384 × 1,536 + 1,536 | 591,360 |
| &nbsp;&nbsp;FFN layer 2 | 1,536 × 384 + 384 | 590,208 |
| 6 blocks | 6 × 1,772,928 | 10,637,568 |
| Final LayerNorm | 384 + 384 | 768 |
| Output layer | weights shared with the token embedding, + 4,096 biases | 4,096 |
| **Total** | | **12,313,600** |

86% of the parameters are in the 6 blocks. The output layer reuses the token embedding (weight tying), which
saves 1.57M parameters.

## Training

| | |
|---|---|
| data | `data/train.txt`: first 500 MB of TinyStories, 639,691 stories, 132.5M tokens · `data/val.txt`: 27,630 stories, 5.7M tokens |
| batches | 32 random windows of 256 tokens; the target is the same window shifted by one token |
| steps | 10,000 (82M tokens) |
| optimizer | AdamW; learning rate warms up to 0.001 in 200 steps, then decays to 0.0001 (cosine); gradients clipped at 1.0 |
| checkpoint | `model.pt` is saved whenever the validation loss is the best so far (checked every 500 steps) |
| time | 43 minutes on the Mac's GPU (MPS) |
| loss | 8.3 at the start (random guess among 4,096 tokens) → **1.90** validation loss |

## Files

| File | What it does |
|---|---|
| `data.py` | downloads TinyStories: `data/train.txt` (500 MB) and `data/val.txt` (22 MB) |
| `bpe.py` | byte-level BPE tokenizer; `python bpe.py` trains `bpe.json` on `data/val.txt` |
| `model.py` | the model: `Attention`, `FFN`, `Block` and `GPT` |
| `train.py` | tokenizes the data (cached in `data/*.npy`), trains the model and saves `model.pt` |
| `generate.py` | continues a story, with temperature, top-k and top-p sampling |
| `explore.py` | model report: settings, validation loss, fixed test stories and next-token guesses; with your own text it shows the next-token guesses |
| `server.py` | web server for the chat page |
| `frontend/` | the chat page (React) |

## Run

```bash
pip install -r requirements.txt
python data.py                   # downloads data/train.txt and data/val.txt
python train.py                  # about 45 minutes on an Apple-silicon Mac
python generate.py "Once upon a time"
python explore.py                # model report, to compare models after training
python explore.py "Lily was very"   # next-token guesses for your own text
```

The data files are not in the repository; `data.py` downloads them. `bpe.json` is in the repository, so
`bpe.py` is only needed to rebuild the tokenizer.

Chat in the browser:

```bash
cd frontend && npm install && npm run build && cd ..
python server.py
```

Then open http://localhost:8000.
