# tiny-gpt

A small GPT language model written from scratch in PyTorch. It is trained on a mix of children's stories,
educational web pages and Python code, and continues the text you start:

> **Once upon a time**, there was a little girl named Lily. She loved to play with her toys and have tea parties.
> One day, she found a big box in her room. She was very excited and wanted to see what was inside.

Stories come out fluent. Web text has good grammar but made-up facts, and code has the right shape
(indentation, `def`, `class`, `self.x = x`) but wrong logic.

## Model

| Setting | Value |
|---|---|
| tokenizer | byte-level BPE, 16,384 tokens (including `<|endoftext|>`) |
| context length | 512 tokens |
| numbers per token (d_model) | 512 |
| attention heads | 8 × 64 |
| FFN hidden size | 2,048 |
| blocks | 8 |

## Parameters: 33,870,848

| Layer | Shape | Parameters |
|---|---|---|
| Token embedding | 16,384 × 512 | 8,388,608 |
| Position embedding | 512 × 512 | 262,144 |
| **Block** (× 8) | | **3,150,336** each |
| &nbsp;&nbsp;LayerNorm 1 | 512 + 512 | 1,024 |
| &nbsp;&nbsp;Attention W_q, W_k, W_v, W_o | 4 × 512 × 512 | 1,048,576 |
| &nbsp;&nbsp;LayerNorm 2 | 512 + 512 | 1,024 |
| &nbsp;&nbsp;FFN layer 1 | 512 × 2,048 + 2,048 | 1,050,624 |
| &nbsp;&nbsp;FFN layer 2 | 2,048 × 512 + 512 | 1,049,088 |
| 8 blocks | 8 × 3,150,336 | 25,202,688 |
| Final LayerNorm | 512 + 512 | 1,024 |
| Output layer | weights shared with the token embedding, + 16,384 biases | 16,384 |
| **Total** | | **33,870,848** |

74% of the parameters are in the 8 blocks and 25% in the token embedding. The output layer reuses the token
embedding (weight tying), which saves 8.4M parameters. Stored as 32-bit floats, `model.pt` is 138 MB.

## Data

| Source | Characters | Documents |
|---|---|---|
| stories: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 300M | ~373,000 stories |
| web: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (one file) | 700M | ~147,000 pages |
| code: [codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) (one file) | 500M | ~56,000 Python files |

`data.py` shuffles each source, keeps 1% of it for validation (`data/val_stories.txt`, `data/val_web.txt`,
`data/val_code.txt`) and mixes the rest into `data/train.txt`: 570,752 documents, 1.5 GB, 382.6M tokens.

## Training

| | |
|---|---|
| batches | 16 random windows of 512 tokens; the target is the same window shifted by one token |
| steps | 60,000 (492M tokens, about 1.3 passes over the data) |
| optimizer | AdamW; learning rate warms up to 0.0006 in 1,000 steps, then decays to 0.00006 (cosine); gradients clipped at 1.0 |
| speed | bfloat16 on the Mac's GPU (MPS), 0.4 s per step, about 19,500 tokens per second |
| saving | every 1,000 steps: `checkpoint.pt` (model, optimizer and step, so training can resume) and `model.pt` when the mean validation loss is the best so far |
| time | 7.0 hours |

| Validation | Loss | Perplexity | Next token guessed right |
|---|---|---|---|
| stories | 1.46 | 4.3 | 62.5% |
| web | 3.62 | 37.2 | 35.2% |
| code | 1.85 | 6.3 | 62.9% |

The loss starts at 9.8, close to a random guess among 16,384 tokens (ln 16,384 = 9.7).

## Files

| File | What it does |
|---|---|
| `data.py` | downloads the three sources and writes `data/train.txt` and the three validation files |
| `bpe.py` | byte-level BPE tokenizer; `python bpe.py` trains `bpe.json` on the first 50M characters of `data/train.txt` |
| `model.py` | the model: `Attention`, `FFN`, `Block` and `GPT` |
| `train.py` | tokenizes the data (cached in `data/*.npy`), trains the model and saves `model.pt` |
| `generate.py` | continues a text, with temperature, top-k and top-p sampling |
| `explore.py` | model report: settings, validation per source, fixed samples and next-token guesses; with your own text it shows the next-token guesses |
| `server.py` | web server for the chat page |
| `frontend/` | the chat page (React) |

## Run

```bash
pip install -r requirements.txt
python data.py                       # downloads about 2.7 GB and writes the data files
python train.py                      # about 7 hours on an Apple-silicon Mac; resumes from checkpoint.pt
python generate.py "def is_prime(n):"
python explore.py                    # model report, to compare models after training
python explore.py "Lily was very"    # next-token guesses for your own text
```

The data files are not in the repository; `data.py` downloads them. `bpe.json` is in the repository, so
`bpe.py` is only needed to rebuild the tokenizer.

Chat in the browser:

```bash
cd frontend && npm install && npm run build && cd ..
python server.py
```

Then open http://localhost:8000.
