"""Step 3 model: token + positional embeddings -> causal self-attention -> linear output layer.

    token IDs (B, T)
       ├─ token_emb = Embedding(vocab_size, d_model)   -> (B, T, d)   "what the word is"
       └─ pos_emb   = Embedding(context_len, d_model)  -> (T, d)      "where it is"
    x      = token_emb + pos_emb                       -> (B, T, d)   (pos is broadcast over B)
    x      = attn(x)                                   -> (B, T, d)   each word mixes in earlier words
    logits = lm_head(x)  Linear(d_model, vocab_size)   -> (B, T, V)   one score per word

Attention (one head, no residual yet; both come in Step 4):
    Q, K, V = x·W_q, x·W_k, x·W_v                      (B, T, d) each
    scores  = Q·Kᵀ / √d                                (B, T, T)  how well word i matches word j
    scores  = -inf above the diagonal                  causal mask: no looking at the future
    weights = softmax(scores)                          (B, T, T)  each row sums to 1
    out     = (weights · V) · W_o                      (B, T, d)
"""

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """One attention head over the full d_model (Step 3). Written out by hand, step by step."""

    def __init__(self, d_model: int, context_len: int):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # "what am I looking for?"
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # "what do I offer?"
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # "what information do I hand over?"
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # output projection
        # mask[i, j] = True where word i may look at word j (j <= i). A buffer: saved, but not learned.
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len, dtype=torch.bool)))
        self.last_weights = None  # (B, T, T) from the latest forward pass, for explore.py

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)  # (B, T, d) each
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d)  # (B, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # hide future words
        weights = torch.softmax(scores, dim=-1)  # (B, T, T), each row sums to 1
        self.last_weights = weights.detach()
        out = weights @ V  # (B, T, d): weighted average of the value vectors
        return self.W_o(out)  # (B, T, d)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, context_len: int = 8, use_pos: bool = True,
                 use_attn: bool = True):
        super().__init__()
        self.context_len = context_len
        self.use_pos = use_pos
        self.use_attn = use_attn  # False = the Step 2 model
        self.token_emb = nn.Embedding(vocab_size, d_model)  # (V, d): one row per word
        self.pos_emb = nn.Embedding(context_len, d_model)  # (T_max, d): one row per position
        self.attn = CausalSelfAttention(d_model, context_len) if use_attn else None
        self.lm_head = nn.Linear(d_model, vocab_size)  # d -> V scores (logits)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.context_len, f"Sequence of {T} tokens > context_len {self.context_len}"
        x = self.token_emb(idx)  # (B, T, d): look up one row per token
        if self.use_pos:
            positions = torch.arange(T, device=idx.device)  # [0, 1, ..., T-1]
            x = x + self.pos_emb(positions)  # (T, d) broadcast over the batch
        if self.use_attn:
            x = self.attn(x)  # (B, T, d): replaced by the attention output (residual comes in Step 4)
        return self.lm_head(x)  # (B, T, V) raw scores, NOT probabilities


def softmax_manual(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """softmax(z)_i = exp(z_i) / sum_j exp(z_j).

    Subtracting the max first doesn't change the result (softmax ignores a shift)
    but keeps exp() from overflowing on large scores.
    """
    z = logits - logits.max(dim=dim, keepdim=True).values
    e = z.exp()
    return e / e.sum(dim=dim, keepdim=True)


def cross_entropy_manual(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int | None = None) -> torch.Tensor:
    """loss = mean over positions of -log(probability given to the correct word)."""
    logits = logits.reshape(-1, logits.shape[-1])  # (N, V)
    targets = targets.reshape(-1)  # (N,)
    if ignore_index is not None:
        keep = targets != ignore_index
        logits, targets = logits[keep], targets[keep]
    probs = softmax_manual(logits)  # (N, V)
    p_true = probs[torch.arange(len(targets)), targets]  # (N,)
    return -p_true.log().mean()


def count_parameters(model: GPT) -> dict[str, int]:
    counts = {
        "token_emb": model.token_emb.weight.numel(),
        "pos_emb": model.pos_emb.weight.numel() if model.use_pos else 0,
    }
    if model.use_attn:
        for name in ("W_q", "W_k", "W_v", "W_o"):
            counts[f"attn.{name}"] = getattr(model.attn, name).weight.numel()
    counts |= {
        "lm_head.weight": model.lm_head.weight.numel(),
        "lm_head.bias": model.lm_head.bias.numel(),
    }
    counts["total"] = sum(counts.values())
    return counts
