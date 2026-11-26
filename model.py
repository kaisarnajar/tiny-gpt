"""Step 4a model: token + positional embeddings -> multi-head causal self-attention -> linear output layer.

    token IDs (B, T)
       ├─ token_emb = Embedding(vocab_size, d_model)   -> (B, T, d)   "what the word is"
       └─ pos_emb   = Embedding(context_len, d_model)  -> (T, d)      "where it is"
    x      = token_emb + pos_emb                       -> (B, T, d)   (pos is broadcast over B)
    x      = attn(x)                                   -> (B, T, d)   each word mixes in earlier words
    logits = lm_head(x)  Linear(d_model, vocab_size)   -> (B, T, V)   one score per word

Multi-head attention (no residual yet; that is Step 4b). h heads of hs = d / h numbers each:
    Q, K, V = x·W_q, x·W_k, x·W_v                      (B, T, d) each
    split   : head i gets columns i·hs .. i·hs+hs-1    (B, h, T, hs) each
    scores  = Q·Kᵀ / √hs                               (B, h, T, T)  per head: how well word i matches j
    scores  = -inf above the diagonal                  causal mask: no looking at the future
    weights = softmax(scores)                          (B, h, T, T)  each row sums to 1
    heads   = weights · V                              (B, h, T, hs)
    out     = join(heads) · W_o                        (B, T, d)     heads side by side, then mixed
With n_heads = 1 this is exactly the Step 3 model.
"""

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (Step 4a). Written out by hand, step by step.

    The four d × d matrices are the same as in Step 3; their columns are only cut into n_heads groups,
    so the number of parameters does not change with n_heads.
    """

    def __init__(self, d_model: int, context_len: int, n_heads: int = 1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads  # hs: numbers per head
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # "what am I looking for?"
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # "what do I offer?"
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # "what information do I hand over?"
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # output projection: mixes the heads
        # mask[i, j] = True where word i may look at word j (j <= i). A buffer: saved, but not learned.
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len, dtype=torch.bool)))
        self.last_weights = None  # (B, h, T, T) from the latest forward pass, for explore.py

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B, h, T, hs): head i gets columns i*hs .. i*hs + hs - 1."""
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)  # (B, T, d) each
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)  # (B, h, T, hs) each
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_size)  # (B, h, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # hide future words (every head)
        weights = torch.softmax(scores, dim=-1)  # (B, h, T, T), each row sums to 1
        self.last_weights = weights.detach()
        heads = weights @ V  # (B, h, T, hs): per head, a weighted average of its value vectors
        out = heads.transpose(1, 2).reshape(B, T, d)  # join: heads side by side again -> (B, T, d)
        return self.W_o(out)  # (B, T, d)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, context_len: int = 8, use_pos: bool = True,
                 use_attn: bool = True, n_heads: int = 1):
        super().__init__()
        self.context_len = context_len
        self.use_pos = use_pos
        self.use_attn = use_attn  # False = the Step 2 model
        self.n_heads = n_heads  # 1 = the Step 3 model
        self.token_emb = nn.Embedding(vocab_size, d_model)  # (V, d): one row per word
        self.pos_emb = nn.Embedding(context_len, d_model)  # (T_max, d): one row per position
        self.attn = CausalSelfAttention(d_model, context_len, n_heads) if use_attn else None
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
