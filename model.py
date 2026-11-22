"""Step 2 model: token embedding + positional embedding -> linear output layer.

    token IDs (B, T)
       ├─ token_emb = Embedding(vocab_size, d_model)   -> (B, T, d)   "what the word is"
       └─ pos_emb   = Embedding(context_len, d_model)  -> (T, d)      "where it is"
    x      = token_emb + pos_emb                       -> (B, T, d)   (pos is broadcast over B)
    logits = lm_head(x)  Linear(d_model, vocab_size)   -> (B, T, V)   one score per word

No attention yet: the prediction at position t sees only the token at t and the
position t itself, nothing earlier in the sentence.
"""

import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, context_len: int = 8, use_pos: bool = True):
        super().__init__()
        self.context_len = context_len
        self.use_pos = use_pos
        self.token_emb = nn.Embedding(vocab_size, d_model)  # (V, d): one row per word
        self.pos_emb = nn.Embedding(context_len, d_model)  # (T_max, d): one row per position
        self.lm_head = nn.Linear(d_model, vocab_size)  # d -> V scores (logits)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.context_len, f"Sequence of {T} tokens > context_len {self.context_len}"
        x = self.token_emb(idx)  # (B, T, d): look up one row per token
        if self.use_pos:
            positions = torch.arange(T, device=idx.device)  # [0, 1, ..., T-1]
            x = x + self.pos_emb(positions)  # (T, d) broadcast over the batch
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
        "lm_head.weight": model.lm_head.weight.numel(),
        "lm_head.bias": model.lm_head.bias.numel(),
    }
    counts["total"] = sum(counts.values())
    return counts
