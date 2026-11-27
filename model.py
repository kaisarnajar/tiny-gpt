import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, context_len, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len)).bool())

    def forward(self, x):
        B, T, d = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        self.weights = torch.softmax(scores, dim=-1)
        out = self.weights @ v
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.W_o(out)


class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, context_len, n_heads):
        super().__init__()
        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, context_len, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T))
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return self.lm_head(self.ln_f(x))
