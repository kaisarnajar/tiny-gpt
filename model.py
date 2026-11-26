import math

import torch
import torch.nn as nn


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


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, context_len, n_heads):
        super().__init__()
        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.attn = Attention(d_model, context_len, n_heads)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T))
        x = x + self.attn(x)
        return self.lm_head(x)
