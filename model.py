"""The GPT black box. In Step 1 it is a single linear layer: y = w * x + b."""

import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 2 parameters: w and b
        nn.init.zeros_(self.linear.weight)  # w starts at 0
        nn.init.zeros_(self.linear.bias)  # b starts at 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: token IDs as floats, shape (N, 1) -> predicted next-token ID, shape (N, 1)
        return self.linear(x)

    def w(self) -> float:
        return self.linear.weight.item()

    def b(self) -> float:
        return self.linear.bias.item()
