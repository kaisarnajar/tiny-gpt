"""Train the dummy GPT (one Linear layer) with AdamW and MSE loss."""

import torch
import torch.nn as nn

from model import GPT
from tokenizer import Tokenizer

EPOCHS = 500
LR = 0.05
WEIGHT_DECAY = 0.0  # AdamW's default 0.01 would pull w, b slightly toward 0
LOG_EVERY = 25


def main():
    torch.manual_seed(0)
    tokenizer = Tokenizer()

    # Training pairs: current word -> next word
    pairs = [("He", "boy"), ("She", "girl")]
    x = torch.tensor([[tokenizer.encode(a)[0]] for a, _ in pairs], dtype=torch.float32)
    y = torch.tensor([[tokenizer.encode(b)[0]] for _, b in pairs], dtype=torch.float32)

    model = GPT()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"start          | w {model.w():.4f} | b {model.b():.4f}")
    for epoch in range(1, EPOCHS + 1):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == EPOCHS:
            print(
                f"epoch {epoch:4d}/{EPOCHS} | loss {loss.item():.6f} "
                f"| w {model.w():.4f} | b {model.b():.4f}"
            )

    print(f"\nFinal: w = {model.w():.4f}, b = {model.b():.4f}")
    torch.save(model.state_dict(), "model.pt")
    print("Saved weights to model.pt")


if __name__ == "__main__":
    main()
