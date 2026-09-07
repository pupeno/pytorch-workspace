"""Minimal refresher: plain `ReduceLROnPlateau`, no composition involved.

Trains a toy linear regression model long enough that the loss plateaus,
so we can watch the scheduler actually fire and drop the LR.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(42)

# y = 3x + 2 + noise
x = torch.linspace(-1, 1, 64).unsqueeze(1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

loss_fn = nn.MSELoss()

for epoch in range(40):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

    lr_before = optimizer.param_groups[0]["lr"]
    scheduler.step(loss.item())  # metrics arg is required -- this is the whole API difference
    lr_after = optimizer.param_groups[0]["lr"]

    flag = "  <- LR reduced" if lr_after != lr_before else ""
    print(f"epoch {epoch:2d}  loss={loss.item():.5f}  lr={lr_after:.5f}  "
          f"num_bad_epochs={scheduler.num_bad_epochs}{flag}")
