#!/usr/bin/env python3
"""Warm the learning rate up, then reduce it whenever the loss stops improving.

`SequentialLR` runs one scheduler at a time and switches at a milestone: here a
`LinearLR` warmup for the first few epochs, and `ReduceLROnPlateau` for the rest
of training. `scheduler.step(loss)` is a single call for both of them -- the
warmup ignores the metric, the plateau scheduler watches it.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR


WARMUP_EPOCHS = 5
EPOCHS = 40

torch.manual_seed(42)

# y = 3x + 2 + noise
x = torch.linspace(-1, 1, 64).unsqueeze(1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS),
        ReduceLROnPlateau(optimizer, factor=0.5, patience=3),
    ],
    milestones=[WARMUP_EPOCHS],
)

loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

    scheduler.step(loss.item())

    lr = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d}  loss={loss.item():.6f}  lr={lr:.6f}")
