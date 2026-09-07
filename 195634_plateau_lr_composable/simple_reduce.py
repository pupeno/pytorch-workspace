#!/usr/bin/env python3
"""Reduce the learning rate when the loss stops improving.

Each epoch reports what the scheduler decided: whether the loss beat its
previous best, how many epochs it has now gone without one, and the epoch where
it runs out of patience and cuts the learning rate.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


EPOCHS = 40
FACTOR = 0.5
PATIENCE = 3

torch.manual_seed(42)

# y = 3x + 2 + noise
x = torch.linspace(-1, 1, 64).unsqueeze(1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
scheduler = ReduceLROnPlateau(optimizer, factor=FACTOR, patience=PATIENCE)

loss_fn = nn.MSELoss()

start_lr = optimizer.param_groups[0]["lr"]
print(
    f"{type(scheduler).__name__}: start at lr {start_lr}, cut x{FACTOR} whenever "
    f"the loss goes {PATIENCE + 1} epochs without a new best.\n"
)

cuts = 0

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

    lr_before = optimizer.param_groups[0]["lr"]
    best_before = scheduler.best
    scheduler.step(loss.item())
    lr_after = optimizer.param_groups[0]["lr"]

    if scheduler.best != best_before:
        note = f"improved, best={scheduler.best:.8f}"
    elif scheduler.num_bad_epochs == 0:
        # Without an improvement, `num_bad_epochs` only returns to zero when the
        # scheduler has just called a plateau (this run has no cooldown).
        cuts += 1
        note = f"PLATEAU DETECTED, cut x{FACTOR}"
    else:
        note = f"no improvement {scheduler.num_bad_epochs}/{PATIENCE + 1}"

    print(
        f"epoch {epoch:2d}  loss={loss.item():.8f}  "
        f"lr {lr_before:.5f} -> {lr_after:.5f}  {note}"
    )

print(
    f"\n{cuts} plateau cut(s) took the lr from {start_lr:.5f} to "
    f"{optimizer.param_groups[0]['lr']:.5f}, x{FACTOR**cuts}."
)
