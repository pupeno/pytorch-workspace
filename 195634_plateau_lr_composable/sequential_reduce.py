#!/usr/bin/env python3
"""Warm the learning rate up, then reduce it when the loss stops improving.

`SequentialLR` runs one scheduler at a time and switches at a milestone: a
`LinearLR` ramp for the first few epochs, then the plateau scheduler for the
rest of training. `scheduler.step(...)` is a single call for both -- the warmup
ignores the metric, the plateau scheduler watches it. Each epoch names which of
the two moved the learning rate, and why.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR


EPOCHS = 40
MILESTONE = 5
FACTOR = 0.5
PATIENCE = 3

torch.manual_seed(42)

# y = 3x + 2 + noise
x = torch.linspace(-1, 1, 64).unsqueeze(1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=MILESTONE)
plateau = ReduceLROnPlateau(optimizer, factor=FACTOR, patience=PATIENCE)

# Whether this line is accepted at all is the whole point of the demo.
scheduler = SequentialLR(optimizer, schedulers=[warmup, plateau], milestones=[MILESTONE])

loss_fn = nn.MSELoss()

print(
    f"LinearLR warmup for {MILESTONE} epochs, then {type(plateau).__name__} "
    f"cutting x{FACTOR} after {PATIENCE + 1} epochs without a new best.\n"
)

cuts = 0
peak_lr = 0.0

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

    lr_before = optimizer.param_groups[0]["lr"]
    best_before = plateau.best
    scheduler.step(loss.item())
    lr_after = optimizer.param_groups[0]["lr"]
    peak_lr = max(peak_lr, lr_before, lr_after)

    # SequentialLR steps the warmup until its own epoch count reaches the
    # milestone, so the call on iteration MILESTONE - 1 is already the handover.
    if epoch < MILESTONE - 1:
        who, what = "warmup ", f"linear ramp {lr_after - lr_before:+.5f}"
    elif epoch == MILESTONE - 1:
        who, what = "handoff", "plateau takes over, this metric is not observed"
    elif plateau.best != best_before:
        who, what = "plateau", f"improved, best={plateau.best:.8f}"
    elif plateau.num_bad_epochs == 0:
        # Without an improvement, `num_bad_epochs` only returns to zero when the
        # scheduler has just called a plateau (this run has no cooldown).
        cuts += 1
        who, what = "plateau", f"PLATEAU DETECTED, cut x{FACTOR}"
    else:
        who, what = (
            "plateau",
            f"no improvement {plateau.num_bad_epochs}/{PATIENCE + 1}",
        )

    print(
        f"epoch {epoch:2d}  loss={loss.item():.8f}  "
        f"lr {lr_before:.5f} -> {lr_after:.5f}  {who}  {what}"
    )

print(
    f"\nfinal lr {optimizer.param_groups[0]['lr']:.5f}, and get_last_lr() agrees: "
    f"{scheduler.get_last_lr()[0]:.5f}"
)
print(
    f"The warmup peaked at lr {peak_lr:.5f}; from there the plateau scheduler "
    f"made {cuts} cut(s) of x{FACTOR}, x{FACTOR**cuts} in total."
)
