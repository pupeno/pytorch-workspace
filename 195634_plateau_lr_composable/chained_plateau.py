#!/usr/bin/env python3
"""Decay the learning rate every epoch, and cut it again when the loss stalls.

`ChainedScheduler` runs every scheduler on every step, each one starting from
the learning rate the previous one just produced, so their effects multiply.
That is the other half of composition, and the counterpart to
`sequential_reduce.py` / `sequential_plateau.py`:

    SequentialLR      runs ONE scheduler at a time and switches at a milestone.
    ChainedScheduler  runs ALL of them on every step, so their effects multiply.

Here a steady `ExponentialLR` decay is chained with a plateau scheduler. Every
epoch takes the x0.9 decay; on the epochs where the loss has stopped improving
the plateau scheduler takes a further x0.5 on top of that, so the learning rate
drops by x0.45 in that single step.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import ChainedScheduler, ExponentialLR, PlateauLR


EPOCHS = 40
GAMMA = 0.9
FACTOR = 0.5
PATIENCE = 2
# `threshold` is what counts as an improvement: the default 1e-4 is relative, and
# on a problem this small the loss keeps clearing it forever, so the plateau
# scheduler would never fire. Asking for 1% steps makes it earn its name.
THRESHOLD = 1e-2

torch.manual_seed(42)

# y = 3x + 2 + noise
x = torch.linspace(-1, 1, 64).unsqueeze(1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

decay = ExponentialLR(optimizer, gamma=GAMMA)
plateau = PlateauLR(
    optimizer, factor=FACTOR, patience=PATIENCE, threshold=THRESHOLD
)

# Whether this line is accepted at all is the whole point of the demo. The
# optimizer is optional -- the chain takes it from its first scheduler -- but
# passing it is clearer.
scheduler = ChainedScheduler([decay, plateau], optimizer=optimizer)

loss_fn = nn.MSELoss()

start_lr = optimizer.param_groups[0]["lr"]
print(
    f"ExponentialLR x{GAMMA} every epoch, chained with {type(plateau).__name__} "
    f"cutting a further x{FACTOR} after {PATIENCE + 1} epochs without a new best.\n"
)

cuts = 0

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

    lr_before = optimizer.param_groups[0]["lr"]
    best_before = plateau.best
    # One call steps the whole chain, in the order the schedulers were given.
    scheduler.step(metrics=loss.item())
    lr_after = optimizer.param_groups[0]["lr"]

    if plateau.best != best_before:
        cut, what = "", f"improved, best={plateau.best:.8f}"
    elif plateau.num_bad_epochs == 0:
        # Without an improvement, `num_bad_epochs` only returns to zero when the
        # scheduler has just called a plateau (this run has no cooldown).
        cuts += 1
        cut = f" + PLATEAU DETECTED x{FACTOR}"
        what = f"net x{lr_after / lr_before:.3f} this epoch"
    else:
        cut = ""
        what = f"no improvement {plateau.num_bad_epochs}/{PATIENCE + 1}"

    did = f"decay x{GAMMA}{cut}"
    print(
        f"epoch {epoch:2d}  loss={loss.item():.8f}  "
        f"lr {lr_before:.5f} -> {lr_after:.5f}  {did:<34}  {what}"
    )

decay_only = start_lr * GAMMA**EPOCHS
print(
    f"\nfinal lr {optimizer.param_groups[0]['lr']:.6f}, and get_last_lr() agrees: "
    f"{scheduler.get_last_lr()[0]:.6f}"
)
print(
    f"ExponentialLR on its own would have ended at {decay_only:.6f}; the rest is "
    f"the {cuts} plateau cut(s) the chain multiplied in, x{FACTOR**cuts}."
)
