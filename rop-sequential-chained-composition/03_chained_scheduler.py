"""`ChainedScheduler`: every scheduler in the chain acts on every step.

This is the other half of what this branch fixes, and the counterpart to
02_sequential_lr_broken.py. The two composition classes are easy to mix up:

    SequentialLR      runs ONE scheduler at a time and switches at a milestone.
    ChainedScheduler  runs ALL of them on every step, each one starting from the
                      learning rate the previous one just produced, so their
                      effects multiply.

Here a steady ExponentialLR decay is chained with a ReduceLROnPlateau -- a
combination ChainedScheduler refused to hold until now (GitHub issue #110761):

    ValueError: ChainedScheduler does not support `ReduceLROnPlateau` scheduler
    as it requires additional kwargs to be specified when calling `step`, but
    got one at index 1 in the given schedulers sequence.

Every epoch the exponential takes its x0.9. On the epochs where the loss has
stopped improving, the plateau scheduler takes a further x0.5 on top of that, so
the learning rate drops by x0.45 in that single step. `step(metrics)` is one
call for the whole chain: the schedulers that don't take a metric ignore it, and
a chain holding a ReduceLROnPlateau requires one.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import ChainedScheduler, ExponentialLR, ReduceLROnPlateau

torch.manual_seed(42)

# y = 3x + 2 + noise
x = torch.linspace(-1, 1, 64).unsqueeze(1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

GAMMA = 0.9
FACTOR = 0.5

decay = ExponentialLR(optimizer, gamma=GAMMA)
# `threshold` is what counts as an improvement: the default 1e-4 is relative, and
# on a problem this small the loss keeps clearing it forever, so the plateau
# scheduler would never fire. Asking for 1% steps makes it earn its name.
plateau = ReduceLROnPlateau(
    optimizer, mode="min", factor=FACTOR, patience=2, threshold=1e-2
)

# `optimizer` is optional -- the chain takes it from its first scheduler -- but
# passing it is clearer. This is the construction that used to fail.
chained = ChainedScheduler([decay, plateau], optimizer=optimizer)

loss_fn = nn.MSELoss()

cuts = 0
epochs = 40

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

    lr_before = optimizer.param_groups[0]["lr"]
    best_before = plateau.best
    # One call steps the whole chain, in the order the schedulers were given.
    chained.step(loss.item())
    lr_after = optimizer.param_groups[0]["lr"]

    if plateau.best != best_before:
        note = f"decay x{GAMMA}  (improved, best={plateau.best:.8f})"
    elif plateau.num_bad_epochs == 0:
        # Without an improvement, `num_bad_epochs` only goes back to zero when
        # the scheduler has just called a plateau. (That is also true of an
        # epoch spent in cooldown, which this run has none of: cooldown=0.)
        cuts += 1
        note = (
            f"decay x{GAMMA} + PLATEAU DETECTED x{FACTOR}"
            f"  (net x{lr_after / lr_before:.3f})"
        )
    else:
        note = (
            f"decay x{GAMMA}  "
            f"(no improvement {plateau.num_bad_epochs}/{plateau.patience + 1})"
        )

    print(
        f"epoch {epoch:2d}  loss={loss.item():.8f}  "
        f"lr {lr_before:.5f} -> {lr_after:.5f}  {note}"
    )

decay_only = 0.5 * GAMMA**epochs
print(
    f"\nfinal lr {optimizer.param_groups[0]['lr']:.6f}, and get_last_lr() agrees: "
    f"{chained.get_last_lr()[0]:.6f}"
)
print(
    f"ExponentialLR on its own would have ended at {decay_only:.6f}; the rest is "
    f"the {cuts} plateau cut(s) the chain multiplied in, x{FACTOR**cuts}."
)
