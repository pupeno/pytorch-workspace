"""The case this branch fixes: `SequentialLR` handing off to `ReduceLROnPlateau`.

The intent here is a common pattern -- warm up with a fixed schedule, then
hand off to ReduceLROnPlateau for the rest of training. This is exactly what
SequentialLR is for, except that until now it refused ReduceLROnPlateau at
construction time (lr_scheduler.py:1136-1141, GitHub issue #68978):

    ValueError: SequentialLR does not support `ReduceLROnPlateau` scheduler as
    it requires additional kwargs to be specified when calling `step`, but got
    one at index 1 in the given schedulers sequence.

With the fix, the construction below is accepted and `step(metrics)` routes the
metric to whichever scheduler is active: the warmup ignores it, the plateau
scheduler uses it. Each line of output names the scheduler that moved the
learning rate that epoch, so the linear ramp and the plateau cuts are easy to
tell apart.

Watch the `handoff` line in particular. The step at the milestone belongs to the
incoming scheduler, exactly as it does for any other pair of schedulers, so the
warmup's last increment is never applied -- the LR peaks at 0.41 rather than the
full 0.5 -- and that epoch's metric is not observed. ReduceLROnPlateau then
carries on from whatever LR the warmup left behind, which is the point of the
handoff.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR

torch.manual_seed(42)

# y = 3x + 2 + noise
x = torch.linspace(-1, 1, 64).unsqueeze(1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

MILESTONE = 5

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=MILESTONE)
plateau = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# This is the line that used to fail.
scheduler = SequentialLR(optimizer, schedulers=[warmup, plateau], milestones=[MILESTONE])

loss_fn = nn.MSELoss()

for epoch in range(40):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

    lr_before = optimizer.param_groups[0]["lr"]
    best_before = plateau.best
    # One call, whichever scheduler is active: `metrics` is only looked at once
    # the milestone hands over to ReduceLROnPlateau.
    scheduler.step(loss.item())
    lr_after = optimizer.param_groups[0]["lr"]

    # SequentialLR steps the warmup until its own epoch count reaches the
    # milestone, which is the call this loop makes on iteration MILESTONE - 1.
    # That call already belongs to ReduceLROnPlateau, so it is the handover.
    if epoch < MILESTONE - 1:
        who, what = "warmup ", f"linear ramp {lr_after - lr_before:+.5f}"
    elif epoch == MILESTONE - 1:
        who, what = "handoff", "plateau takes over, this metric is not observed"
    elif plateau.best != best_before:
        who, what = "plateau", f"improved, best={plateau.best:.8f}"
    elif plateau.num_bad_epochs == 0:
        # Without an improvement, `num_bad_epochs` only goes back to zero when
        # the scheduler has just called a plateau. (That is also true of an
        # epoch spent in cooldown, which this run has none of: cooldown=0.)
        cut = f"cut x{plateau.factor}" if lr_after < lr_before else "lr already at its floor"
        who, what = "plateau", f"PLATEAU DETECTED, {cut}"
    else:
        who, what = (
            "plateau",
            f"no improvement {plateau.num_bad_epochs}/{plateau.patience + 1}",
        )

    print(f"epoch {epoch:2d}  loss={loss.item():.8f}  "
          f"lr {lr_before:.5f} -> {lr_after:.5f}  {who}  {what}")
