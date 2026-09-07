"""Same toy problem as 00_rop_baseline.py, but with no LR scheduler at all.

Constant LR throughout -- a control to compare against the ReduceLROnPlateau run.
"""

import torch
from torch import nn

torch.manual_seed(0)

# y = 3x + 2 + noise
x = torch.linspace(-1, 1, 64).unsqueeze(1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

loss_fn = nn.MSELoss()

for epoch in range(40):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

    lr = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d}  loss={loss.item():.5f}  lr={lr:.5f}")
