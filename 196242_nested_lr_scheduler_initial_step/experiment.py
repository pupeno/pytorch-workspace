#!/usr/bin/env python3
"""Compare composite LR schedules at one and two nesting levels."""

import torch
from torch.nn import Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    ExponentialLR,
    SequentialLR,
)


STEPS = 5


def collect_lrs(make_scheduler, nesting_levels):
    optimizer = SGD([Parameter(torch.zeros(1))], lr=0.1)
    scheduler = make_scheduler(optimizer)
    if nesting_levels == 2:
        scheduler = SequentialLR(optimizer, [scheduler], milestones=[])

    lrs = []
    for _ in range(STEPS):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return lrs


def make_sequential(optimizer):
    return SequentialLR(
        optimizer,
        [
            ConstantLR(optimizer, factor=0.5, total_iters=2),
            ConstantLR(optimizer, factor=0.2, total_iters=10),
        ],
        milestones=[2],
    )


def make_chained(optimizer):
    return ChainedScheduler(
        [
            ConstantLR(optimizer, factor=0.5, total_iters=2),
            ExponentialLR(optimizer, gamma=0.9),
        ],
        optimizer=optimizer,
    )


def print_comparison(name, make_scheduler):
    one_level_lrs = collect_lrs(make_scheduler, nesting_levels=1)
    two_level_lrs = collect_lrs(make_scheduler, nesting_levels=2)

    print(f"\n{name}")
    print("epoch |    one level |   two levels | result")
    print("------+--------------+--------------+-------")
    for epoch, (one_level, two_levels) in enumerate(
        zip(one_level_lrs, two_level_lrs, strict=True)
    ):
        result = "same" if one_level == two_levels else "DIFF"
        print(f"{epoch:5d} | {one_level:12.6f} | {two_levels:12.6f} | {result}")


print(f"PyTorch {torch.__version__}")
print(f"Imported from {torch.__file__}")
print_comparison("SequentialLR inside SequentialLR", make_sequential)
print_comparison("ChainedScheduler inside SequentialLR", make_chained)
