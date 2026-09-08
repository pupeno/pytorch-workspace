#!/usr/bin/env python3
"""Compare composite LR schedules at one and two nesting levels."""

import sys

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


def collect_lrs(scheduler):
    lrs = []
    for _ in range(STEPS):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.optimizer.step()
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
    two_level_optimizer = SGD([Parameter(torch.zeros(1))], lr=0.1)
    two_level_scheduler = SequentialLR(
        two_level_optimizer,
        [make_scheduler(two_level_optimizer)],
        milestones=[],
    )

    one_level_optimizer = SGD([Parameter(torch.zeros(1))], lr=0.1)
    one_level_scheduler = make_scheduler(one_level_optimizer)

    two_level_lrs = collect_lrs(two_level_scheduler)
    one_level_lrs = collect_lrs(one_level_scheduler)

    headers = ["", *(f"Epoch {epoch}" for epoch in range(STEPS))]
    rows = [
        ["One level", *(f"{lr:g}" for lr in one_level_lrs)],
        ["Two levels", *(f"{lr:g}" for lr in two_level_lrs)],
    ]
    widths = [
        max(len(row[column]) for row in [headers, *rows])
        for column in range(len(headers))
    ]

    def print_row(row, styles=None):
        values = []
        for column, (value, width) in enumerate(zip(row, widths, strict=True)):
            value = value.ljust(width)
            if styles is not None and styles[column] and sys.stdout.isatty():
                value = f"{styles[column]}{value}\033[0m"
            values.append(value)
        print("| " + " | ".join(values) + " |")

    print(f"\n{name}:")
    print_row(headers)
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    print_row(rows[0])
    styles = [
        "",
        *(
            "\033[32m" if one_level == two_level else "\033[1;31m"
            for one_level, two_level in zip(
                one_level_lrs, two_level_lrs, strict=True
            )
        ),
    ]
    print_row(rows[1], styles)


print(f"PyTorch {torch.__version__}")
print(f"Imported from {torch.__file__}")
print_comparison("SequentialLR", make_sequential)
print_comparison("ChainedScheduler", make_chained)
