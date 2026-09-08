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

def run_sequential_lr_example():
    print("\nSequentialLR:")

    one_level_optimizer = SGD([Parameter(torch.zeros(1))], lr=0.1)
    one_level_scheduler = SequentialLR(
        one_level_optimizer,
        [
            ConstantLR(one_level_optimizer, factor=0.2),
            ConstantLR(one_level_optimizer, factor=0.5),
        ],
        milestones=[2],
    )

    # SequentialLR inside SequentialLR
    two_level_optimizer = SGD([Parameter(torch.zeros(1))], lr=0.1)
    two_level_scheduler = SequentialLR(
        two_level_optimizer,
        [SequentialLR(
            two_level_optimizer,
            [
                ConstantLR(two_level_optimizer, factor=0.2),
                ConstantLR(two_level_optimizer, factor=0.5),
            ],
            milestones=[2],
        )],
        milestones=[],
    )

    one_level_lrs = _collect_lrs(one_level_scheduler)
    two_level_lrs = _collect_lrs(two_level_scheduler)
    _print_comparison(one_level_lrs, two_level_lrs)

def run_chained_scheduler_example():
    print("\nChainedScheduler:")
    one_level_optimizer = SGD([Parameter(torch.zeros(1))], lr=0.1)
    one_level_scheduler = ChainedScheduler(
        [
            ConstantLR(one_level_optimizer, factor=0.5, total_iters=2),
            ExponentialLR(one_level_optimizer, gamma=0.9),
        ],
        optimizer=one_level_optimizer,
    )

    # ChainedScheduler inside SequentialLR
    two_level_optimizer = SGD([Parameter(torch.zeros(1))], lr=0.1)
    two_level_scheduler = SequentialLR(
        two_level_optimizer,
        [ChainedScheduler(
            [
                ConstantLR(two_level_optimizer, factor=0.5, total_iters=2),
                ExponentialLR(two_level_optimizer, gamma=0.9),
            ],
            optimizer=two_level_optimizer,
        )],
        milestones=[],
    )

    one_level_lrs = _collect_lrs(one_level_scheduler)
    two_level_lrs = _collect_lrs(two_level_scheduler)
    _print_comparison(one_level_lrs, two_level_lrs)


def main():
    run_sequential_lr_example()
    run_chained_scheduler_example()


def _collect_lrs(scheduler):
    lrs = []
    for _ in range(STEPS):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.optimizer.step()
        scheduler.step()
    return lrs


def _print_comparison(one_level_lrs, two_level_lrs):
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


if __name__ == "__main__":
    main()
