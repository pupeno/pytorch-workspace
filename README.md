# Pablo's PyTorch Contributing Workspace

This repository is Pablo's development workspace for contributing to [PyTorch](https://github.com/pytorch/pytorch).

It keeps the devcontainer configuration, scratch experiments, useful scripts, and documentation together, while the `pytorch/` contains the PyTorch source code itself.

`pytorch/` is a plain clone of [the fork](https://github.com/pupeno/pytorch) that this repository ignores, not a submodule of it. The devcontainer's post-create script clones it on first start. Nothing here records which PyTorch commit is checked out, so working in `pytorch/` never dirties this repository, and no git command run in the workspace can move the branch being worked on.

## Resolved tickets

- [#176069 - `posix_fallocate` error handling reads stale `errno` instead of return value](https://github.com/pytorch/pytorch/issues/176069).

## Pull requests

- [#195633 - Minimal way to make `ReduceLROnPlateau` composable with `SequentialLR` and `ChainedScheduler`](https://github.com/pytorch/pytorch/pull/195633): fixes [#68978](https://github.com/pytorch/pytorch/issues/68978) and [#110761](https://github.com/pytorch/pytorch/issues/110761).
- [#195634 - PlateauLR, a composable version of ReduceLROnPlateau (and extensible LRSchedule step API to support it)](https://github.com/pytorch/pytorch/pull/195634): fixes [#68978](https://github.com/pytorch/pytorch/issues/68978) and [#110761](https://github.com/pytorch/pytorch/issues/110761).

## Setup

Complete NVIDIA's [host driver and container-toolkit setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to expose the GPU.

Clone this repository into a directory named `pytorch`, open that outer directory in [Zed](https://zed.dev), then reopen it in its devcontainer:

```bash
git clone https://github.com/pupeno/pytorch-workspace.git pytorch
```

Build PyTorch:

```bash
cd /workspaces/pytorch
./build-pytorch.sh
```

## Common Commands

PyTorch has submodules of its own, pinned to exact commits the build needs. These commands are about those, and belong in `pytorch/`.

Pull PyTorch and update its nested submodules at the same time:

```bash
cd /workspaces/pytorch/pytorch
git pull --recurse-submodules
```

Update submodules after `git pull` (without `--recurse-submodules`):

```bash
cd /workspaces/pytorch/pytorch
git submodule update --init --recursive
```

Run all tests:

```bash
cd /workspaces/pytorch/pytorch
python test/run_test.py
```

Run a specific test:

```bash
cd /workspaces/pytorch/pytorch
python test/optim/test_lrscheduler.py
```

Run local experiments:

```bash
cd /workspaces/pytorch
python scratch/rop-sequential-chained-composition/00_no_scheduler_baseline.py
```
