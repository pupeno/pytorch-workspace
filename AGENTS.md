# Workspace layout

The project root at `/workspaces/pytorch/` is a plain directory containing two
sibling Git repositories:

- `/workspaces/pytorch/pytorch/` is the PyTorch repository. Run PyTorch Git
  commands, inspect its diffs, and make source changes there.
- `/workspaces/pytorch/workspace/` is the workspace repository. It owns the
  devcontainer and editor configuration, scripts, notes, and experiments.
- Experiment directories live directly under `/workspaces/pytorch/workspace/`.
  They are not part of the PyTorch repository unless a task explicitly says to
  move a change from an experiment into PyTorch.

`setup.sh` links configuration from the workspace repository into the project
root. The script defines which links are managed.

Keep the two repositories as siblings; do not turn the PyTorch checkout into a
submodule or nest it inside the workspace repository.
