# Nested learning-rate scheduler initialization experiment

This experiment runs the same composite learning-rate schedule at one and two
nesting levels and prints the results side by side. Run it against both the
`main` checkout and the PR checkout:

```bash
cd /path/to/pytorch-checkout
PYTHONPATH=. python /workspaces/pytorch/workspace/nested_lr_scheduler_initial_step/experiment.py
```

The script prints the imported `torch` path so the checkout under test can be
verified.

Before the fix, the two-level columns contain `DIFF` entries. After the fix,
every row reports `same`.
