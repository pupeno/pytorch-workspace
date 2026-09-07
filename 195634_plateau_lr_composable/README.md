# #195634 — `PlateauLR`, a composable version of `ReduceLROnPlateau`

Demos for [PR #195634](https://github.com/pytorch/pytorch/pull/195634).

Diff each `*_reduce.py` against its `*_plateau.py` to see the whole API change, and nothing else: the import, the class name, and `step(metric)` becoming `step(metrics=metric)`. Three hunks, the same three in all three pairs.

- `simple_*.py` — plateau scheduling on its own.
- `sequential_*.py` — `SequentialLR`, which runs one scheduler at a time and switches at a milestone: warmup, then hand off to the plateau scheduler.
- `chained_*.py` — `ChainedScheduler`, which runs all of them on every step so their effects multiply: `ExponentialLR` decay with plateau cuts on top.

| Script                  | On `main`                         | On `reduce-lr-on-plateau-composable-new-api` |
| ----------------------- | --------------------------------- | -------------------------------------------- |
| `simple_reduce.py`      | works                             | works, with deprecation `FutureWarning`      |
| `sequential_reduce.py`  | **`ValueError`** at construction  | **`ValueError`** at construction             |
| `chained_reduce.py`     | **`ValueError`** at construction  | **`ValueError`** at construction             |
| `simple_plateau.py`     | **`ImportError`**, no `PlateauLR` | works                                        |
| `sequential_plateau.py` | **`ImportError`**, no `PlateauLR` | works                                        |
| `chained_plateau.py`    | **`ImportError`**, no `PlateauLR` | works                                        |

The *simple* pair is the control: `PlateauLR` is a drop-in for the plain, uncomposed case, and its 40-epoch trace is identical to `ReduceLROnPlateau`'s, the only differing line is the class name in the header. `simple_reduce.py` works both before and after this PR, showing a dimension of backward compatibility (with a warning).

The *sequential* and *chained* pairs are what the PR unlocks: `ReduceLROnPlateau` is rejected at construction by both composition classes, on `main` and still on the PR branch, so `PlateauLR` is the only way in.
