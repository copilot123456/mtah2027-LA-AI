# Put your submissions here

Every `*.csv` in this folder is scored by `run_evaluation.py`, and the file name becomes the name
on the board — so call them something you will recognise.

```bash
python evaluation/run_evaluation.py         # scores whatever is in here
python starter/baseline.py                  # regenerates the four baselines
python starter/make_dummy_submission.py     # three trivial ones, to see the loop work
```

**Closing every iteration (amended 2026-11-11, GT freeze):** close with the GT-free
diagnostics (`evaluation/gt_free_diagnostics.py`) against the IDSSE reference — NOT with the
public scoreboard. Looking at public-GT numbers each cycle and picking the next fix from them
is selection on the GT even when no parameter is calibrated on it. The public-GT comparison
against `baseline_smooth50` (the published floor) is the FINAL report: run it then, print the
head-to-head with deltas, report `macroF1` and `spellF1` both — but its numbers never choose
the next objective. Objectives come from GT-free diagnostics; GT is for evaluation only.

## What is already here

Four baselines, committed on purpose. They are the output of `starter/baseline.py` — nearest
tracked player to the ball, then a majority vote over a window of `±0`, `±25`, `±50` and `±125`
frames. `baseline_smooth50` is the published floor.

They are worth scoring before you write anything, because the four differ in exactly one number
and the board still reorders them. That is the clearest demonstration in the repository of what
`spellF1` measures and `macroF1` does not — see [EVALUATION.md](../../EVALUATION.md).

## Why these are committed and yours will not be

`*.csv` here is **gitignored**, with a single exception for `baseline_*.csv`.

A submission is a complete set of labels for the match, and a correct one is a copy of the ground
truth. Nothing that belongs in this folder belongs in a public repository, and your own working
files will not end up in a commit.

The baselines are the exception because of how they are made, not because they are ours: they read
the ATD and nothing else — never a ground truth, public or private — so they cannot carry
information about the answer.
