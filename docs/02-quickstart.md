# 02 · Quickstart — reproduce the submission from zero

By the end of this page you will have run the full pipeline on your machine and proven,
byte for byte, that you hold the shipping submission. Every step has a **verify
checkpoint** — run it, compare, continue. Nothing here touches the ground truth, the
scoreboard or the video.

**Prerequisites:** Linux, Python 3.14 (any modern CPython works), ~3 GB disk (data +
cache), the organiser data bundle (tracking + smart tagging; gated — never committed).
Tested on CPython 3.14.4.

## Step 1 — environment (2 min)

```bash
git clone -b ruiz-work <repo-url> more-than-a-hack-2027
cd more-than-a-hack-2027
python3 -m virtualenv .venv          # or: python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

> `python3 -m venv` needs `ensurepip`, which some distributions strip out; `virtualenv`
> works everywhere.

**Verify:**

```bash
.venv/bin/python -V        # Python 3.14.x
```

## Step 2 — the data (1 min)

The match data is gated and lives outside git. Unpack the organiser bundle and symlink it,
exactly as [`README.md`](../README.md) §Reproduce describes:

```bash
ln -s /path/to/tactical_data   data/tactical
ln -s /path/to/broadcast_data  data/broadcast
```

**Verify:**

```bash
ls -la data/          # two symlinks + README.md; no .cache yet (cold start!)
```

## Step 3 — the tests (2 min cold, 40 s warm)

```bash
.venv/bin/python -m pytest -q
```

**Verify:** `182 passed` — or `181 passed, 1 skipped` if you don't have the optional
external IDSSE data (`data/idsse/`); the skipped test is the IDSSE units check and is
designed to skip when external data is absent.

First run parses the tracking file and builds a local cache under `data/.cache/`
(~2 min, ~260 MB). The cache is an accelerator only — deleting it changes nothing in the
output.

## Step 4 — regenerate the submission (4 s warm)

```bash
.venv/bin/python -m src.submit
```

**Verify:**

```bash
md5sum LA+AI.csv
# must print: c121715836ed603fc5792b85daf6f279
```

`src.submit` prints `Valid.` (the official format validator) and a summary of the labels.
**"Valid." means the format is right, not that the labels are right.** The md5 is the
proof of reproducibility: if it matches, your machine produced exactly the file we submit.

<details>
<summary>What does the printed summary mean?</summary>

The lines after `OK` are own-output sanity statistics (never used for tuning): number of
segments and possessions, median spell length, the `D` share and the four label counts.
If your md5 matches, your summary matches ours — see [03 · how the decoder works](03-how-the-decoder-works.md)
for what produces them.

</details>

## Step 5 — the gate readers (optional, 1 min each)

Every tuning stage has a **gate**: a small program that applies the pre-declared decision
rule to the committed matrices and prints the verdict. They are read-only over
`tuning/results/` — re-running them re-reads evidence, it never re-decides:

```bash
.venv/bin/python -m tuning.gate1      # P6 grid closure — REJECT (frozen cell ships)
.venv/bin/python -m tuning.gate9      # P9 bundle — ADOPT (C + K + T)
.venv/bin/python -m tuning.gate10     # P10 K-weight — REJECT (0.3 is the knee)
.venv/bin/python -m tuning.kickoff_check   # K1 rule — PASS (amended checks)
```

## What you must NOT run

| Command | Why |
|---|---|
| `python evaluation/run_evaluation.py` on `LA+AI.csv` | Operator-only during development. Runs against the ground truth; our protocol allows one guarded run per code path, by the operator, band + flags only. |
| `python -m tuning.guarded_eval` | Same — operator-only. |
| Anything reading `evaluation/gt/`, `evaluation/scoreboard.csv`, `evaluation/reports/` | Ground truth and result boards stay closed to the development loop. |

## Next

You have reproduced the artifact. Now understand it:
**[03 · how the decoder works](03-how-the-decoder-works.md)**.
