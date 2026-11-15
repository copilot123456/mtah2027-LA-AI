# training/ — reproduce and improve the ball detector

Everything needed to train the RT-DETR-L tile detector behind `src/fot_video.py` on a
local GPU, and to tell honestly whether a new checkpoint is better than the committed
one. Nothing here is used by the default pipeline or by `python -m src.submit`.

**Running this on the RTX PC?** Follow [`SOP_RTX_AGENT.md`](./SOP_RTX_AGENT.md) — it is
the step-by-step runbook with gates. This page is the map.

## The one rule

Training and tuning use **external data only** — never the provided match. Every tool
here refuses provided-match paths (`compliance.py`), and a dataset cannot be used
until its name, licence and source are written in a manifest (`manifest.py`).

## What is known about the original training — and what is not

| | Status |
|---|---|
| RT-DETR recipe (model, epochs, batch, lr, augmentation, …) | **Recovered** from the checkpoint itself: `python -m training.inspect_checkpoint models/rtdetr_fold0_best.pt`. The `baseline` profile is that recipe. |
| FoT v2 architecture | **Confirmed**: the committed weights load strictly into `build_fot_model`. |
| The dataset ("games 4 and 6, validated on game 9") | **Unknown** — not named anywhere in the repo. Must come from the author. |
| Tile-building code, negative ratio, label box size, pitch masking | **Unknown** — `tiles.py` is a documented reconstruction; its assumptions are written into each dataset's `provenance.json`. |
| FoT v2 training loop, loss, meaning of `hit5` | **Unknown** — `train_fot.py` is a reconstruction with five listed assumptions. |

## Tools

| Command | What it does |
|---|---|
| `python -m training.env_check --require-gpu` | preflight: packages, GPU compute, video I/O, checkpoint audit |
| `python -m training.inspect_checkpoint <pt>` | static audit and recorded training arguments (nothing is unpickled) |
| `python -m training.synth --out …` | synthetic games for smoke tests |
| `python -m training.tiles --manifest … --out …` | external games → 320 px tiles on the inference grid |
| `python -m training.train_rtdetr --data … --name …` | train; `--profile`, `--set k=v`, `--resume`, `--dry-run` |
| `python -m training.val_tiles --data … --weights …` | tile-level validation; dataset integrity check |
| `python -m training.eval_detector --manifest … --games … --weights …` | full-frame evaluation on held-out external games, paired comparison |
| `python -m training.smoke_inference [--rtdetr …]` | the real inference chain on a synthetic clip (acceptance test) |
| `python -m training.compare_with_original` | the reviewed `src/fot_video.py` vs the author's original, real models, synthetic clip |
| `python -m training.train_fot …` | FoT v2 training (reconstruction) |

## Typical flow

```bash
python -m training.env_check --require-gpu
python -m training.tiles --manifest data/external/<set>/manifest.json \
    --out data/external/<set>_tiles_fold0 --train-games 4 6 --val-games 9
python -m training.train_rtdetr --data data/external/<set>_tiles_fold0/data.yaml --name fold0_baseline
python -m training.eval_detector --manifest data/external/<set>/manifest.json --games 9 \
    --weights models/rtdetr_fold0_best.pt runs/rtdetr/fold0_baseline/weights/best.pt \
    --labels kaggle local --out runs/eval/fold0
```

"Better" is decided by `eval_detector` on a game no compared checkpoint was trained on:
recall and precision of the single detection the pipeline keeps per frame, within a
pixel radius of the annotated ball. The interval on the difference is corrected for
autocorrelation between neighbouring frames, and with fewer than 100 effectively
independent frames the verdict is `insufficient data`. Tile-level mAP50-95 is the wrong
yardstick for a ball a few pixels wide.

## Tests

`tests/test_training.py` covers the planning, scoring, validation and guard logic with
numpy and pandas only; tests that need torch or OpenCV skip themselves where those are
absent. Dependencies for the GPU path: `training/requirements-train.txt`.
