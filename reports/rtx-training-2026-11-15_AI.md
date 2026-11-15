# RTX training run — 2026-11-15 (PC2, RTX 4060 Ti 8 GB)

Agent run of `training/SOP_RTX_AGENT.md` v1.1, Phases 1–6b. The run stopped at the
Phase 6 gate (finding below) and was resumed by the operator's decision to complete
6b plus one read-only diagnostic before reporting. Phases 7+ have not run: they need
the external dataset, which is still unknown (Jose owes the answer).

- **Machine:** `ruiz-ai-workstation-B550M-DS3H-AC-R2`, Ubuntu 26.04.1 LTS, kernel
  7.0.0-43-generic, 12 cores, 14 GB RAM, 280 GB free at `~`.
- **GPU:** NVIDIA GeForce RTX 4060 Ti, **8 GB** variant (8188 MiB), driver 595.99.02,
  CUDA 13.2 reported by nvidia-smi.
- **Branch:** `ruiz-rtdetr-fot-v2-review` @ `2340bbd` (on top of Jose's `0e7b9af`).
- **Data used:** synthetic only (`training/synth.py`, generated — no third-party
  content). Nothing from the provided match was opened, read or run.

All numbers below are printed by the commands shown. Anything not measured is
`PENDING`.

---

## 1. Phase 1 — machine probe (verbatim)

```text
$ hostname; lsb_release -ds; uname -r; nproc; free -g | head -2; df -h ~ | tail -1
ruiz-ai-workstation-B550M-DS3H-AC-R2
Ubuntu 26.04.1 LTS
7.0.0-43-generic
12
               total        used        free      shared  buff/cache   available
Mem:              14           5           6           0           3           9
/dev/nvme0n1p2  468G  165G  280G  38% /

$ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 4060 Ti, 595.99.02, 8188 MiB

$ python3 --version; command -v uv virtualenv tmux git gh
Python 3.14.4
uv MISSING
virtualenv MISSING
tmux MISSING
/usr/bin/git
/home/ruiz-ai-workstation/.local/bin/gh

$ sudo -n true && echo SUDO_OK || echo SUDO_NEEDS_PASSWORD
SUDO_NEEDS_PASSWORD
```

**Gate 1: PASS.** Driver 595.99.02 >= 580 → torch 2.15.0 CUDA wheel (Phase 3 option A).
The card is the **8 GB** variant → batch-size decisions must assume 8 GB. `tmux` is
missing (needed from Phase 9 on — long runs; operator step to install, or the runs
must be managed another way). No passwordless sudo (not needed so far).

## 2. Phase 2 — code

The repo was already cloned at `~/Documents/Github/mtah2027-LA-AI`. The local branch
`ruiz-rtdetr-fot-v2-review` was found mispointed at `ac9fbc1` (the `ruiz-work` tip —
an accidental local copy, confirmed by the operator): it shadowed the remote name and
had no `training/`. Fixed with `git switch -C ruiz-rtdetr-fot-v2-review
origin/ruiz-rtdetr-fot-v2-review` (the operator approved; nothing was lost —
`ac9fbc1` remains on `ruiz-work` locally and on origin).

**Gate 2: PASS.** Branch at `2340bbd` = `65628ad` + `f62d8a3` + report pair on top
of `0e7b9af`, tree clean, `training/SOP_RTX_AGENT.md` in the working tree.

## 3. Phase 3 — environment

`virtualenv` and `uv` are both absent (Phase 1). The built-in `python3 -m venv` was
probed read-only first and works on this box (ensurepip present, pip 25.1.1), so the
environment was created with it — **no tool was installed** and no sudo was used.
This substitutes for the SOP's `virtualenv` step; recorded here as the deviation it is.

```text
$ python3 -m venv .venv-train
$ .venv-train/bin/pip install torch==2.15.0 torchvision==0.29.2
Successfully installed ... torch-2.15.0 torchvision-0.29.2 ...
$ .venv-train/bin/pip install -r training/requirements-train.txt
Successfully installed ... ultralytics-8.4.212 opencv-python-5.0.0.93 pytest-9.2.0 ...
$ .venv-train/bin/yolo settings sync=False
```

## 4. Phase 4 — preflight and tests

```text
$ .venv-train/bin/python -m training.env_check --require-gpu
[PASS] python                       3.14.4
[PASS] numpy                        2.5.4
[PASS] pandas                       3.1.1
[PASS] torch                        2.15.0+cu132
[PASS] torchvision                  0.29.2+cu132
[PASS] cv2                          5.0.0
[PASS] ultralytics                  8.4.212
[PASS] gpu                          NVIDIA GeForce RTX 4060 Ti, 7.6 GB, CUDA runtime 13.2, torch 2.15.0+cu132
[PASS] gpu_compute                  conv forward/backward with AMP ran
[PASS] video_io                     OpenCV 5.0.0 wrote and read an MP4 (mp4v only — the codec of the real videos is proven by the first tiles/eval run on them)
[PASS] checkpoint:rtdetr_fold0_best.pt 66.2 MB, import audit clean
[PASS] checkpoint:fot_v2_fold0.pt   5.3 MB, import audit clean
[PASS] ultralytics_sync             analytics sync is off
[PASS] disk                         270 GB free at /home/ruiz-ai-workstation/Documents/Github/mtah2027-LA-AI

0 failed, 0 warnings
```

**Gate 4a: PASS.**

```text
$ .venv-train/bin/python -m pytest -q -p no:cacheprovider tests/test_training.py tests/test_fot_video.py tests/test_fot_video_gaps.py
206 passed, 3 xfailed, 7 warnings in 5.38s
```

**Gate 4b: PASS** (the 3 `xfail(strict=True)` are the pinned known defects).

## 5. Phase 5 — synthetic smoke chain

```text
$ .venv-train/bin/python -m training.synth --out data/external/synth --games 3 --frames 120
synthetic dataset written: data/external/synth/manifest.json
dataset 'synthetic-smoke' (generated by training/synth.py - no third-party content) — train games ['1', '2'], validation games ['3']
game 1 [train] 640x480, 110 annotated frames
game 2 [train] 640x480, 109 annotated frames
game 3 [val] 640x480, 112 annotated frames

$ .venv-train/bin/python -m training.tiles --manifest data/external/synth/manifest.json --out data/external/synth_tiles --train-games 1 2 --val-games 3
train: 120 frames (109 with ball) -> 218 positive, 181 background, 0 hard-negative tiles
val: 60 frames (56 with ball) -> 115 positive, 91 background, 0 hard-negative tiles
data yaml: /home/ruiz-ai-workstation/Documents/Github/mtah2027-LA-AI/data/external/synth_tiles/data.yaml
```

**Tile counts (synthetic-smoke):** train 218 positive / 181 background; val 115
positive / 91 background.

```text
$ time .venv-train/bin/python -m training.train_rtdetr --data data/external/synth_tiles/data.yaml --name smoke --profile finetune --weights models/rtdetr_fold0_best.pt --set epochs=2 --set plots=false
...
2 epochs completed in 0.008 hours.
...
results     : {"epochs_logged": 2, "best": {"epoch": 2.0, "metrics/precision(B)": 0.06676, "metrics/recall(B)": 0.03478, "metrics/mAP50(B)": 0.0098, "metrics/mAP50-95(B)": 0.00119}, "last": {"epoch": 2.0, "metrics/precision(B)": 0.06676, "metrics/recall(B)": 0.03478, "metrics/mAP50(B)": 0.0098, "metrics/mAP50-95(B)": 0.00119}}

real	0m39.867s
```

`runs/rtdetr/smoke/train_meta.json`: `"status": "finished"`. The smoke metrics are
meaningless about real balls by design (2 epochs on generated data).

```text
$ .venv-train/bin/python -m training.eval_detector --manifest data/external/synth/manifest.json --games 3 --weights models/rtdetr_fold0_best.pt runs/rtdetr/smoke/weights/best.pt --labels kaggle smoke --out runs/eval/smoke --frame-step 5
```

`runs/eval/smoke/summary.md` (verbatim, complete):

````markdown
# Detector evaluation — held-out external game(s)

- **dataset:** synthetic-smoke (generated by training/synth.py - no third-party content)
- **games:** 3
- **checkpoints:** kaggle=/home/ruiz-ai-workstation/Documents/Github/mtah2027-LA-AI/models/rtdetr_fold0_best.pt (imgsz 320), smoke=/home/ruiz-ai-workstation/Documents/Github/mtah2027-LA-AI/runs/rtdetr/smoke/weights/best.pt (imgsz 320)
- **frame step (from the caches):** 5

Top-1 detection per frame after tiling and duplicate suppression.
`recall` = frames with a ball whose top detection is within the radius; `precision` = top detections that are within the radius; `any_candidate_recall` = ceiling if the best candidate were always picked; `false_alarm_rate_empty` = frames with no visible ball that still got a detection.

## kaggle

24 frames evaluated, 23 with a visible ball.

| min_conf | radius_px | recall | precision | any_candidate_recall | false_alarm_rate_empty | median_err_px | p90_err_px |
|---|---|---|---|---|---|---|---|
| 0.07 | 5 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.07 | 10 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.07 | 20 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.18 | 5 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.18 | 10 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.18 | 20 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.3 | 5 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.3 | 10 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.3 | 20 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.4 | 5 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.4 | 10 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.4 | 20 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.5 | 5 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.5 | 10 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.5 | 20 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.68 | 5 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.68 | 10 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |
| 0.68 | 20 | 0.609 | 0.583 | 0.783 | 1.000 | 0.928 | 1.210 |

## smoke

24 frames evaluated, 23 with a visible ball.

| min_conf | radius_px | recall | precision | any_candidate_recall | false_alarm_rate_empty | median_err_px | p90_err_px |
|---|---|---|---|---|---|---|---|
| 0.07 | 5 | 0.478 | 0.458 | 0.478 | 1.000 | 4.093 | 4.676 |
| 0.07 | 10 | 0.826 | 0.792 | 1.000 | 1.000 | 4.676 | 6.976 |
| 0.07 | 20 | 0.826 | 0.792 | 1.000 | 1.000 | 4.676 | 6.976 |
| 0.18 | 5 | 0.478 | 0.524 | 0.478 | 1.000 | 4.093 | 4.676 |
| 0.18 | 10 | 0.826 | 0.905 | 0.826 | 1.000 | 4.676 | 6.976 |
| 0.18 | 20 | 0.826 | 0.905 | 0.826 | 1.000 | 4.676 | 6.976 |
| 0.3 | 5 | 0.174 | 0.667 | 0.174 | 0.000 | 4.257 | 4.872 |
| 0.3 | 10 | 0.261 | 1.000 | 0.261 | 0.000 | 4.816 | 6.974 |
| 0.3 | 20 | 0.261 | 1.000 | 0.261 | 0.000 | 4.816 | 6.974 |
| 0.4 | 5 | 0.087 | 1.000 | 0.087 | 0.000 | 4.397 | 4.844 |
| 0.4 | 10 | 0.087 | 1.000 | 0.087 | 0.000 | 4.397 | 4.844 |
| 0.4 | 20 | 0.087 | 1.000 | 0.087 | 0.000 | 4.397 | 4.844 |
| 0.5 | 5 | 0.043 | 1.000 | 0.043 | 0.000 | 3.839 | 3.839 |
| 0.5 | 10 | 0.043 | 1.000 | 0.043 | 0.000 | 3.839 | 3.839 |
| 0.5 | 20 | 0.043 | 1.000 | 0.043 | 0.000 | 3.839 | 3.839 |
| 0.68 | 5 | 0.000 | nan | 0.000 | 0.000 | nan | nan |
| 0.68 | 10 | 0.000 | nan | 0.000 | 0.000 | nan | nan |
| 0.68 | 20 | 0.000 | nan | 0.000 | 0.000 | nan | nan |

## Paired comparison (same frames, autocorrelation-corrected)

`n_eff` = effectively independent frames; below 100 no verdict is given.

| A | B | min_conf | radius_px | frames | recall B - A | 95% CI | n_eff | verdict |
|---|---|---|---|---|---|---|---|---|
| kaggle | smoke | 0.07 | 5 | 23 | -0.130 | [-0.487, +0.226] | 9 | insufficient data |
| kaggle | smoke | 0.07 | 10 | 23 | +0.217 | [-0.220, +0.654] | 9 | insufficient data |
| kaggle | smoke | 0.07 | 20 | 23 | +0.217 | [-0.220, +0.654] | 9 | insufficient data |
| kaggle | smoke | 0.18 | 5 | 23 | -0.130 | [-0.487, +0.226] | 9 | insufficient data |
| kaggle | smoke | 0.18 | 10 | 23 | +0.217 | [-0.220, +0.654] | 9 | insufficient data |
| kaggle | smoke | 0.18 | 20 | 23 | +0.217 | [-0.220, +0.654] | 9 | insufficient data |
| kaggle | smoke | 0.3 | 5 | 23 | -0.435 | [-0.765, -0.105] | 12 | insufficient data |
| kaggle | smoke | 0.3 | 10 | 23 | -0.348 | [-0.769, +0.074] | 9 | insufficient data |
| kaggle | smoke | 0.3 | 20 | 23 | -0.348 | [-0.769, +0.074] | 9 | insufficient data |
| kaggle | smoke | 0.4 | 5 | 23 | -0.522 | [-0.873, -0.170] | 8 | insufficient data |
| kaggle | smoke | 0.4 | 10 | 23 | -0.522 | [-0.873, -0.170] | 8 | insufficient data |
| kaggle | smoke | 0.4 | 20 | 23 | -0.522 | [-0.873, -0.170] | 8 | insufficient data |
| kaggle | smoke | 0.5 | 5 | 23 | -0.565 | [-0.914, -0.217] | 8 | insufficient data |
| kaggle | smoke | 0.5 | 10 | 23 | -0.565 | [-0.914, -0.217] | 8 | insufficient data |
| kaggle | smoke | 0.5 | 20 | 23 | -0.565 | [-0.914, -0.217] | 8 | insufficient data |
| kaggle | smoke | 0.68 | 5 | 23 | -0.609 | [-0.934, -0.284] | 9 | insufficient data |
| kaggle | smoke | 0.68 | 10 | 23 | -0.609 | [-0.934, -0.284] | 9 | insufficient data |
| kaggle | smoke | 0.68 | 20 | 23 | -0.609 | [-0.934, -0.284] | 9 | insufficient data |
````

All paired rows say `insufficient data` (23 graded frames, `n_eff` 8–12) — expected on
a smoke clip; no verdict is drawn from it.

**Gate 5: PASS** (all four commands exit 0; `train_meta.json` `"status": "finished"`;
`summary.md` with a table per checkpoint exists).

## 6. Phase 6 — acceptance test (real checkpoints, synthetic clip) — **GATE FAILS**

```text
$ .venv-train/bin/python -m training.smoke_inference
{
  "device": "cuda:0",
  "rtdetr_weights": ".../models/rtdetr_fold0_best.pt",
  "fot_weights": ".../models/fot_v2_fold0.pt",
  "frames": 30,
  "frame_size": [1920, 1080],
  "imgsz": 320,
  "tiles_per_frame": 32,
  "detect_seconds": 7.963,
  "frames_per_second": 3.768,
  "frames_detected": 30,
  "hit_rate": 1.0,
  "radius_px": 5.0,
  "median_error_px": 0.8204108867817739,
  "max_error_px": 1.8631422599713596,
  "fused_states": {"DETECTED": 30},
  "fused_frames": 30,
  "fused_hit_rate": 0.03333333333333333,
  "fused_median_error_px": 20.725293746978394,
  "min_hit_rate": 0.9,
  "passed": false
}
```

Run twice (seed 0) with identical numbers.

**Gate 6: FAIL.** `hit_rate` 1.0 passes (median detector error 0.82 px, max 1.86 px —
matching the PC1 reference quality). `fused_hit_rate` is 0.033 (1/30) against the
required 0.9, with `fused_median_error_px` 20.7 — while all 30 fused states are
`DETECTED` (no gaps bridged; every frame is a detector anchor that fusion was free to
move).

**Throughput (first real measurement on this card):** 7.963 s for 30 frames =
**3.77 fps** at 1920×1080, 32 tiles/frame, imgsz 320. GPU peak memory is `PENDING`
(no logger was attached for this short run; Phase 9 measures it).

**Ruled out (read-only inspection, no fixes made):**
- Not a units bug in the test: `fuse_predictions` returns pixel coordinates (its last
  step is `reproject_predictions`, "D: back to pixels") and the test compares in the
  same pixel space that gives the detector stage 0.82 px.
- Not one of the three pinned `xfail` defects (track capture, anchor count,
  `out_of_pitch` recompute) — none touches RTS/fusion drift.
- Not the environment: `env_check` clean; the detector stage of the same chain is
  pixel-accurate on the same frames.

## 7. Read-only diagnostic — RTS smoothing is the drift source (operator-approved)

The run was repeated with `fot_rts_enabled: False` in the fusion cfg (scratch script,
repo untouched):

```python
import src.fot_video as F
F.CFG = {**F.CFG, "fot_rts_enabled": False}
from training.smoke_inference import run
result = run()
```

```text
"frames_per_second": 3.779,
"frames_detected": 30,
"hit_rate": 1.0,
"median_error_px": 0.8204108867817739,
"max_error_px": 1.8631422599713596,
"fused_states": {"DETECTED": 30},
"fused_frames": 30,
"fused_hit_rate": 1.0,
"fused_median_error_px": 0.8204013812555634
```

With RTS off, the fused output equals the detector output exactly (0.82 px median,
hit rate 1.0). **The entire ~20 px drift is introduced by the RTS smoother.**
Reading of the finding (hypothesis, not measured): the smoother runs in pitch metres
with `q`/`r` constants tuned for real ball dynamics; the synthetic `ball_path` moves
faster than that motion model expects, so the smoothed track lags behind perfect
detections. On real footage the same constants may well be appropriate — that can
only be judged on an external real game (Phase 11), not on this clip.

## 8. Phase 6b — reviewed code equals Jose's original — **PASS**

```text
$ .venv-train/bin/python -m training.compare_with_original
{
  "original_ref": "0e7b9af",
  "device": "cuda:0",
  "frames": 30,
  "frame_size": [1920, 1080],
  "history_differences": [],
  "prediction_differences": [],
  "n_history_differences": 0,
  "n_prediction_differences": 0,
  "fused_states": {"DETECTED": 27, "FOT_BRIDGED": 3},
  "identical": true
}
```

**Gate 6b: PASS.** At full size (1920×1080, 30 frames, real weights) the reviewed
code's detections and fused predictions are identical to Jose's original at `0e7b9af`,
with `DETECTED` and `FOT_BRIDGED` frames both present. Consequence for §6: the fused
behavior of the chain — including the RTS drift — is the original's behavior, not
something the review introduced.

## 9. Gate verdicts

| Gate | Verdict |
|---|---|
| 1 machine probe | PASS |
| 2 code on branch | PASS (after fixing the accidentally mispointed local branch — operator-confirmed) |
| 4a preflight | PASS |
| 4b scoped tests | PASS (206 passed, 3 xfailed = pinned defects) |
| 5 synthetic smoke chain | PASS |
| 6 acceptance | **FAIL** — `fused_hit_rate` 0.033 vs 0.9; cause isolated to RTS smoothing (§7); detector stage passes cleanly |
| 6b equivalence | PASS — `"identical": true` vs Jose's original at full HD |

## 10. PENDING / not run

- Phases 7–12: blocked on the external dataset (name, licence, source) — Jose owes
  the answer (Appendix C of the SOP). Nothing was downloaded or guessed.
- Phase 13 (FoT v2 training): only on the operator's request.
- GPU peak memory in training: `PENDING` (Phase 9 probe, when a real run happens).
- **tmux is not installed on this PC** — needed for Phases 9–10 (long runs under
  `systemd-inhibit`); install is an operator step.

## 11. Decisions returned to the operator

1. **Gate 6 interpretation.** As written, the gate fails for Jose's original code too
   on this synthetic clip: the smoother's lag on a synthetic ball is baked into the
   chain the gate tests. Options: (a) treat §7 as the acceptance evidence (chain
  healthy, drift isolated and benign-by-construction on synthetic data) and let
   Phase 11 on a real external game judge the smoother; (b) change the acceptance
   clip's dynamics to be realistic; (c) keep the gate failing and treat every trained
   checkpoint's `fused_hit_rate` on the same clip as a relative, not absolute,
   measure. The SOP is the operator's document — the call is theirs.
2. **Push of this report** — committed locally on `ruiz-rtdetr-fot-v2-review`, push
   only on the operator's word.
3. `tmux` install (or an agreed alternative) before Phase 9.

Weights produced so far (not in git, per Rail 7):
`runs/rtdetr/smoke/weights/best.pt` (smoke run only — synthetic, not a candidate).
