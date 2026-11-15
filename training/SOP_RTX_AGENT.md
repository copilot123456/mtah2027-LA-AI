# SOP — set up, run and train the ball detector on the RTX PC

**Audience:** the AI agent on PC2 (`ruiz-ai-workstation`, Ubuntu 26.04, NVIDIA RTX 4060 Ti),
working for the operator (David). Humans can follow it too.
**Scope:** Jose's video ball detector (`src/fot_video.py`: tiled RT-DETR-L + FoT v2) — make it
run on the GPU, reproduce his training, and run declared experiments to get a better detector.
**Branch:** `ruiz-rtdetr-fot-v2-review` (a copy of Jose's `RT-DETR-L-FoT-v2-adapter` plus the
`training/` package and fixes). **Version:** 1.1 · 2026-11-14 (after the second review).

Read this file top to bottom once before running anything. Work phase by phase. Every phase
ends with a **gate**: a check you run and a condition that must hold. If a gate fails, stop
and report — do not improvise around it.

---

## 0. Rails — these override everything else in this file

1. **Jose's branch is read-only.** Never commit to, push to, rebase or delete
   `RT-DETR-L-FoT-v2-adapter` (local or remote). All work happens on
   `ruiz-rtdetr-fot-v2-review`. Never set that branch's upstream to Jose's branch.
2. **Push only with the operator's confirmation in the current session.** Commit locally as
   you go. No force-push, no history rewrite, ever.
3. **External data only for training and tuning.** Competition rule: nothing may be trained,
   fitted or tuned on the provided match (BAR-ATH video, tracking, Smart Tagging, ground
   truth). Every tool under `training/` refuses provided-match paths
   (`training/compliance.py`); never rename, copy or symlink provided files to get past it.
4. **Do not look at provided-match data.** You are a cloud model: provided data must not
   reach you. Never open, print, view or summarise frames, rows or per-frame values of the
   provided match, and never run `python -m src.fot_video` (it reads the provided video)
   unless the operator asks for it in this session. This SOP does not need the provided
   match at all — if `data/tactical` or `data/broadcast` are absent on this PC, that is fine.
5. **No fabricated numbers.** Report only what a command printed. A number you did not
   measure is written `PENDING`. Timings and memory use are measured here (Phase 9), never
   assumed.
6. **No sudo without the operator.** Probe with `sudo -n true`; never wait on a password
   prompt. Driver installation, `apt upgrade` and reboots are operator steps.
7. **Nothing generated goes into git.** Datasets, tiles, runs, weights and caches stay out
   (`data/*`, `runs/`, `datasets/` are git-ignored). Never commit a new `.pt` file. Never
   weaken `.gitignore`.
8. **Do not replace the committed weights.** `models/rtdetr_fold0_best.pt` and
   `models/fot_v2_fold0.pt` stay as they are. A new checkpoint lives under `runs/` until the
   operator decides what to do with it.
9. **Foreign checkpoints are audited before loading:**
   `python -m training.inspect_checkpoint <file.pt>` must print `SAFE`.
10. **One change per experiment**, and the comparison is always on a held-out *external*
    game (Phase 11). "Better" is never decided on the provided match.
11. **Irreversible actions need the operator's explicit approval** (deleting data, pushing,
    installing system packages, uploading anything anywhere).

## 0.1 What the operator must provide (you cannot continue past the gate without it)

| Needed at | What | Why |
|---|---|---|
| Phase 2 | The branch on `origin` (operator pushes it) or a git bundle | The code is not on PC2 yet |
| Phase 1, only if the GPU is not usable | sudo + reboot for the NVIDIA driver | Agent cannot do it |
| Phase 7 | The external dataset on disk, with its **name, licence and source URL** | Training data; the report must disclose it. Jose trained on "games 4 and 6, validated on game 9" of a dataset that is **not named anywhere in the repo** |
| Phase 7 (optional, better) | Jose's Kaggle training notebook(s) and/or his prepared tile dataset | Exact reproduction instead of reconstruction |

Questions to send Jose are listed in Appendix C.

---

## Phase 1 — Probe the machine (read-only, no sudo)

```bash
hostname; lsb_release -ds; uname -r; nproc; free -g | head -2; df -h ~ | tail -1
```
```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```
```bash
python3 --version; command -v uv virtualenv tmux git gh
```
```bash
sudo -n true && echo SUDO_OK || echo SUDO_NEEDS_PASSWORD
```

Record all four outputs in the run log (Phase 14).

**Gate 1.** `nvidia-smi` lists the RTX 4060 Ti and a driver version.
- Note `memory.total`: the card exists in **8 GB and 16 GB** variants; batch size decisions
  depend on it.
- Driver **>= 580** → use torch 2.15.0 (Phase 3, option A).
- Driver **525–579** → use torch 2.14.1 + cu126 (option B), or ask the operator to upgrade.
- `nvidia-smi` missing or failing → **STOP.** Tell the operator the driver is not usable and
  give them the official path: `sudo ubuntu-drivers install` then reboot, then `nvidia-smi`.
  Do not attempt it yourself.

## Phase 2 — Get the code

Convention on both PCs: repositories live under `~/Documents/Github/`.

```bash
cd ~/Documents/Github && ls
```

If `more-than-a-hack-2027` is not there:

```bash
git clone https://github.com/copilot123456/mtah2027-LA-AI.git ~/Documents/Github/more-than-a-hack-2027
```

Then:

```bash
cd ~/Documents/Github/more-than-a-hack-2027 && git fetch origin && git branch -a
```
```bash
git switch ruiz-rtdetr-fot-v2-review
```
```bash
git log --oneline -5 && git status -sb
```

**Gate 2.** `git switch` succeeded, the tree is clean, and `training/SOP_RTX_AGENT.md` exists
in the working tree. If `origin/ruiz-rtdetr-fot-v2-review` does not exist, **STOP**: the
operator has not pushed it yet (Rail 2 — it was left unpushed on PC1 on purpose).

All commands below run from the repository root.

## Phase 3 — Python environment

The stack below was validated together on CPython **3.14.4** (the system Python of both PCs):
torch 2.15.0, torchvision 0.29.2, ultralytics 8.4.212, opencv-python 5.0.0.93, numpy 2.5.4,
pandas 3.1.1. On PC1 that validation used the CPU build of torch; the CUDA build is
**UNVALIDATED until Gate 4 passes on this PC**.

Create the environment (no sudo; `python3 -m venv` may lack `ensurepip` on Ubuntu, so use
`virtualenv`, which PC1 also uses):

```bash
virtualenv -p python3 .venv-train
```

If `virtualenv` is not installed, use uv instead (`uv venv --seed .venv-train`). If neither
exists, **STOP** and ask the operator (installing a tool is their call).

Install PyTorch **first**, on its own, pinned:

- Option A — driver >= 580 (CUDA 13.2 wheel, the default on PyPI):
  ```bash
  .venv-train/bin/pip install torch==2.15.0 torchvision==0.29.2
  ```
- Option B — driver 525–579 (CUDA 12.6 wheel):
  ```bash
  .venv-train/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
  ```

The wheels bring their own CUDA runtime; only the NVIDIA driver is needed on the host. Do not
install a system CUDA toolkit.

Then the project pins:

```bash
.venv-train/bin/pip install -r training/requirements-train.txt
```

Turn off ultralytics usage analytics and its auto-installer (it would otherwise `pip install`
things by itself):

```bash
.venv-train/bin/yolo settings sync=False
```
```bash
export YOLO_AUTOINSTALL=False
```

Put that `export` in every shell (and tmux pane) you use for this work.

## Phase 4 — Preflight and tests

```bash
.venv-train/bin/python -m training.env_check --require-gpu
```

**Gate 4a.** Exit code 0 and no `[FAIL]` line. Expected lines: `gpu` PASS with the card name
and memory, `gpu_compute` PASS, `video_io` PASS, both `checkpoint:` lines PASS
("import audit clean"), `ultralytics_sync` PASS.

```bash
.venv-train/bin/python -m pytest -q -p no:cacheprovider tests/test_training.py tests/test_fot_video.py tests/test_fot_video_gaps.py
```

**Gate 4b.** All tests pass (some are reported `xfailed`: those pin known defects and are
expected). Do **not** run the whole `tests/` folder here: the rest of the suite needs the
provided match bundle, which this SOP does not use.

## Phase 5 — Smoke test of the training chain (synthetic data, a few minutes)

This proves manifest → tiles → training → evaluation on the GPU before any real data is
involved. The data is generated; its metrics mean nothing about real balls.

```bash
.venv-train/bin/python -m training.synth --out data/external/synth --games 3 --frames 120
```
```bash
.venv-train/bin/python -m training.tiles --manifest data/external/synth/manifest.json --out data/external/synth_tiles --train-games 1 2 --val-games 3
```
```bash
.venv-train/bin/python -m training.train_rtdetr --data data/external/synth_tiles/data.yaml --name smoke --profile finetune --weights models/rtdetr_fold0_best.pt --set epochs=2 --set plots=false
```
```bash
.venv-train/bin/python -m training.eval_detector --manifest data/external/synth/manifest.json --games 3 --weights models/rtdetr_fold0_best.pt runs/rtdetr/smoke/weights/best.pt --labels kaggle smoke --out runs/eval/smoke --frame-step 5
```

**Gate 5.** All four commands exit 0; `runs/rtdetr/smoke/train_meta.json` has
`"status": "finished"`; `runs/eval/smoke/summary.md` exists with a table per checkpoint.
If training fails with a CUDA out-of-memory error, add `--set batch=8` and note it.
If the loss prints `nan`, see Appendix A ("NaN loss").

## Phase 6 — Acceptance test of the real checkpoints (synthetic clip, under a minute)

Runs Jose's real inference code (`detect_window` + `fuse_predictions`) with the committed
weights on a generated clip — no match data.

```bash
.venv-train/bin/python -m training.smoke_inference
```

**Gate 6.** Exit code 0, `"passed": true`, `"device": "cuda:0"`, and both `hit_rate` (the
detector's pick per frame) and `fused_hit_rate` (the smoothed output) at least 0.9. Record
`frames_per_second` and `tiles_per_frame` (32 at 1920x1080): this is the first real
throughput measurement of inference on this card. Reference from PC1 (CPU only, 640x480
clip): the committed detector found the ball on every frame with sub-pixel to ~2 px error —
the same quality is expected here; a lower hit rate on the GPU means the environment
differs, not the model.

Use the same command with `--rtdetr <new best.pt>` on every checkpoint you train: a retrained
detector that fails this is not a drop-in replacement. The input size is read from the
checkpoint (a model trained with `--profile upsample2x` runs at 640).

## Phase 6b — Handed over from PC1: the reviewed code equals the original

PC1 has no GPU; these checks were run there only in miniature and are repeated here at full
size. They need no match data.

```bash
.venv-train/bin/python -m training.compare_with_original
```

It renders a 1920x1080 synthetic clip, runs Jose's original `src/fot_video.py` (read from
git, commit `0e7b9af`) and the reviewed one with the same committed weights, and compares
every detection and every fused prediction.

**Gate 6b.** Exit code 0 and `"identical": true`. On PC1 (CPU, 640x480, 8 frames) the result
was identical with `DETECTED`, `FOT_BRIDGED` and `NONE` frames all present. Any difference
here is a finding: stop and report the first differing frames it lists.

---

## Phase 7 — The dataset gate  (**STOP for the operator unless the data is already here**)

Ask the operator where the external dataset is. Do not download a dataset on your own guess,
and do not train on any data whose name, licence and source you cannot write down.

Research already done (PC1, 2026-11-14) — hypotheses only, **none confirmed**:
"H250" in Jose's older notes = SoccerNet_v3_H250 (YOLO-format keyframes, no game numbering —
*not* the source of the numbered games). The numbered games most plausibly come from the
SoccerNet tracking/GSR family or from SoccerTrack v2; their licences differ (several are
research / non-commercial). The answer must come from Jose.

Two ways in, depending on what the operator has:

### Path A — Jose's prepared tile dataset (exact reproduction; preferred if available)

Put it under `data/external/<name>/` (git-ignored). It must be an ultralytics detection
dataset: `images/{train,val}`, `labels/{train,val}`, and a YAML. Edit only the `path:` line
of the YAML to the new absolute location. Then write `data/external/<name>/provenance.json`
by hand with at least `dataset`, `license`, `source_url`, `train_games`, `val_games` — ask
the operator for the values; do not invent them.

Integrity check — the committed checkpoint recorded its own validation metrics; on the same
tiles it must reproduce them:

```bash
.venv-train/bin/python -m training.val_tiles --data data/external/<name>/<yaml> --weights models/rtdetr_fold0_best.pt
```

**Gate 7A.** mAP50 within about ±0.02 of the recorded 0.646 (and P ≈ 0.73, R ≈ 0.70).
A large gap means these are not the tiles the checkpoint was trained on: stop and report.

With Path A only tile-level comparisons are possible (use `training.val_tiles` in Phase 11)
unless the raw games are also available.

### Path B — raw external games (video or frame folders + ball annotations)

Write `data/external/<dataset>/manifest.json` (format in `training/manifest.py`; every field
is mandatory, including `dataset`, `license`, `source_url`; a game given as a `frames_dir`
must also state `frame_offset` — 1 for MOT-style `img1/000001.jpg` folders). Convert the
annotations to the canonical CSV `frame,x,y[,w,h]` (pixel centre of the ball) with the
converters in `training/manifest.py` (`convert_mot`, `convert_csv`). The tools refuse
annotation frame numbers outside the frame range and ball centres outside the image, which
catches the usual offset and resolution mistakes; still look at a couple of frames **of the
external game** with their label before trusting a new dataset.

```bash
.venv-train/bin/python -m training.tiles --manifest data/external/<dataset>/manifest.json --out data/external/<dataset>_tiles_fold0 --train-games 4 6 --val-games 9 --dry-run
```

**Gate 7B.** The dry run prints the dataset name and licence, the split, and per game the
frame size, the frame range, a non-zero annotated-frame count and the annotated range,
with no error.

## Phase 8 — Build the tiles (Path B only)

```bash
.venv-train/bin/python -m training.tiles --manifest data/external/<dataset>/manifest.json --out data/external/<dataset>_tiles_fold0 --train-games 4 6 --val-games 9
```

Defaults: every 2nd frame (`frame_step` 2, as recorded in `models/fold0_train_meta.json`),
every tile containing the ball as a positive, one background tile per positive. Unknown from
Jose's run and therefore assumptions here: the negative ratio, the label box size when the
dataset has none (`--box-px`, default 10), and whether his tiles were pitch-masked (ours are
not). They are recorded in `<out>/provenance.json`.

**Gate 8.** The command prints tile counts for both splits; `<out>/data.yaml`,
`tiles_index.csv` and `provenance.json` exist; positives > 0 in both splits (the builder
refuses a split without one). Record the counts. Check free disk space afterwards
(`df -h ~`). To rebuild a tile set, re-run with `--force`: it deletes that folder's tiles and
labels and writes them again (it refuses a folder `training.tiles` did not write).

## Phase 9 — Measure before you plan (throughput and memory)

Never plan from assumed timings. Run a short probe and time it:

```bash
mkdir -p runs
```
```bash
tmux new-session -d -s gpu "nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv -l 30 >> runs/gpu_log.csv"
```
```bash
time .venv-train/bin/python -m training.train_rtdetr --data <data.yaml> --name probe --set epochs=1 --set fraction=0.1 --set plots=false
```

From the probe: seconds for 10 % of one epoch → seconds per epoch → hours for a 12-epoch run
and for a 36-epoch run. Peak `memory.used` from `runs/gpu_log.csv`. Write these four numbers
in the run log. Reference only (different hardware, do not treat as a target): the original
12-epoch run took 6,240 s on two Kaggle GPUs.

**Gate 9.** You have measured s/epoch and peak memory, and the planned run fits the time the
operator has (ask if unsure). If memory is within ~1 GB of the card's total, use
`--set batch=8`.

## Phase 10 — Reproduce the baseline

Long runs go in tmux and under a sleep inhibitor:

```bash
tmux new-session -d -s train "systemd-inhibit --what=sleep:idle --why='detector training' .venv-train/bin/python -m training.train_rtdetr --data <data.yaml> --name fold0_baseline 2>&1 | tee runs/fold0_baseline.log"
```

Check on it with `tmux capture-pane -pt train | tail -5` at long intervals; do not poll in a
tight loop. If the session dies mid-run, continue with the same command plus `--resume`.

The `baseline` profile is the recipe read out of Jose's checkpoint, unchanged except for the
device. If time allows, also run `--profile baseline_bs8 --name fold0_baseline_bs8`
(structurally closer to his two-GPU run: batch 8 per GPU).

**Gate 10 — acceptance band, not equality.** RT-DETR training is not bit-reproducible on
CUDA, and one GPU is not two. Success = the run finishes and its best validation mAP50 lands
near the recorded **0.646** (precision ≈ 0.73, recall ≈ 0.70, mAP50-95 ≈ 0.27); treat a
difference of a few hundredths as seed noise. A much lower value (say below 0.55) means the
data or the tiles differ from the original — report it, do not tune around it. Results are in
`runs/rtdetr/fold0_baseline/train_meta.json`.

## Phase 11 — Evaluate on the held-out game (the only place "better" is decided)

Path B (raw games available) — full-frame evaluation through the inference code path:

```bash
.venv-train/bin/python -m training.eval_detector --manifest data/external/<dataset>/manifest.json --games 9 --weights models/rtdetr_fold0_best.pt runs/rtdetr/fold0_baseline/weights/best.pt --labels kaggle local --out runs/eval/fold0
```

Path A (tiles only):

```bash
.venv-train/bin/python -m training.val_tiles --data <data.yaml> --weights models/rtdetr_fold0_best.pt runs/rtdetr/fold0_baseline/weights/best.pt
```

Read `runs/eval/fold0/summary.md`. The rows that matter for the pipeline are
`min_conf` 0.18 and 0.30 at `radius_px` 10: recall, precision and
`false_alarm_rate_empty`. The paired-comparison table gives, per row, the recall
difference, its 95 % interval, `n_eff` (effectively independent frames — adjacent frames
are strongly correlated) and a `verdict`: `B better`, `A better`, `no evidence` or
`insufficient data` (fewer than 100 effective frames). If you get `insufficient data`,
evaluate more frames (`--frame-step 2`, or more held-out games) rather than reading the
point estimate.

Detection results are cached per checkpoint label in `--out` together with the exact
checkpoint file and settings that produced them. Re-running with a different checkpoint
under the same label, or a different `--frame-step` / `--max-frames`, stops with an error
instead of silently reusing old detections: use a new `--out` or new `--labels`. Always pass
`--labels` (two runs' `best.pt` would otherwise share the label `best`).

**Gate 11.** The verdict for `kaggle` vs `local` is `no evidence` (or `insufficient data`
with a small difference) at the rows above. That is the proof the local pipeline reproduces
Jose's result. If `kaggle` is clearly better (`A better`), stop and report before any
experiment.

The held-out game must not be in the training split of **any** checkpoint being compared.

## Phase 12 — Experiments (only after Gates 10 and 11)

One factor at a time, each against the same baseline, each evaluated with the Phase 11
command (add the new checkpoint and a label). Run them in this order and stop when the
operator's time budget is used:

| # | Command change | What it tests |
|---|---|---|
| E1 | `--profile upsample2x --name fold0_up2x` (lower `--set batch=` if memory runs out) | Resolution: each 320 px tile fed at 640 px, so a ~6 px ball covers more than one cell of RT-DETR's finest (stride-8) feature map. The literature's highest-impact lever for tiny objects; costs about 4x compute |
| E2 | `--profile no_bias_warmup --name fold0_nobias` | The recorded run started the bias learning rate at 0.1 (1000x the base rate under AdamW) |
| E3 | Hard negatives (below) → `--profile finetune --weights <baseline best.pt> --name fold0_hardneg` | False alarms on boots, spots and lines |
| E4 | `--profile no_mosaic --name fold0_nomosaic` | Mosaic was active only in the two weakest epochs and shrinks an already tiny ball |
| E5 | `--profile long --name fold0_long` | Schedule length (the recorded validation mAP was flat from epoch 5 — expect little) |
| E6 | More training games / another fold (`--fold k` in `training.tiles`), if the dataset has more games | Data diversity — the most likely real lever, per the recorded curve |

The evaluation and acceptance tools read each checkpoint's training input size, so an E1
model is automatically evaluated at 640.

E3, hard-negative mining — **on the training games only**, never on the held-out game. Mine
with the same `--frame-step` the tiles use (2); the tile builder stops with an error if mined
frames are never visited:

```bash
.venv-train/bin/python -m training.eval_detector --manifest <manifest> --games 4 6 --weights runs/rtdetr/fold0_baseline/weights/best.pt --labels local --out runs/eval/mine_train --frame-step 2
```
```bash
cat runs/eval/mine_train/candidates__local__g4.csv > runs/eval/mine_train/all.csv && tail -n +2 runs/eval/mine_train/candidates__local__g6.csv >> runs/eval/mine_train/all.csv
```
```bash
.venv-train/bin/python -m training.tiles --manifest <manifest> --out data/external/<dataset>_tiles_fold0_hn --train-games 4 6 --val-games 9 --hard-negatives runs/eval/mine_train/all.csv
```

**Decision rule (write the verdict per experiment in the run log):**
- *Better* = in the paired comparison against the baseline on the held-out game, at
  `min_conf` 0.18 **and** 0.30, `radius_px` 10: the `verdict` column says `B better`, **and**
  `false_alarm_rate_empty` did not rise.
- *No evidence* = `no evidence` or `insufficient data`. Say so; do not pick a winner on the
  point estimate.
- One held-out game is thin evidence. If the dataset has more games, confirm a "better" on a
  second fold before calling it.
- Never tune on the held-out game (no threshold picked on it, no early stopping on it beyond
  what ultralytics does on the validation split) and never mix experiments.

A better detector is **not** automatically usable by the pipeline: six thresholds in
`src/fot_video.py` and the FoT v2 weights were set against the original detector's score
scale. Run Phase 6 on the new checkpoint and report its score distribution; whether to adopt
it is the operator's and the team's decision (PIPELINE_STANDARD §7).

## Phase 13 — FoT v2 (optional, only if the operator asks)

`training/train_fot.py` is a **reconstruction**: the original training loop is not in the
repo and five assumptions (A1–A5 in its docstring) are unconfirmed. The review also found
that FoT's bridging overlaps almost entirely with the Hermite bridge downstream, so the
detector is where the effort belongs. If asked:

```bash
.venv-train/bin/python -m training.eval_detector --manifest <manifest> --games 4 6 9 --weights <detector best.pt> --labels det --out runs/eval/cands --frame-step 1
```
```bash
.venv-train/bin/python -m training.train_fot --manifest <manifest> --candidates-dir runs/eval/cands --label det --train-games 4 6 --val-games 9 --name fot_fold0 --init models/fot_v2_fold0.pt
```

`--frame-step 1` matters here: FoT is trained on consecutive detections, sub-sampled the way
inference does it. "Bridged steps" in its report are only the gaps inference would hand to
FoT (at most 4 sub-sampled steps, with a detection on both sides).

**Gate 13.** The script prints "beats the interpolation baseline on bridged steps". If it
prints "does NOT beat", the checkpoint is not an improvement — report that and stop. The
held-out game also picked the best epoch, so even a "beats" needs a second game to confirm.

## Phase 14 — Report, commit, hand back

1. Write `reports/rtx-training-<YYYY-MM-DD>_AI.md` on this branch: the Phase 1 probe output,
   package versions, the dataset identity (name, licence, source — as given by the
   operator), the tile counts, the measured timings and memory, every command you ran for
   Phases 10–12 with the resulting `train_meta.json` "results" block and the
   `summary.md` tables, and the verdict per experiment by the decision rule. Paste real
   output only; anything not measured is `PENDING`. External-game metrics are fine to
   report; nothing from the provided match may appear.
2. `git status --short` must show only that report (and nothing under `runs/`, `data/`,
   `models/`). Commit it with explicit paths.
3. **STOP and ask the operator** before `git push`. Tell them: which runs exist, where the
   best weights are (`runs/rtdetr/<name>/weights/best.pt`), the verdicts, what is still
   `PENDING`, and that weights are not in git — moving them to PC1 is their decision.
4. Update the vault project note per the vault's own conventions (the project is
   `projects/project-hackathon-mtah2027.md`; this handoff is
   `projects/handoff-mtah2027-rtx-training.md`).

---

## Appendix A — Troubleshooting

| Symptom | What to do |
|---|---|
| `CUDA out of memory` | `--set batch=8` (then 4). Record the batch you ended up with. |
| Loss becomes `nan`, or the log says "recovering from last.pt" repeatedly | Known RT-DETR + AMP risk. Start a **new** run name with `--set amp=false` (AMP cannot be changed on `--resume`); expect about twice the memory, so lower the batch. |
| Warning about `grid_sampler_2d_backward_cuda` and deterministic algorithms | Benign. RT-DETR cannot be made deterministic on CUDA. |
| `env_check` → `gpu` FAIL with a `+cpu` torch | The CPU wheel got installed. Reinstall torch with the Phase 3 command, then re-run. |
| `Failed to initialize NVML: Driver/library version mismatch` | A driver update without reboot. Operator must reboot. Never run `apt upgrade` during training. |
| Training dies with the terminal | It was not in tmux. Restart with the same command + `--resume`. |
| ultralytics tries to download `rtdetr-l.pt` and fails | The `baseline` profiles start from the public COCO weights (about 64 MB, from the ultralytics GitHub releases). If the machine is offline, tell the operator; do not substitute another file. |
| `ProvidedDataError` | You pointed a training tool at provided-match material. Stop; this is Rail 3 working. |
| `ManifestError` about `license` / `source_url` | The dataset is not documented. Ask the operator; do not fill it with a guess. |
| Dataloader crashes or hangs | `--set workers=0` to diagnose, then 2–4. |
| Disk filling up | Tiles and runs are the big items (`du -sh data/external runs`). Ask before deleting anything. |

## Appendix B — Command map

| Tool | Purpose |
|---|---|
| `python -m training.env_check --require-gpu` | preflight |
| `python -m training.inspect_checkpoint <pt>` | static audit + recorded training arguments of a checkpoint (nothing is unpickled) |
| `python -m training.synth` | synthetic smoke-test dataset |
| `python -m training.tiles` | external games → 320 px tile dataset (same grid as inference) |
| `python -m training.train_rtdetr` | training; `--profile`, `--set k=v`, `--resume`, `--dry-run` |
| `python -m training.val_tiles` | tile-level validation / dataset integrity check |
| `python -m training.eval_detector` | full-frame evaluation on held-out external games; candidates cache |
| `python -m training.smoke_inference` | real inference chain on a synthetic clip (acceptance test) |
| `python -m training.compare_with_original` | reviewed vs original `src/fot_video.py` on a synthetic clip, real models |
| `python -m training.train_fot` | FoT v2 training (reconstruction); `--runs-dir` to write elsewhere |

Profiles of `train_rtdetr` (`--dry-run` prints the resolved arguments and the reason for each
profile): `baseline`, `baseline_bs8`, `upsample2x`, `no_bias_warmup`, `no_mosaic`, `long`,
`finetune`.

Opt-in switches of the video stage (`src/fot_video.py`, all off by default, defaults reproduce
the original): `fot_use_fot`, `fot_rts_enabled` / `fot_rts_*`, `fot_bridged_tier`,
`fot_age_clock`, `fot_decode`, `fot_rtdetr_weights` / `fot_fot_weights` / `fot_imgsz`. They
exist so effects can be measured; none may be switched on for the submission without
external-game evidence and the team's decision.

## Appendix C — Questions for Jose (send through the operator)

1. Which dataset do games 4, 6 and 9 come from (exact name, URL, licence)? Public or a private
   upload? How many games are there and what are the other folds?
2. Can you share the Kaggle notebook(s) — at least the tile-building code and the FoT v2
   training loop — and, if possible, the prepared tile dataset (`tmp_fold0_tiles`)?
3. How were tiles sampled (grid or ball-centred, negatives per positive, pitch-masked or not)
   and what label box size was used for the ball?
4. Where do the ball labels come from (human boxes, interpolated tracks, pseudo-labels)?
5. What is `hit5` in FoT v2, what were the losses, and how were the training inputs made
   (which detector, which thresholds)?
6. The constants at the top of `src/fot_video.py` ("the notebook's values") — on which
   footage were they chosen? The rules require tuning on external data only.
