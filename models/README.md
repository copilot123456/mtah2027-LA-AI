# models/ — what these checkpoints are

Facts read from the files themselves (static inspection, nothing unpickled:
`python -m training.inspect_checkpoint models/*.pt`) and from
`fold0_train_meta.json`. Anything not established is marked **PENDING** — it is not
known to this repository and must not be filled in by guessing.

| | `rtdetr_fold0_best.pt` | `fot_v2_fold0.pt` |
|---|---|---|
| What | RT-DETR-L ball detector, 1 class (`ball`), 320 px tiles | FoT v2 transformer (12 features → position + hit logit), window 64 |
| Format | ultralytics checkpoint (pickle; needs the `ultralytics` package to load) | bare PyTorch `state_dict` (loaded with `weights_only=True`) |
| Trained with | ultralytics 8.4.212, from `rtdetr-l.pt`, saved 2026-11-14 | PENDING (training code not in the repo) |
| Recipe | 12 epochs, batch 16, imgsz 320, AdamW lr0 1e-4, cosine, AMP, two GPUs — full list via `inspect_checkpoint`; it is the `baseline` profile of `training/train_rtdetr.py` | PENDING |
| Recorded validation (best epoch) | precision 0.732, recall 0.699, mAP50 0.646, mAP50-95 0.271 (tile level, one validation game) | `best_fot_loss` 1.022 (definition PENDING) |
| Split | train games 4 and 6, validation game 9 (`fold0_train_meta.json`) | same |
| **Training dataset** | **PENDING — name, source and licence are not recorded anywhere in this repository** | **PENDING** |
| Licence note | the file self-declares `AGPL-3.0 (https://ultralytics.com/license)` | no declaration in the file |
| Import audit | clean (only `torch` / `ultralytics` / `collections` classes) | clean (only `torch` / `collections`) |

## Before these weights may influence a submission

1. **Training data.** The technical report must name the training data, and the rules
   require that it is external to the provided match. Until the dataset is named and
   its licence recorded here, the weights are not adoptable.
2. **Licence.** The detector was produced with, and needs at run time, the AGPL-3.0
   `ultralytics` package, and the checkpoint carries an AGPL-3.0 notice; the repository's
   `LICENSE` is the organisers' MIT file and does not mention model weights. Which terms
   apply to `models/` is a decision for the team to take before the repository is made
   public. (A statement of facts, not legal advice.)
3. **Thresholds.** Several constants in `src/fot_video.py` are thresholds on these
   models' confidence outputs (`TUNABLE_PROVENANCE` there). A retrained detector
   invalidates them.

## Retraining

New checkpoints are written under `runs/` (git-ignored) by `training/`. They are not
copied here: these two files stay exactly as their author committed them, so results
can always be compared against them. To run the video stage with another checkpoint,
pass `--rtdetr-weights` / `--fot-weights` to `python -m src.fot_video`, or set
`cfg["fot_rtdetr_weights"]` / `cfg["fot_fot_weights"]`.
