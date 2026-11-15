---
branch: ruiz-rtdetr-fot-v2-review
slug: ruiz-rtdetr-fot-v2-review
baseline: de91a9fd707b4b19fc06f8461f34240bb6e14155   # merge-base with main; Jose's branch head is 0e7b9af
author: David Ruiz + Claude (review of Jose's RT-DETR-L-FoT-v2-adapter)
date: 2026-11-14
status: PROPOSED
verdict: ITERATE
kpi_band: PENDING   # the video stage is not part of any submission file; no guarded run applies
---

# Branch report (AI edition) — ruiz-rtdetr-fot-v2-review

## 1. Purpose & scope

A review branch of Jose's `RT-DETR-L-FoT-v2-adapter` (three commits: the video ball-detector
adapter `src/fot_video.py`, its spec `pipeline_spec_fot_video.py`, 31 tests, and the fold-0
checkpoints). Jose's branch is untouched; this branch is a local copy (`0e7b9af` + 2 commits)
that (1) records a multi-agent review of his work, (2) fixes what could be fixed without
changing behaviour, (3) adds a `training/` package that makes the detector re-trainable and
measurable on external data, and (4) ships an SOP for the agent on the RTX 4060 Ti PC.
Nothing here changes the default pipeline or `LA+AI.csv`.

## 2. Anchor

- Baseline (common ground): `de91a9f` (main). Jose's head: `0e7b9af` (`origin/RT-DETR-L-FoT-v2-adapter`, unchanged).
- Branch: `ruiz-rtdetr-fot-v2-review`, pushed to `origin` on operator instruction (2026-11-14). Remote refs checked before and after: only this branch was added; `RT-DETR-L-FoT-v2-adapter` (`0e7b9af`), `ruiz-work` (`ac9fbc1`) and `main` (`de91a9f`) unchanged.
- Working copy: `/home/ruiz-lenovo/Documents/Github/mtah2027-rtdetr-review` (a git worktree of the main clone).
- `git log --oneline de91a9f..HEAD`:
  ```
  f62d8a3 Add training/: reproduce and improve the ball detector on external data, with an SOP for the RTX PC
  65628ad Review of the RT-DETR + FoT v2 adapter: fixes, opt-in switches, gap tests
  0e7b9af Add the fold-0 model checkpoints under models/                      (Jose)
  972a797 Add CLI export to src.fot_video (notebook play_*.{mp4,csv} as a script)   (Jose)
  946325f Adapt FotV2_videotest notebook functions to the pipeline framework contract (Jose)
  ```
- This report pair (`reports/ruiz-rtdetr-fot-v2-review{.html,_AI.md}`) was reviewed and approved by the operator on 2026-11-14 and committed with the questions for Jose added.

## 3. Verification batch (real output, 2026-11-14, PC1)

```text
$ MTAH_DATA_DIR=…/more-than-a-hack-2027/data .venv/bin/python -m pytest -q
289 passed, 9 skipped, 3 xfailed in 26.84s

$ .venv/bin/python -m src.submit --out v.csv && md5sum v.csv LA+AI.csv
Valid. Note this checks the format only -- it says nothing about whether the labels are right.
aeef0058a08860c9e2407423b1172684  v.csv
aeef0058a08860c9e2407423b1172684  LA+AI.csv

$ git diff de91a9f...HEAD -- config.py | wc -l
0

# with torch 2.15.0+cpu, ultralytics 8.4.212, opencv 5.0.0.93 (scratch venv, Python 3.14.4):
$ pytest tests/test_training.py tests/test_fot_video.py tests/test_fot_video_gaps.py
206 passed, 3 xfailed

$ python -m training.compare_with_original --frames 8 --width 640 --height 480 --device cpu
"n_history_differences": 0, "n_prediction_differences": 0,
"fused_states": {"DETECTED": 5, "FOT_BRIDGED": 1, "NONE": 2}, "identical": true
```

The 9 skips are tests needing torch/cv2 in the pipeline venv; the 3 xfails pin review findings
deliberately left unchanged (§5.3).

## 4. Diff inventory

`git diff --stat de91a9f...HEAD`: 28 files, +9,022 lines (Jose's 3 commits + ours).
Our two commits:

| Area | Files | Nature |
|---|---|---|
| Video stage | `src/fot_video.py` (+431/−79), `pipeline_spec_fot_video.py` (docstring) | fixes, opt-in switches, provenance table, two helpers extracted |
| Tests | `tests/test_fot_video_gaps.py` (new), `tests/test_training.py` (new) | gap tests from the review + training package tests |
| Training | `training/*.py` (14 modules), `requirements-train.txt`, `README.md`, `SOP_RTX_AGENT.md` | new package |
| Docs / hygiene | `models/README.md`, `.gitignore` (`runs/`, `datasets/`, `play_*.csv`) | new |

Config deltas: none (`config.py` untouched; all new knobs are stage-local `fot_*` keys whose
defaults reproduce the original).

## 5. Evidence

### 5.1 Review method

Seven independent reviewers (detection & gate, FoT fusion, framework integration, compliance,
tests, training gap, performance) plus four researchers (ultralytics RT-DETR, Ubuntu 26.04 GPU
setup, dataset identity, small-ball detection literature) ran on PC1. 95 raw findings
(1 critical, 23 high, 36 medium, 24 low, 11 info) with heavy overlap. The 15 findings that drive
decisions were sent to adversarial verifiers asked to refute them (21 votes): **15/15 held**,
4 fully and 11 partially — most at a lower severity than first reported, none refuted. The run
was cut short by low RAM on PC1; 10 verifier agents did not finish (all on overlapping
compliance points already verified elsewhere). A second review of our own `training/` package
finished 2 of 5 reviewers (data path, training tools); the evaluation/FoT part was completed
by hand from the interrupted reviewer's experiments (§5.4).

### 5.2 Findings on Jose's work (verified; severity after verification)

| # | Finding | Sev. | What we did |
|---|---|---|---|
| J1 | Training data of both checkpoints is unnamed, unlicensed, unreproducible (only "games 4/6, val 9"); no training code on any ref | high | `models/README.md` marks it PENDING; `training/` rebuilds the pipeline; questions for Jose (§9) |
| J2 | ~45 tunables (15 named, ~25 inline) outside the provenance table; source notebook rendered overlays on the provided match → tuning on the provided video cannot be ruled out | high | `TUNABLE_PROVENANCE` table + test; question to Jose; SOP rail |
| J3 | CLI default rendered the detector over the provided video and wrote a per-frame CSV into the repo root (not git-ignored) | high | overlay opt-in (`--render`), outputs to ignored `runs/fot_video/`, `play_*.csv` ignored |
| J4 | Licence: checkpoint self-declares AGPL-3.0, code imports ultralytics, repo LICENSE is the organisers' MIT | medium | facts in `models/README.md`; team decision before going public |
| J5 | Kinematic gate: any in-radius candidate ≥ 0.07 pre-empts a ≥ 0.68 far hit; a static look-alike can hold the track | medium | documented + pinned (xfail); candidates kept for offline study; change needs external evidence |
| J6 | Tier mapping: real detections get 0 team confidence, bridged frames ~0.79 via the kinematic tier | medium | documented; opt-in `fot_bridged_tier="none"` |
| J7 | Decoder gate (base de91a9f) can revert a video hand-over and swallow the next segment | medium | documented in the spec; port onto `ruiz-work` before any adoption |
| J8 | Handler covers one seeded 30 s clip (0.47 % of graded frames); no full-match path | high (unverified, 3 reviewers) | documented; full-match pass is future work |
| J9 | Silent truncation on a failed video read; empty-window numpy crash | low | fixed (raises) |
| J10 | Seek-based frame positioning unverified on H.264 | low | opt-in `fot_decode="sequential"` |
| J11 | FoT bridging largely redundant with the Hermite bridge | low | opt-in `fot_use_fot=False` ablation |
| J12 | RTS smoother moves exact detections 0.6–2.5 m around kicks/traps | low | opt-in `fot_rts_enabled` / `fot_rts_*` |
| J13 | Gate-kept detections below the profile threshold never reach output | medium | diagnostic counts in `ctx.extra`; comment fixed |
| J14 | Anchor prune / spike prune docstrings promise more than the code does | low | docstrings corrected; 2-frame case pinned (xfail) |
| J15 | `age_s` counts video detections as raw observations | low | opt-in `fot_age_clock="atd"` |

Not separately verified but reported by several reviewers: unknown profile names fall back
silently (fixed), `cuda:N` index dropped (fixed), `torch.load` without `weights_only` (fixed for
FoT), duplicate tiles on tiny frames (fixed), branch is 30 commits behind the adopted
pipeline (noted), 66 MB + 5 MB binaries in plain git (noted), test suite executes 47 % of the
module and 54/69 mutants survived (gap tests added).

### 5.3 Deliberately not changed

Anything that alters tracking output (gate priority, RTS defaults, tier mapping, FoT use)
stays as Jose wrote it and is reachable only through opt-in switches. Changing the defaults
needs external-game evidence (`training.eval_detector`), not a review opinion. The three
`xfail(strict=True)` tests document J5, the 2-frame spike case and a latent out-of-pitch flag
loss; they will fail loudly the day someone fixes them, forcing the marker off.

### 5.4 Review of our own `training/` package (fixed)

| Finding | Fix |
|---|---|
| Checkpoint audit bypassable (memo-fed `STACK_GLOBAL`, `INST`) — reported SAFE on crafted files | fails closed: name allowlist (not package roots), memo-resolved operands, `INST`, extension opcodes; reviewer's 3 crafted files now NOT SAFE, real checkpoints SAFE |
| Paired comparison over-confident: null "significant" rate 15–45 % on autocorrelated frames | autocorrelation-corrected interval with a run-length floor; measured false-"better" rate 0–6 % across simulated conditions; `insufficient data` below 100 effective frames |
| Detection cache reused silently across checkpoints / strides (two `best.pt` scored as one model) | cache identity file (weights path, size, mtime, stride, cap, imgsz); mismatch is an error; comparisons on shared frames only |
| FoT "bridged steps" verdict included gaps inference never bridges | `bridgeable_mask` (≤ max_gap, anchored both sides) |
| `val_tiles` validated every checkpoint at 320 | reads each checkpoint's trained size |
| YAML guard skipped list-form / commented paths | parser handles quotes, comments, lists; fails closed |
| MOT 1-based frames → silent one-frame label shift | `frame_offset` required for frame folders; range and resolution checks |
| `--force` merged old tiles (stale labels, train/val leak) | `--force` deletes and rebuilds; refuses foreign folders |
| Hard negatives silently halved (stride mismatch) | builder errors on unvisited mined frames; SOP mines at the tile stride |
| Unpadded frame names mis-ordered; hidden files read | natural sort, hidden files skipped, size check per frame |
| Empty `convert_mot` output → all-background dataset, exit 0 | header / no-match errors; split without positives refused |
| Symlink `..` bypass of the provided-data guard | realpath of the raw path; bundle folders resolved |
| Placeholder check rejected real licence text | whole-value placeholder match |

### 5.5 Test counts

| Suite | Before (Jose) | After |
|---|---|---|
| `tests/test_fot_video.py` | 31 (27 pass + 4 skip without torch) | 31, unchanged |
| `tests/test_fot_video_gaps.py` | — | 72 (incl. 3 strict xfail) |
| `tests/test_training.py` | — | 106 |
| Whole repo, pipeline venv + data | 119 passed, 4 skipped | 289 passed, 9 skipped, 3 xfailed |

## 6. Reproduction from a clean clone

1. `git fetch origin && git switch ruiz-rtdetr-fot-v2-review` (once the operator pushes it).
2. `virtualenv -p python3 .venv && .venv/bin/pip install -r requirements-dev.txt`
3. `MTAH_DATA_DIR=<bundle> .venv/bin/python -m pytest -q` → 289 passed, 9 skipped, 3 xfailed.
4. `.venv/bin/python -m src.submit --out /tmp/v.csv && md5sum /tmp/v.csv` → `aeef0058…` (= committed `LA+AI.csv`).
5. GPU path: `training/SOP_RTX_AGENT.md`, Phases 1–6b.

## 7. Compliance log (AI-SOP-SKILL §4)

- [x] No provided-match per-frame values anywhere — none read or written; the CLI was never run on the provided video; only pass/fail and md5 crossed.
- [x] No numbers from a guarded evaluator run — none run.
- [x] No claim derived from the video or the ground truth.
- [x] Every CFG delta carries provenance — no CFG delta; stage-local tunables now carry `TUNABLE_PROVENANCE`.
- [x] Behaviour changes: none by default → no >0.02 IDSSE bar triggered; verdict ITERATE, not ADOPT.
- [x] Framework touched? No (`pipeline_spec.py`, `src/pipeline.py`, `src/steps.py`, `config.py` unchanged).
- [x] Verification batch pasted; md5 matches baseline.
- [x] Human and AI reports agree.
- [x] Operator reviewed before commit — approved 2026-11-14.

## 8. Standard-change detection

No change to `pipeline_spec.py`, `src/pipeline.py`, `src/steps.py`, `config.py`, `AI-SOP-SKILL.md`
or the pipeline standard. New stage-local keys (`fot_*`) live only in `src/fot_video.py`. No
PIPELINE_STANDARD §7 step is triggered.

## 9. Open questions / handoff

**For Jose** (sent with the human report, which carries the same list with the reason for
each; 1–3 block adoption, 4–7 block exact reproduction):

1. Training data: dataset name and link for games 4, 6, 9; licence / terms; public or private
   upload; number of games and the other folds.
2. Thresholds: on which footage the constants at the top of `src/fot_video.py` were chosen
   (0.07, 0.18, 0.30/0.40, FoT 0.45/0.72, 0.68, 0.72, gate radius and speeds, RTS noise);
   whether any value changed after watching clips of BAR-ATH.
3. Where the notebook ran (local, Kaggle, other cloud) and whether the provided video or
   homography was ever uploaded there.
4. The Kaggle notebook(s): tile builder, FoT training loop, Input list, ideally the prepared
   tiles (`tmp_fold0_tiles`); Python / torch / CUDA versions.
5. Tiles: grid or ball-centred, negatives per positive, pitch masking, label box size, meaning
   of `frame_step 2`.
6. Labels: source of the ball positions, occluded balls, camera / resolution / fps of the
   source videos.
7. FoT v2: meaning of `hit5`, losses and weights, which detector and thresholds made its
   inputs, what `best_fot_loss 1.022` measures.
8. Whether SoccerNet "H250" was used; what "FoT" stands for.

**For the operator:** the licence position before the repo goes public; whether the video stage should be ported onto
`ruiz-work` at all before 2026-11-17 (it cannot change the submission as written).

**For the RTX agent:** `training/SOP_RTX_AGENT.md` v1.1 — Phases 1–6b need no dataset (setup,
GPU smoke, acceptance, full-HD equivalence check); Phase 7 stops for the operator (dataset).
Nothing was run on a GPU yet: throughput, memory and training time are unmeasured.
