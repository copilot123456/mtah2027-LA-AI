# Stage 4 — E2 + E3 + E11 combined pass (Gate 1 read)

Pre-declared: `dossier.md` §7 (2026-11-09, commit `a516fe6`) — grid G6 (144 cells),
four decision tables {original, aligned} × {uniform, phase α=2} plus the fill-on
(E11) and α=3 twins; primary = phase × aligned; plateau rule + the > 0.02 paired
(≥ 5/7) + no-harm conditions, all applied through the P5 `sweep.flat_select` code path.

Matrices: `stage4_uniform.csv`, `stage4_phase_a2.csv`, `stage4_phase_a3.csv`,
`stage4_phase_a2_fill.csv` (144 cells × 7 matches each) + `stage4_masks.csv`.
Run: 2026-11-09, 1921 s wall, 7 workers; all 7 matches, tackles mapped 100 %
(173–229 per match, 12–23 StartFrame anchors each).

## Selection (primary table)

- plateau candidate: `switch_pen=1/dead_pen=12/sigma_m=1.5/min_dwell_s=2/switch_support_s=1` — mean **0.3699**
- frozen cell: mean **0.3667** (rank 12/144; best cell 0.3731; 23/144 within 0.01 of best)
- paired delta **+0.0032**, positive on 3/7 matches, bootstrap 95 % CI [-0.0078, +0.0179]

## The four decision tables (candidate − frozen, mean spellF1)

| table | Δ | positive |
|---|---|---|
| uniform × original | +0.0014 | 4/7 |
| uniform × aligned | +0.0032 | 5/7 |
| phase × original | +0.0044 | 4/7 |
| **phase × aligned (primary)** | **+0.0032** | 3/7 |

## Verdict — **REJECT → the frozen cell ships**

- primary delta > 0.02: **no** (+0.0032, CI straddles 0)
- paired ≥ 5/7: **no** (3/7)
- no-harm on the other three: **yes**
- delta ≤ 0.01 → reject under the pinned rule; frozen cell unchanged, submission md5 `aeef0058a08860c9e2407423b1172684` untouched.

E2's hypothesis moved the plateau in the predicted direction (lower switch
penalty, shorter support) but at noise magnitude — the gating bottleneck does
not clear the adoption bar under realistic missingness. The four tables remain
the robustness evidence for the frozen cell, as pre-declared.

## LOMO (primary; plateau on 6, report on held-out)

- fold pick identical to the global pick on **all 7 folds** (`switch_pen=1/dead_pen=12/sigma_m=1.5/min_dwell_s=2/switch_support_s=1`)
- LOMO mean 0.3699 (global-pick mean 0.3699)

## Twins (reported only, never select)

- fill-on vs fill-off (E11): candidate +0.0000, frozen +0.0000 — the fill tier adds nothing under phase masks (P5 ablation stands)
- α=3 vs α=2: candidate -0.0087, frozen -0.0040; α=3 plateau picks `switch_pen=1/dead_pen=12/sigma_m=1.5/min_dwell_s=2/switch_support_s=1` (**SAME**) 

## Mask statistics (7-match means; the simulator did its job)

| kind | masked | cue1 in/out | cue2 in/out | cue3 in/out | anycue in/out |
|---|---|---|---|---|---|
| uniform | 0.433 | 0.429/0.434 | 0.450/0.432 | 0.423/0.434 | 0.428/0.435 |
| phase_a2 | 0.432 | 0.486/0.414 | 0.506/0.429 | 0.465/0.427 | 0.475/0.410 |
| phase_a3 | 0.432 | 0.494/0.412 | 0.544/0.428 | 0.484/0.424 | 0.488/0.404 |

Frame shares: cue1 0.246, cue2 0.034, cue3 0.128. Uniform is flat (the control); phase lifts removal
onto every cue, α=3 harder than α=2, consistently across the 7 matches.

Context: E1-aligned labels lift the frozen cell by +0.1049 over the per-touch labels on the phase table.

## Reproduce

```bash
.venv/bin/python -m tuning.stage4          # full pass (~35 min, 7 workers)
.venv/bin/python -m tuning.gate1           # this read + report
.venv/bin/python -m pytest -q              # 116 green
```
