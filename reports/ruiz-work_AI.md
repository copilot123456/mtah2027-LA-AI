---
branch: ruiz-work
slug: ruiz-work
baseline: de91a9f                   # merge-base with main (main locked at de91a9f — true baseline)
head: 3a7d7aa                       # at report time (16 commits on the branch)
author: operator + Copilot (LA+AI)
date: 2026-11-09
status: PROPOSED                    # shipped on the branch; team adoption Wed 2026-11-11
verdict: ITERATE                    # four pre-declared gates: Gate 1 REJECT (frozen ships), E7 REJECT (OFF), E10 retract+closure, E1 label bed shipped
kpi_band: C                         # operator-guarded run; flags: count 23-30 · f1_D low · balance normal · latency none
---

# Branch report (AI) — P6 on `ruiz-work`: four pre-declared gates, zero adoptions, frozen submission preserved

## 1. Purpose & scope

The P6 external-evidence phase for the LA+AI possession decoder, executed on `ruiz-work`
off the locked common ground: E10 (FAQ compliance rule), E1 (challenge-definition labels),
the E2+E3+E11 combined 144-cell pass (Gate 1) and E7 (set-piece dead-ball prior + the
band-C D-feature audit). Every grid, rule and gate was pinned in `dossier.md` §7 *before*
its compute; every verdict is recorded there. Scope: external evidence only (IDSSE),
no GT/video/window tuning, submission byte-identical throughout. For the team: what was
tried, what the bars said, and what ships (frozen cell + evidence + OFF-by-default features).

## 2. Anchor

- baseline: `de91a9f` (= `git merge-base main ruiz-work`; main locked), head `3a7d7aa`,
  tree clean at report time (`git status --porcelain` → 0 files).
- 16 commits, `de91a9f..HEAD`:

```
3a7d7aa E7: set-piece dead-ball prior — gate REJECT (stays OFF) + band-C D-feature audit
7388a47 dossier §7: pre-declare E7 — set-piece dead-ball prior + band-C D-feature audit
5f86440 Gate 1: REJECT — frozen cell ships; stage-4 matrices + report + gate1 reader
12f0c21 stage 4: E2 phase simulator (three IDSSE cues) + combined-pass runner + tests
a516fe6 dossier §7: pre-declare the E2+E3+E11 combined pass — grid G6, tables, simulator, rule
6ee2b24 dossier §3: log the guarded evaluator run — band C + flags, Gate 0 proceed
d131a77 E10 closed as infeasible (disposition A): window probes + closure docs; guarded-eval redacting wrapper
eb9bbb2 E1: derive_letters_controlled + statistics; fidelity probe vs plan section 0 PASSES
d0d6765 dossier: pre-declare E1 challenge-definition labels (rule + T=3 a priori)
7c347d1 Add ruiz-work branch report pair: E10 full story (gate PASS, retract, decision open)
34e76ba E10 adoption RETRACTED: effect 17x the approved envelope; frozen state restored
537b0df E10 ADOPTED: no-play rule on; LA+AI.csv regenerated (md5 0ed725f7)
ea357c7 tuning: add E10 no-harm gate runner (paired off/on, pre-declared tolerances)
2613ba8 E10: implement generic no-play rule behind no_play_rule=False (OFF)
2f5ac66 dossier: pre-declare E10 generic no-play rule + no-harm gate (before any run)
abd9e5c dossier: add organizer Q4 (video as input source; weights provenance)
```

## 3. Verification batch (real output, report time)

```
$ git status --porcelain | wc -l
0
$ git merge-base main ruiz-work
de91a9fd707b4b19fc06f8461f34240bb6e14155
$ .venv/bin/python -m pytest -q
122 passed in 24.64s
$ .venv/bin/python -m src.submit --out /tmp/vrep2.csv && md5sum /tmp/vrep2.csv
Valid. … (format validator OK)
aeef0058a08860c9e2407423b1172684  /tmp/vrep2.csv
$ git diff --stat de91a9f...HEAD | tail -1
 34 files changed, 7191 insertions(+), 12 deletions(-)
```

md5 equals the baseline's frozen value → all changes are OFF/default or external tooling;
no regeneration was needed (the E10 adopt→retract cycle restored it byte-identically at
`34e76ba`).

## 4. Diff inventory

`git diff --stat de91a9f...HEAD`: 34 files, +7 191 −12. `LA+AI.csv` not among them.

### Config deltas (key | baseline | branch | origin | evidence)

| key | baseline | branch | origin | evidence |
|---|---|---|---|---|
| `no_play_rule` | (absent) | `False` | prior, E10 | gate PASS (`no_harm_no_play.csv`) → adopt → **retract** (17× envelope) → redesign closed infeasible (§7); provenance carries the full saga |
| `no_play_s` | (absent) | `10.0` | prior, E10 | same scale as `d_missing_run_s`, a-priori |
| `d_w_set_piece` | (absent) | `0.0` (ON value 1.0 pre-declared) | prior, E7 | gate **REJECT** (primary −0.0075, 2/7); `e7_gate.csv` |
| `sp_pre_s` | (absent) | `2.0` | prior, plan §2 E7 | window [start−2 s, start+1 s], a-priori |
| `sp_post_s` | (absent) | `1.0` | prior, plan §2 E7 | same |

Behaviour-bearing defaults are unchanged; the frozen cell (switch 2 / dead 5 / σ 1.5 /
dwell 2.0 / support 1.5, fill off, X off, no_play off) is untouched.

### Code surface

- `src/emission.py` — `set_piece` dead feature + `z += d_w_set_piece·f_sp` (zero-weight
  by default; `build_emission(..., setpiece_starts=None)` default-safe); `activity` field.
- `src/data.py` — `MatchArrays.setpiece_starts` (59 `SET PIECES` tag starts).
- `src/steps.py` — one pass-through line (`setpiece_starts=ctx.arrays.setpiece_starts`).
- `src/decoder.py` — `apply_no_play()` (OFF).
- `tuning/` — `simulate.py` (3 phase cues, weighted `mask_ball`, event-row alignment),
  `stage4.py` (G6 runner), `gate1.py` (Gate-1 reader), `e7.py` (gate + audit),
  `idsse.py` (`derive_letters_controlled`), `guarded_eval.py` (redacting wrapper).
- `dossier.md` — §3 guarded-run log; §7 four pre-declarations + two gate reads.
- `pipeline_spec.py`, `src/pipeline.py`, `games.py`, `evaluation/`, `starter/`,
  `.gitignore`: **untouched**.

## 5. Evidence

### P6 band + flags (operator-guarded run; numbers never entered any AI channel)

Band **C** · flags `count 23-30 · f1_D low · balance normal · latency none` — delivered by
the operator after `tuning/guarded_eval.py` (full board archived operator-side,
`evaluation/reports/guarded_board_*.txt`). Logged in dossier §3. Gate 0: proceed.

### IDSSE artifacts (all committed)

| Artifact | Path | Result |
|---|---|---|
| E1 fidelity | `tuning/results/e1_label_stats.csv` | T=3: 2.485/min · 13.42 s · 0.034 vs plan §0 audited 2.49 · 13.4 · 0.03 — PASS on every figure; dead share invariant |
| E10 gate | `tuning/results/no_harm_no_play.csv` | PASS (mean −0.0001, worst −0.0005) → adopt → **retract** (30.9 % flips = 17× envelope); redesign probes → closed infeasible (A) |
| Gate-1 matrices | `tuning/results/stage4_{uniform,phase_a2,phase_a3,phase_a2_fill}.csv` + `stage4_masks.csv` | plateau candidate `switch1/dead12/σ1.5/dwell2/support1` 0.3699 vs frozen 0.3667; **Δ +0.0032 (3/7)**, CI [−0.0078, +0.0179]; LOMO 7/7; fill twin +0.0000; α=3 same cell → **REJECT, frozen ships** |
| Gate-1 report | `tuning/results/report_stage4.md` | full four-table + twin + mask statistics |
| E7 gate/audit | `tuning/results/e7_gate.csv`, `e7_audit.csv`, `e7_noise.csv`, `report_e7.md` | primary −0.0075 (2/7), all tables negative, OFF drift 0.000000 → **REJECT, stays OFF**; audit: missed-D = ball-absent (long_missing 53.0 % vs base 19.2 %; set_piece covers 6.9 %) |
| Simulator validation | inside the above | phase masks lift onto all three cues (anycue 0.475/0.410) vs flat uniform (0.428/0.435); TacklingGame/set-piece rows mapped 100 % on all 7 matches |

Tests: 92 (baseline) → **122**. E1 labels: `derive_letters_controlled()` + 6 tests;
aligned labels lift the frozen cell **+0.105** over per-touch labels on the phase table
(scoring use only, per §7).

## 6. Reproduction from a clean clone

1. `git clone -b ruiz-work <repo> && cd <repo> && python -m venv .venv && .venv/bin/pip install -r requirements.txt`
   (IDSSE files fetch on first use into gitignored `data/idsse/`; caches under `data/.cache/`).
2. `.venv/bin/python -m pytest -q` → 122 passed.
3. `.venv/bin/python -m src.submit --out /tmp/v.csv && md5sum /tmp/v.csv` → `aeef0058a08860c9e2407423b1172684`.
4. `.venv/bin/python -m tuning.gate1` → Gate-1 read + regenerates `report_stage4.md`.
5. `.venv/bin/python -m tuning.e7` → E7 gate + audit + `report_e7.md` (~20 min first run
   for event parsing, ~3 min warm).
6. `.venv/bin/python -m tuning.no_harm_no_play` → E10's gate (~minutes).

## 7. Compliance log (SOP §4 with results)

- [x] No provided-match per-frame values in either report — coarse aggregates only.
- [x] No guarded-evaluator numbers — band letter + categorical flags only (as delivered).
- [x] No claim derived from the video or the ground truth — evidence = IDSSE + public docs + own-output aggregates.
- [x] Every CFG delta carries provenance origin + evidence (§4 table above).
- [x] Behaviour changes > 0.02 paired on IDSSE, pre-declared §7 — or explicitly **not met → REJECT**: Gate 1 (+0.0032) and E7 (−0.0075) both rejected; E10's adoption retracted on divergence; nothing ADOPTED.
- [x] Framework touched → adoption step flagged (§8).
- [x] Verification batch pasted with real output; md5 matches baseline.
- [x] Human and AI reports agree, claim for claim.
- [x] Operator reviewed before commit (dispositions A / E7 choice / report refresh instructed by the operator).

## 8. Standard-change detection

- `pipeline_spec.py` — **not touched** (stage names, requires/provides unchanged; the new
  `setpiece` data rides inside `arrays`).
- `src/pipeline.py` — not touched. This SOP / PIPELINE_STANDARD — not touched.
- `src/steps.py` — **touched**: one pass-through line in the emission step.
- `config.py` provenance — **grew**: 5 new keys, all OFF/default, each with origin+evidence.
- **Adoption-step reminder:** per PIPELINE_STANDARD, the `steps.py` line + config keys are
  shared with the team at the Wed 2026-11-11 sync before anything merges to common ground.
  Behaviour is unchanged today (all weights zero / flags false; byte-identity proven).

## 9. Open questions / handoff

1. **Organizer Q1–Q4 open** (dossier §6). Q1 gates any future α-calibration; Q4 (video as
   input) would reopen E10's window question — until answers arrive, both stay closed.
2. **Frozen invariant:** md5 `aeef0058…`, frozen cell, all flags OFF. A second guarded
   evaluator run is allowed only if a *code path* changes before the freeze (none did).
3. **E13 queued** (detector-tier value simulation, Jose's notebooks) — needs its own §7
   pre-declaration before any run; E8/E12 discarded in the plan.
4. **Limitation candidate for the freeze write-up:** missed-D availability under missing
   ball (E7 audit: `long_missing` 53.0 % of missed-D, stationary/boundary silent);
   fixing it would need a new pre-declared > 0.02 gate — not attempted here.
5. **Band-C ordering was executed as plan §1 prescribes:** E2/E1 first (done), E3 demoted
   (folded into G6, did not clear the bar), D-audit folded into E7 (done).
6. Downstream: `python -m tuning.gate1` and `python -m tuning.e7` are the two readers;
   matrices are immutable records — never re-run selection with a different rule.
