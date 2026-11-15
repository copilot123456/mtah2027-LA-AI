---
branch: ruiz-work
slug: ruiz-work
baseline: de91a9f                   # merge-base with main (main locked at de91a9f — true baseline)
head: 3a8b67c                       # at report time (32 commits on the branch; this pair is the next commit)
author: operator + Copilot (LA+AI)
date: 2026-11-15
status: ADOPTED                     # freeze decision 2026-11-15: this state is the submission
verdict: ADOPT                      # operator: stop improving and freeze with md5 c1217158…
kpi_band: A                         # latest guarded run (operator); flags: count 14-22 · f1_D low · balance normal · latency none
---

# Branch report (AI) — `ruiz-work`: five gated phases to the freeze — P7–P9 adopted, P10 rejected, submission md5 `c1217158…`

## 1. Purpose & scope

The `ruiz-work` improvement and hardening phase for the LA+AI possession decoder, from the
locked common ground (`main` @ `de91a9f`) to the freeze. Five phases, each pre-declared in
`dossier.md` §7 before its compute and gated on external evidence only (IDSSE): P6 gates
(E1 label bed, E2+E3+E11 combined pass, E7 set-piece prior, E10 no-play saga + closure),
the 2026-11-14 design/process audit and its P7 follow-up (units fix, low-activity feature
OFF, kick-off rule ON), P8 (switch-gate move mode + team-switch delay), P9 (coalescing
min-dwell + player-only kinematic dead-ball and team terms on ball-less frames) and P10
(K weight past the grid edge — rejected). Four guarded evaluator runs (operator-only,
band + flags) passed correctness gates. Freeze state: md5 `c121715836ed603fc5792b85daf6f279`,
182 tests, clean-clone regeneration byte-identical. For the team and for any agent picking
this up: what was tried, what each bar said, and what ships.

## 2. Anchor

- baseline: `de91a9f` (= `git merge-base main ruiz-work`; main locked), head `3a8b67c`,
  tree clean at report time (`git status --porcelain` → 0 files).
- 32 commits, `de91a9f..HEAD`:

```
3a8b67c P10 stage 10: K weight past the grid edge — REJECT, 0.3 ships unchanged
a39b9b3 dossier §7: pre-declare P10 — K weight past the grid edge (fresh pin)
ac9fbc1 dossier §3: fourth guarded run — band A on md5 c1217158
2f52cb5 P9 adoption (hold released): C + K 0.3 + T 0.03, tier kin; LA+AI.csv regenerated
ff1b988 P9 stage 9: read — bundle +0.122 primary, 7/7; adoption HELD (own-output > 20 %)
8c736f2 P9 implementation, all OFF: C, K, T, S plumbing + stage-9 harness + 22 tests
c46aef8 dossier §3/§4/§7: pre-declare P9 (C, K, T, S, record cell, parity checks)
e7f1e5f dossier §3/§4: third guarded run — band A; operator statement on the idea list
235e84a P8 adoption: gate move mode (G) + team-switch delay 0.64 s (S); regenerated
44120e3 dossier §3/§4/§7: second guarded run (band B); pre-declare P8 (G, S, M)
637685c P7 K1: kick-off tag rule ON by operator decision; LA+AI.csv regenerated
1267a82 P7 adoption: low-activity feature OFF (U1); kick-off rule implemented, OFF (K1)
32593d4 P7 stage 5: fix IDSSE speed units (km/h -> m/s), dead-cover masks, baseline control
36d3945 dossier §7: pre-declare P7 audit follow-up (U0, U1, D1, K1, B1)
392ad98 dossier §3/§4: log the 2026-11-11 scoreboard-read incident (disclosed, no use)
90c602c Branch report refresh: full P6 story (E1, band C, Gate 1, E7)
3a7d7aa E7: set-piece prior — gate REJECT (OFF) + band-C D-feature audit
7388a47 dossier §7: pre-declare E7 + D-feature audit
5f86440 Gate 1: REJECT — frozen cell ships; stage-4 matrices + report + gate1 reader
12f0c21 stage 4: E2 phase simulator + combined-pass runner + tests
a516fe6 dossier §7: pre-declare the E2+E3+E11 combined pass (grid G6, tables, rule)
6ee2b24 dossier §3: log the guarded evaluator run — band C + flags, Gate 0 proceed
d131a77 E10 closed as infeasible (A): window probes + closure docs; guarded-eval wrapper
eb9bbb2 E1: derive_letters_controlled + statistics; fidelity probe PASS
d0d6765 dossier: pre-declare E1 challenge-definition labels (rule + T=3 a priori)
7c347d1 Branch report pair: E10 full story (gate PASS, retract, decision open)
34e76ba E10 adoption RETRACTED: effect 17x the approved envelope; frozen state restored
537b0df E10 ADOPTED: no-play rule on; LA+AI.csv regenerated (md5 0ed725f7)
ea357c7 tuning: E10 no-harm gate runner (paired off/on, pre-declared tolerances)
2613ba8 E10: generic no-play rule behind no_play_rule=False (OFF)
2f5ac66 dossier: pre-declare E10 generic no-play rule + no-harm gate
abd9e5c dossier: add organizer Q4 (video as input source; weights provenance)
```

## 3. Verification batch (real output, report time)

```
$ git merge-base main ruiz-work
de91a9fd707b4b19fc06f8461f34240bb6e14155
$ .venv/bin/python -m pytest -q
182 passed in 35.67s
$ .venv/bin/python -m src.submit --out /tmp/v.csv && md5sum /tmp/v.csv
Valid. … (official format validator OK)
c121715836ed603fc5792b85daf6f279  /tmp/v.csv
$ git diff --stat de91a9f...HEAD | tail -1
 92 files changed, 95674 insertions(+), 58567 deletions(-)
```

Own-output summary (sanity only, never used to select): 516 segments / 389 possessions of
mean 10.6 s · labels H 29.8 % · A 21.8 % · D 28.1 % · X 20.3 %.

md5 ≠ the baseline's frozen value (`aeef0058…`) — **intentional and documented**: three
regenerations on the branch, each committed together with the change that caused it
(P7 → `1b0cf9e4…`, P8 → `de14eb1c…`, P9 → `c1217158…`), each with its §7 pre-declaration,
gate read and green suite. Clean-clone cold-cache regeneration **PASS 2026-11-15**
(fresh clone, fresh venv, `requirements-dev.txt`, README data symlinks): tests green
(181 passed + 1 skip without the optional IDSSE data), validator PASS, byte-identical.

## 4. Diff inventory

`git diff --stat de91a9f...HEAD`: 92 files, +95 674 −58 567 (most of it `LA+AI.csv`
regeneration and tuning matrices). Outside `tuning/`, `tests/`, `reports/`, `results/`:
`LA+AI.csv` (intentionally regenerated), `config.py`, `dossier.md`, `README.md`,
`TECHNICAL_REPORT.md`, `PIPELINE_STANDARD.html` + `_AI.md`, `src/data.py`,
`src/decoder.py`, `src/emission.py`, `src/kinematics.py` + `src/kin_model.json` (new),
`src/steps.py`, `pipeline_spec.py` and `src/pipeline.py` (**md5 strings in doc text only**).
`games.py`, `evaluation/`, `starter/`, `.gitignore`: untouched.

### Config deltas (key | baseline | branch | origin | evidence)

| key | baseline | branch | origin | evidence |
|---|---|---|---|---|
| `d_w_low_activity` | `0.5` | `0.0` | external_fit, P7 U1 | units-corrected bed: 0 beats 0.5 on 8/8 tables, 7/7 (+0.025 … +0.061); §7 P7 |
| `kickoff_rule` | (absent) | `True` | documents-only rule, P7 K1 | operator decision 2026-11-14, amended first-half check (`tuning/kickoff_check.py`); < 2 % of graded labels |
| `gate_mode` | `"revert"` | `"move"` | external_fit, P8 G | unsupported boundary moves instead of merging away; §7 P8 |
| `switch_delay_s` | `0` | `0.64` | external_fit, P8 S | +0.031 external evidence; §7 P8 |
| `gap_switch_frac` | (absent) | `0.0` (ON 0.5 pre-declared) | prior, P8 G | NOT adopted — pinned maximin rule; +0.026 … +0.033 on three mask kinds only |
| `d_min_dwell_s` | (absent) | `0.0` | prior, P8 M | step dropped, no gain |
| `dwell_coalesce` | (absent) | `True` | external_fit, P9 C | +0.014 primary, 7/7; §7 P9 |
| `kin_w_dead` | (absent) | `0.3` | external_fit, P9 K | +0.017 primary, 7/7; +0.05 … +0.095 where ball loss is dead-tied; P10 confirms 0.3 is the knee |
| `kin_w_team` | (absent) | `0.03` | external_fit, P9 T | +0.091 primary, 7/7; −0.009 / −0.014 on cover_q90 inside the no-harm bound |
| `gate_kin_support` | (absent) | `True` | external_fit, P9 | kinematic term counts as switch-gate support on ball-less frames |
| `switch_delay_slope` / `_ref_ms` / `_max_s` | (absent) | `0.0` / `9.5` / `1.6` | prior, P9 S | S dropped; plumbing inert |
| `no_play_rule` / `no_play_s` | (absent) | `False` / `10.0` | prior, E10 | adopted → **RETRACTED** (17× envelope, gate vacuous) → redesign closed infeasible; provenance carries the saga |
| `d_w_set_piece` / `sp_pre_s` / `sp_post_s` | (absent) | `0.0` / `2.0` / `1.0` | prior, E7 | gate **REJECT** (−0.0075, 2/7); stays OFF |

Frozen decoder cell unchanged: switch 2 / dead 5 / σ 1.5 / dwell 2.0 / support 1.5, fill
off, X off. `pipeline_spec.py` / `src/pipeline.py` differ from baseline only in the
recorded expected-md5 strings (doc text), updated at each intentional regeneration.

### Code surface

- `src/kinematics.py` + `src/kin_model.json` (new) — player-only dead-ball and team terms
  for ball-less frames; model fitted on 7 external IDSSE matches (leave-one-match-out in
  every stage run); `tuning/kin_fit.py` is the fitter.
- `src/emission.py` — kinematic term wiring (K/T, kinematic gate tier), `set_piece` dead
  feature (weight 0), low-activity feature now weight 0.
- `src/decoder.py` — switch-gate `move` mode, coalescing min-dwell, kick-off tag rule,
  `apply_no_play()` (OFF).
- `src/steps.py` — pass-through wiring for the new emission inputs.
- `src/data.py` — tag starts (`Kick Off`, set pieces) + IDSSE-unrelated loader fixes.
- `tuning/` — `idsse.py` (units fix km/h → m/s behind a self-check, `derive_letters_controlled`),
  `simulate.py`, `stage4/5/8/9/10.py` + `gate1/5/8/9/10.py`, `kin_parity.py`,
  `kickoff_check.py`, `guarded_eval.py` (operator-only redacting wrapper).
- Tests: **92 → 182**.

## 5. Evidence

### Guarded evaluator runs (operator-only; band + flags — numbers never entered any AI channel)

| # | date | file | band | flags | covers |
|---|---|---|---|---|---|
| 1 | 2026-11-09 | `aeef0058…` | **C** | `count 23-30 · f1_D low · balance normal · latency none` | Gate 0 |
| 2 | 2026-11-14 | `1b0cf9e4…` | **B** | `count 14-22 · f1_D low · balance normal · latency none` | P7 (U1, K1) |
| 3 | 2026-11-14 | `de14eb1c…` | **A** | `count 14-22 · f1_D low · balance normal · latency none` | P8 (G, S) |
| 4 | 2026-11-14 | `c1217158…` | **A** | `count 14-22 · f1_D low · balance normal · latency none` | P9 (C, K, T) |

Every run is a correctness gate, never a selector between files; every adoption was made
on IDSSE before its run. `f1_D` stayed `low` in all four — a categorical bucket on 17
possessions, not evidence either way. No run is owed for the freeze file (P10 changed no
code path).

### IDSSE artifacts (all committed)

| Phase | Artifacts | Result |
|---|---|---|
| P6 Gate 1 | `stage4_*.csv`, `report_stage4.md` | plateau +0.0032 (3/7), CI straddles 0 → **REJECT**, frozen cell ships |
| P6 E7 | `e7_gate.csv`, `e7_audit.csv`, `report_e7.md` | −0.0075 (2/7) → **REJECT**, OFF; missed-D is ball-absent |
| P7 U0/U1/D1/K1/B1 | `stage5_*.csv`, `report_stage5.md`, `kickoff_check.py` | units fixed; low-activity **OFF** (+0.049 for off-vs-on); dead re-sweep REJECT; kick-off **ON** (operator) |
| P8 | `stage8_*.csv`, `report_stage8.md` | G **adopt** + S 0.64 **adopt**; M drop; gap placement OFF |
| P9 | `stage9_*.csv`, `report_stage9.md`, `kin_parity.py` | bundle **+0.122** primary, 7/7, +0.083 … +0.122 all 8 tables; parity Q1–Q6 PASS; own-output bucket > 20 % → **held → released by the operator → ADOPT** |
| P10 | `stage10_*.csv`, `report_stage10.md` | K > 0.3: primary +0.011 (0.4) decaying to +0.002 (1.0) → **REJECT**; 0.3 is the knee |

## 6. Reproduction from a clean clone

1. `git clone -b ruiz-work <repo> && cd <repo> && python3 -m virtualenv .venv && .venv/bin/pip install -r requirements-dev.txt`
2. Symlink the organiser bundle per `README.md` §Reproduce (`data/tactical`, `data/broadcast`).
3. `.venv/bin/python -m pytest -q` → 182 passed (181 + 1 skip without the optional `data/idsse`).
4. `.venv/bin/python -m src.submit` → `Valid.`, writes `LA+AI.csv`; `md5sum` → `c121715836ed603fc5792b85daf6f279`.

Gate readers (re-apply pinned rules, never re-select): `python -m tuning.gate1`,
`tuning.gate8`, `tuning.gate9`, `tuning.gate10`, `tuning.e7`, `tuning.kickoff_check`.
Matrices under `tuning/results/` are immutable records.

## 7. Compliance log (SOP §4 with results)

- [x] No provided-match per-frame values anywhere — coarse aggregates only.
- [x] No guarded-evaluator numbers — band letters + categorical flags only.
- [x] No claim derived from the video or the ground truth.
- [x] Every CFG delta carries provenance origin + evidence (§4 table).
- [x] Behaviour changes > 0.02 paired on IDSSE, pre-declared §7 — P7 U1, P8 G+S, P9 bundle
      all cleared their bars; Gate 1 / E7 / P8 M / P10 explicitly **not met → REJECT**.
- [x] Framework touched → adoption step flagged (§8).
- [x] Verification batch pasted with real output; md5 change intentional and documented
      (three regenerations, each committed with its cause).
- [x] Human and AI reports agree, claim for claim.
- [x] Operator reviewed before commit (freeze instruction 2026-11-15; P9 hold released by
      the operator; K1 ON by operator decision).

## 8. Standard-change detection

- `pipeline_spec.py` / `src/pipeline.py` — **behaviour untouched**; only the recorded
  expected-md5 strings (doc text) changed at each regeneration.
- `src/steps.py`, `src/emission.py`, `src/decoder.py` — behaviour-bearing (kinematic terms,
  gate move mode, coalesce, kick-off rule), all behind config keys with provenance.
- `config.py` provenance grew by 15 keys (P7/P8/P9), each with origin + evidence.
- `PIPELINE_STANDARD.html` / `PIPELINE_STANDARD_AI.md` updated alongside.
- **Adoption-step reminder:** per PIPELINE_STANDARD, the `steps.py`/`emission.py`/`decoder.py`
  changes and the new config keys must be shared with the team before anything merges to
  common ground.

## 9. Open questions / handoff

1. **Freeze checklist (operator-side):** repository made public, submission form rehearsal,
   team size confirmed (3–5), `LA+AI.pdf` (technical report, title without "(draft)"),
   push of `ruiz-work` on the operator's word → submit by 2026-11-17.
2. **Organizer Q1–Q5** (draft in `results/mtah2027/audit-2026-11-14/organizer-email-draft.md`)
   — not sent; Q1 gates any provided-match calibration, Q5 gates the "no ball → dead" variant.
3. **Limitations to carry into the write-up:** `f1_D` low in all four runs (dead-ball
   evidence is ball-absent; ~41 % of the window has no ball); the IDSSE ball is frozen
   while dead (ball-visible dead-ball features untestable on the bed); kinematic models
   fitted on another tracker's Bundesliga data — likeness checks, not transfer; IDSSE
   set-piece rows misaligned on most matches (E7's rejection is weak evidence).
4. **P10 closed:** `kin_w_dead = 0.3` is the knee; matrices + `report_stage10.md` are the
   evidence; never re-run selection with a different rule.
5. Optional future items (each needs its own §7 pre-declaration + guarded run): Smart
   Tagging restart analogue, team-phase prior (documents-only call), kinematic dead-ball
   term on ball-visible frames (needs the detector track — ball-detector branch, Phase 7
   blocked on Jose's dataset answer).
