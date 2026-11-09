---
branch: ruiz-work
slug: ruiz-work
baseline: de91a9f                   # merge-base with main (main locked at de91a9f — true baseline)
head: 34e76ba                       # at report time (6 commits on the branch)
author: operator + Copilot (LA+AI)
date: 2026-11-09
status: PROPOSED                    # shipped on the branch; team adoption Wed 2026-11-11
verdict: ITERATE                    # E10 rule: gate passed, adoption retracted, redesign closed infeasible (A)
kpi_band: PENDING                   # no guarded evaluator run yet — categorical only, never numbers
---

# Branch report (AI) — E10 no-play rule: gate PASS, adoption RETRACTED, redesign closed as infeasible (A)

## 1. Purpose & scope

First experiment on `ruiz-work`: execute **E10** from the P6 backlog — a generic rule that
labels FAQ-documented pre-kick-off frames (and any sustained no-play stretch) as `D`, fixing
the measured 37 % deficiency (plan §0, 2026-11-07). Scope: own-output compliance against a
public doc fact; no parameter tuning; no GT/video.

## 2. Anchor

- **Baseline (common ground):** `de91a9f` — main, locked by team agreement until the common
  solution is pushed.
- **Head:** `34e76ba` — 6 commits, all on `ruiz-work`.
- **Net state vs baseline: 8 files, +286/−2 — `LA+AI.csv` byte-identical**
  (adopt-then-retract restored the frozen output; README/PIPELINE md5 references net to zero).
- Clean tree at verification.

### Commit trail

| Commit | What |
|---|---|
| `abd9e5c` | dossier §6: organizer Q4 (video-as-input-source; weights provenance) |
| `2f5ac66` | dossier §7: E10 rule + gate tolerances pre-declared (**before any implementation or run**) |
| `2613ba8` | implementation behind `no_play_rule=False` + 5 tests (97 total) |
| `ea357c7` | gate runner `tuning/no_harm_no_play.py` (post_checks --final semantics) |
| `537b0df` | gate PASS → adopted: flag ON, CSV regenerated (md5 `0ed725f7…`) |
| `34e76ba` | **adoption RETRACTED**: effect measured at 17× the approved envelope; frozen state restored |

## 3. Verification batch (real output, 2026-11-09)

```text
$ .venv/bin/python -m pytest -q
97 passed in 21.99s

$ rm -f /tmp/v.csv && .venv/bin/python -m src.submit --out /tmp/v.csv && md5sum /tmp/v.csv
Valid. Note this checks the format only -- it says nothing about whether the labels are right.
aeef0058a08860c9e2407423b1172684  /tmp/v.csv      # == committed == baseline: byte-identical
```

## 4. The E10 record (all four stages)

### 4.1 Pre-declaration (dossier §7, `2f5ac66` — before code, before the run)

Rule: inside `decode()`, after gating + min-dwell, mark a frame `D` iff it lies in a maximal
run of ≥ `no_play_s` = 10.0 s (a priori, same scale as `d_missing_run_s`) of consecutive
frames with per-frame mean player speed < `d_activity_thr_ms` = 2.0 (existing prior). Per
decoded half; no frame numbers, no match identifiers, no ball-position conditions (DATA.md:
camera/homography unreliable across half-time). Gate: IDSSE frozen cell, 7 matches, paired
off/on, same seeds; **pass iff mean Δ ≥ −0.005 AND no match < −0.02**.

### 4.2 Gate result (`tuning/results/no_harm_no_play.csv`): **PASS**

```text
harness sanity: max |rule-off - committed post_checks_final baseline| = 0.000000
paired Δ (on − off): 6 matches +0.0000 | J03WPY -0.0005
mean -0.0001 | worst -0.0005  →  gate: PASS
```

### 4.3 Measured effect after adoption (`537b0df`) — the reason for retraction

```text
rows changed: 49,165 of 199,319 (24.7%) — all within graded frames, ALL H/A → D
changed within graded: 49,165 of 158,857 (30.9%)   [half1 24,723 | half2 24,442]
transitions: H→D 26,293 | A→D 22,872
D share on graded: 26.6% → 57.5% (+30.9 pp)   [public window dead share: 19.6%]
activity structure: mean player speed < 2 m/s on 65.6% (half1) / 67.2% (half2) of frames;
  qualifying ≥10 s runs: 50 (half1, 40,143 frames) + 52 (half2, 41,556 frames)
```

- Plan §2 E10 envelope: "**117 s of one half** … ≈1.8 % of graded … 0 … +0.005".
  Actual: **30.9 % of graded — 17× the envelope.**
- Cause: `mean(player speeds) < 2 m/s` is the **normal state of live football** (most of 22
  players walk at any instant) — fine as one weak sigmoid feature (weight 0.5), a D-hammer as
  a hard match-wide rule.
- **Why the gate couldn't see it:** the rule fires on 0–1 of 7 IDSSE matches (different speed
  distributions) — a *vacuous* no-harm pass. **Methodological learning: future no-harm gates
  must check external firing rates** (does the rule fire on external data at all, and at a
  comparable rate?) before "no harm" means anything.

### 4.4 Retraction (`34e76ba`)

`no_play_rule` → False; `LA+AI.csv` regenerated → byte-identical to baseline
(`aeef0058a08860c9e2407423b1172684`); live md5 references restored; provenance records the
full story. Evidence kept: pre-declaration, runner, results CSV, tests (rule remains
implemented and tested, OFF).

## 5. Evidence

| Claim | Evidence | Artifact |
|---|---|---|
| FAQ fact | reference-docs/FAQ.md: frames 110,808–113,741 "graded and all of them are `D`" | docs quote, dossier §7 |
| Baseline deficiency | own output = 37 % `D` on the window (measured 2026-11-07, plan §0) | plan §0 |
| Rule is generic | no frame ids/match ids/ball-position conditions; unit tests on synthetic runs | `tests/test_no_play.py` |
| No-harm gate | IDSSE 7 matches paired, mean −0.0001, worst −0.0005, sanity drift 0.000000 | `tuning/results/no_harm_no_play.csv` |
| Effect magnitude | 49,165 flips, D share +30.9 pp, all H/A→D | measured on the adopted commit `537b0df` |
| Frozen state restored | fresh run == committed == baseline md5 | verification batch above |
| P6 KPI | `PENDING` — band letter + flags arrive from the operator | — |

## 6. Reproduction from a clean clone

```bash
.venv/bin/python -m pytest -q                       # 97 passed
.venv/bin/python -m src.submit --out /tmp/v.csv && md5sum /tmp/v.csv
# aeef0058a08860c9e2407423b1172684
.venv/bin/python -m tuning.no_harm_no_play          # re-runs the gate (minutes)
```

## 7. Compliance log

- [x] No provided-match per-frame values in this report — aggregates/counts only.
- [x] No guarded-evaluator numbers — `kpi_band: PENDING`; no run performed.
- [x] No GT/video-derived claim — FAQ is a public doc; effects measured on our own output;
      gate numbers are committed external (IDSSE) results.
- [x] Every CFG delta carries provenance (`no_play_rule`, `no_play_s` — origin `prior`).
- [x] Behaviour changes went through the pre-declared external gate (record in §7) — and the
      adoption was retracted when the measured effect left the operator-approved envelope.
- [x] Frame-specific rule never used (plan ban honoured).
- [x] Forbidden paths untouched; no `data/*`; no history rewrite; tests green after every
      commit (92 → 97).
- [x] Human ↔ AI reports agree, claim for claim.
- [x] Operator reviewed the loop (design_now approval 2026-11-09); disposition A taken the
      same day — redesign attempted, probes showed the sanctioned signals absent from the
      window, E10 closed as infeasible under the rules (dossier §7 closure note).

## 8. Standard-change detection

This branch did **not** change the pipeline framework contract (`src/pipeline.py`,
`pipeline_spec.py`, `src/steps.py` — untouched). It touched `dossier.md` (§6 Q4, §7
pre-declaration) and — during the adopted window — refreshed md5 references in README and
PIPELINE_STANDARD (restored by the retraction; net zero). The **methodological addition**
(firing-rate parity for no-harm gates) is proposed for the P6 plan via the vault, not a
framework change.

## 9. Open questions / handoff

1. **Operator decision (2026-11-09): disposition A — E10 closed as infeasible under the
   allowed generic signals.** The kickoff-geometry redesign was probed read-only (no
   GT/video): ball or kicker within 1.5 m of the centre spot on 0.0 % of the window,
   formation milling (29.9 m median displacement vs 36.3 m live), four-clause conjunction
   0.0 % — the window is warm-up, not a formed-up restart, and the plan §2 sanctioned rule
   has no firing opportunity in it. Flag stays OFF; 37 % stays documented; no gate run.
   Remaining honest routes: accept as-is, E7's tag/timing turf, or an organizer answer.
2. **Gate-methodology fix (queued):** require external firing-rate evidence in future no-harm
   gates — proposed as a P6 plan note.
3. E1 (challenge-definition labels, T = 3 s a priori) is next per operator instruction; P6
   band + flags, E1 rule T — now approved by "proceed with E1"; organizer Q1–Q4 remain owed.
4. First real `pipeline_spec.py` handler swap by a team member has not happened yet —
   Wednesday 2026-11-11 is the stress test.
