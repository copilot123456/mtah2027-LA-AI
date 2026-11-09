# E7 — set-piece dead-ball prior: gate + D-feature audit

Pre-declared: `dossier.md` §7 (2026-11-09, commit `7388a47`) — frozen cell only,
four decision tables, primary = phase α=2 × aligned, ON value 1.0, gate
> 0.02 paired (≥ 5/7) + no-harm. OFF side = committed stage-4 frozen rows;
ON side = 14 fresh decodes with noisy IDSSE set-piece input (jitter ±1 s,
drop 25 %, +10 % spurious, seeded).

## Gate (ON − OFF, mean spellF1; OFF fresh-vs-committed drift last)

| table | Δ | positive | OFF drift |
|---|---|---|---|
| uniform × orig | -0.0092 | 0/7 | 0.000000 |
| uniform × aligned | -0.0112 | 0/7 | 0.000000 |
| phase a2 × orig | -0.0057 | 1/7 | 0.000000 |
| phase a2 × aligned | -0.0075 | 2/7 | 0.000000 |

**Verdict: REJECT → stays OFF** — primary Δ -0.0075 (> 0.02: no), positive 2/7 (≥ 5: no), no-harm: yes.

The plan's prior (+0.00 … +0.01) sits below the bar; a rejection here is the
expected outcome and leaves the frozen submission untouched.

## Tagging-noise simulation (a priori; per match raw → noisy, +spurious)

raw mean 98 (range 90–108), noisy mean 84, spurious 9 per match.

## D-feature audit (band C; phase masks, firing rate by bucket, 7-match mean)

| feature | D-correct | missed-D | false-D | base (all) | base (gt D) |
|---|---|---|---|---|---|
| out_of_pitch | 0.041 | 0.000 | 0.001 | 0.016 | 0.035 |
| stationary | 0.719 | 0.013 | 0.068 | 0.290 | 0.608 |
| near_boundary | 0.489 | 0.012 | 0.066 | 0.198 | 0.414 |
| long_missing | 0.113 | 0.530 | 0.283 | 0.192 | 0.180 |
| low_activity | 0.145 | 0.024 | 0.001 | 0.056 | 0.126 |
| set_piece | 0.083 | 0.069 | 0.017 | 0.044 | 0.081 |

Read: which features separate missed-D from D-correct frames is the external
picture of the band-C `f1_D low` flag; `set_piece`'s missed-D coverage is the
E7 upside estimate. The audit informs — the gate above is the only decider.

## Reproduce

```bash
.venv/bin/python -m tuning.e7        # this gate + audit (~3 min)
.venv/bin/python -m pytest -q
```
