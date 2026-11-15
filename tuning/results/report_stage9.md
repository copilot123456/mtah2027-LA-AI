# Stage 9 — P9 pass (C coalescing min-dwell, K / T kinematic terms, S switch delay)

Pre-declared: `dossier.md` §7 P9 (2026-11-14). Shipped configuration as reference, four
mask kinds × two label sets, 7 IDSSE matches; kinematic models fitted leave-one-match-out,
kinematic features of the test match from dropout-degraded player tracks. Δ = 7-match
mean of paired spellF1 differences; (n/7) = matches where the difference is positive.

Harness pin: the reference cell reproduces the stage-8 shipped rows, max |Δ| = 0.0e+00.

## C — coalescing min-dwell (against: coalesce off · K 0 · T 0 · tier ball · delay 0.64 s · slope 0)

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| coalesce on | +0.0132 (7/7) | +0.0081 (6/7) | +0.0146 (7/7) | +0.0143 (7/7) | +0.0183 (7/7) | +0.0142 (6/7) | +0.0157 (7/7) | +0.0114 (7/7) |

- candidate (maximin over the four mask kinds, aligned labels): **coalesce on** — g = +0.0081 (best +0.0081)
- primary (phase_a2 × aligned): **+0.0143**, positive 7/7 (needs > 0.005 and ≥ 5/7) · no-harm on the other seven tables: **yes**
- **Step verdict: KEEP**

## K — kinematic dead-ball term (weight × gate tier) (against: coalesce on · K 0 · T 0 · tier ball · delay 0.64 s · slope 0)

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| K 0.03 | +0.0000 (0/7) | +0.0000 (0/7) | -0.0002 (0/7) | -0.0003 (0/7) | +0.0004 (1/7) | +0.0005 (1/7) | +0.0004 (1/7) | +0.0010 (2/7) |
| K 0.1 | -0.0001 (1/7) | -0.0000 (1/7) | +0.0004 (3/7) | +0.0014 (2/7) | +0.0044 (5/7) | +0.0070 (6/7) | +0.0069 (6/7) | +0.0095 (7/7) |
| K 0.3 | +0.0010 (4/7) | +0.0008 (5/7) | +0.0025 (5/7) | +0.0039 (5/7) | +0.0117 (7/7) | +0.0158 (7/7) | +0.0196 (7/7) | +0.0244 (7/7) |
| K 0.03 · tier kin | +0.0000 (0/7) | +0.0000 (0/7) | -0.0002 (0/7) | -0.0003 (0/7) | +0.0004 (1/7) | +0.0005 (1/7) | +0.0004 (1/7) | +0.0010 (2/7) |
| K 0.1 · tier kin | +0.0026 (6/7) | +0.0031 (5/7) | +0.0012 (5/7) | +0.0020 (6/7) | +0.0201 (7/7) | +0.0296 (7/7) | +0.0314 (7/7) | +0.0417 (7/7) |
| K 0.3 · tier kin | +0.0122 (7/7) | +0.0124 (7/7) | +0.0143 (7/7) | +0.0167 (7/7) | +0.0505 (7/7) | +0.0669 (7/7) | +0.0776 (7/7) | +0.0946 (7/7) |

- candidate (maximin over the four mask kinds, aligned labels): **K 0.3 · tier kin** — g = +0.0124 (best +0.0124)
- primary (phase_a2 × aligned): **+0.0167**, positive 7/7 (needs > 0.005 and ≥ 5/7) · no-harm on the other seven tables: **yes**
- **Step verdict: KEEP**

## T — kinematic team term (weight × gate tier) (against: coalesce on · K 0.3 · T 0 · tier kin · delay 0.64 s · slope 0)

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| T 0.03 · tier ball | -0.0082 (1/7) | +0.0016 (3/7) | -0.0090 (2/7) | -0.0029 (3/7) | -0.0372 (0/7) | -0.0471 (0/7) | -0.0650 (0/7) | -0.0811 (0/7) |
| T 0.1 · tier ball | -0.0111 (1/7) | -0.0023 (3/7) | -0.0160 (1/7) | -0.0105 (2/7) | -0.0380 (0/7) | -0.0486 (0/7) | -0.0652 (0/7) | -0.0815 (0/7) |
| T 0.3 · tier ball | -0.0132 (1/7) | -0.0062 (1/7) | -0.0180 (1/7) | -0.0147 (2/7) | -0.0381 (0/7) | -0.0487 (0/7) | -0.0624 (0/7) | -0.0783 (0/7) |
| T 0.03 | +0.0581 (7/7) | +0.0904 (7/7) | +0.0556 (7/7) | +0.0910 (7/7) | +0.0189 (7/7) | +0.0339 (7/7) | -0.0088 (1/7) | -0.0136 (1/7) |
| T 0.1 | +0.0627 (7/7) | +0.0903 (7/7) | +0.0665 (7/7) | +0.0930 (7/7) | +0.0214 (7/7) | +0.0292 (7/7) | -0.0087 (0/7) | -0.0162 (1/7) |
| T 0.3 | +0.0687 (7/7) | +0.0925 (7/7) | +0.0751 (7/7) | +0.0925 (7/7) | +0.0275 (7/7) | +0.0343 (7/7) | -0.0036 (2/7) | -0.0130 (1/7) |

- candidate (maximin over the four mask kinds, aligned labels): **T 0.03** — g = -0.0136 (best -0.0130)
- primary (phase_a2 × aligned): **+0.0910**, positive 7/7 (needs > 0.005 and ≥ 5/7) · no-harm on the other seven tables: **yes**
- **Step verdict: KEEP**

## S — team-switch delay (against: coalesce on · K 0.3 · T 0.03 · tier kin · delay 0.64 s · slope 0)

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| delay 0.8 s | +0.0014 (5/7) | +0.0013 (6/7) | +0.0009 (5/7) | +0.0007 (3/7) | +0.0018 (5/7) | +0.0028 (6/7) | +0.0034 (7/7) | +0.0026 (7/7) |
| delay 0.96 s | +0.0015 (6/7) | +0.0023 (6/7) | +0.0002 (2/7) | +0.0003 (3/7) | +0.0037 (5/7) | +0.0049 (6/7) | +0.0047 (7/7) | +0.0047 (6/7) |
| delay 1.12 s | +0.0010 (6/7) | +0.0026 (5/7) | -0.0002 (3/7) | +0.0015 (3/7) | +0.0043 (6/7) | +0.0058 (6/7) | +0.0060 (6/7) | +0.0069 (6/7) |
| slope 0.03 | +0.0019 (7/7) | +0.0018 (5/7) | +0.0023 (7/7) | +0.0025 (7/7) | +0.0026 (6/7) | +0.0027 (5/7) | +0.0038 (7/7) | +0.0026 (7/7) |
| slope 0.06 | +0.0038 (6/7) | +0.0041 (6/7) | +0.0031 (7/7) | +0.0036 (7/7) | +0.0056 (7/7) | +0.0057 (6/7) | +0.0066 (7/7) | +0.0058 (6/7) |
| delay 0.8 s · slope 0.03 | +0.0037 (6/7) | +0.0031 (5/7) | +0.0026 (5/7) | +0.0022 (5/7) | +0.0054 (7/7) | +0.0056 (7/7) | +0.0085 (7/7) | +0.0072 (7/7) |
| delay 0.8 s · slope 0.06 | +0.0049 (6/7) | +0.0048 (6/7) | +0.0036 (5/7) | +0.0041 (7/7) | +0.0082 (7/7) | +0.0081 (6/7) | +0.0119 (7/7) | +0.0105 (7/7) |
| delay 0.96 s · slope 0.03 | +0.0047 (6/7) | +0.0050 (7/7) | +0.0022 (5/7) | +0.0022 (6/7) | +0.0069 (6/7) | +0.0074 (6/7) | +0.0098 (7/7) | +0.0097 (7/7) |
| delay 0.96 s · slope 0.06 | +0.0058 (6/7) | +0.0068 (7/7) | +0.0028 (5/7) | +0.0044 (6/7) | +0.0100 (6/7) | +0.0106 (6/7) | +0.0137 (7/7) | +0.0131 (6/7) |
| delay 1.12 s · slope 0.03 | +0.0033 (6/7) | +0.0048 (7/7) | +0.0017 (6/7) | +0.0026 (5/7) | +0.0083 (6/7) | +0.0092 (6/7) | +0.0112 (7/7) | +0.0113 (6/7) |
| delay 1.12 s · slope 0.06 | +0.0057 (6/7) | +0.0069 (7/7) | +0.0034 (6/7) | +0.0049 (7/7) | +0.0112 (6/7) | +0.0111 (6/7) | +0.0147 (7/7) | +0.0139 (7/7) |

- candidate (maximin over the four mask kinds, aligned labels): **delay 0.8 s** — g = +0.0007 (best +0.0049)
- primary (phase_a2 × aligned): **+0.0007**, positive 3/7 (needs > 0.005 and ≥ 5/7) · no-harm on the other seven tables: **yes**
- **Step verdict: DROP — the step keeps its current value**

## Bundle (the cell that would ship) against the reference

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| coalesce on · K 0.3 · T 0.03 · tier kin · delay 0.64 s · slope 0 | +0.0834 (7/7) | +0.1110 (7/7) | +0.0845 (7/7) | +0.1220 (7/7) | +0.0878 (7/7) | +0.1150 (7/7) | +0.0845 (7/7) | +0.0923 (7/7) |

- primary: **+0.1220**, positive 7/7 (needs > 0.02 and ≥ 5/7) · no-harm: **yes**
- **Verdict: ADOPT**

## Levels, predicted dead share and external firing (7-match means)

| cell | uniform spellF1 aligned | phase_a2 spellF1 aligned | cover_q50 spellF1 aligned | cover_q90 spellF1 aligned | uniform D share | phase_a2 D share | cover_q50 D share | cover_q90 D share | uniform changed | phase_a2 changed | cover_q50 changed | cover_q90 changed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 0.4095 | 0.3971 | 0.3810 | 0.3821 | 0.474 | 0.443 | 0.254 | 0.072 | 0.0% | 0.0% | 0.0% | 0.0% |
| after C | 0.4176 | 0.4114 | 0.3952 | 0.3935 | 0.469 | 0.438 | 0.249 | 0.070 | 1.5% | 1.4% | 1.9% | 2.1% |
| after K | 0.4300 | 0.4281 | 0.4621 | 0.4880 | 0.519 | 0.502 | 0.438 | 0.374 | 6.9% | 8.2% | 21.7% | 34.0% |
| after T | 0.5204 | 0.5191 | 0.4960 | 0.4744 | 0.421 | 0.418 | 0.391 | 0.353 | 19.3% | 19.9% | 27.0% | 35.3% |

Share of in-match frames changed by a kept step on its own, smallest … largest per-match value over the four mask kinds (the parity reference of the pinned hold):

- C alone (coalesce on · K 0 · T 0 · tier ball · delay 0.64 s · slope 0): 1.3% … 2.6%
- K alone (coalesce off · K 0.3 · T 0 · tier kin · delay 0.64 s · slope 0): 4.3% … 38.4%
- T alone (coalesce off · K 0 · T 0.03 · tier kin · delay 0.64 s · slope 0): 10.6% … 33.1%
- bundle: 17.4% … 41.6%

## Team-switch latency (aligned labels, predicted − true, frames; C / K / T as kept)

| delay · slope | uniform | phase_a2 | cover_q50 | cover_q90 |
|---|---|---|---|---|
| 0.64 s · 0 | +2.6 | +1.9 | +0.1 | -3.1 |
| 0.8 s · 0 | +3.7 | +3.1 | +2.0 | -1.1 |
| 0.96 s · 0 | +4.9 | +5.0 | +3.3 | +0.8 |
| 1.12 s · 0 | +5.5 | +6.4 | +4.7 | +2.5 |
| 0.64 s · 0.03 | +2.1 | +1.6 | +0.1 | -3.3 |
| 0.8 s · 0.03 | +3.7 | +3.7 | +2.0 | -1.1 |
| 0.96 s · 0.03 | +4.9 | +5.4 | +3.8 | +1.2 |
| 1.12 s · 0.03 | +5.7 | +6.6 | +5.0 | +2.9 |
| 0.64 s · 0.06 | +2.6 | +2.5 | +0.8 | -2.7 |
| 0.8 s · 0.06 | +3.9 | +4.1 | +2.6 | -0.6 |
| 0.96 s · 0.06 | +4.6 | +5.1 | +3.5 | +1.0 |
| 1.12 s · 0.06 | +6.2 | +6.3 | +5.4 | +3.3 |

## Team-swap equivariance (share of in-match frames not mirrored; 0 = symmetric)

| cell | uniform | phase_a2 | cover_q50 | cover_q90 |
|---|---|---|---|---|
| reference | 3.56% | 4.37% | 7.41% | 11.39% |
| T 0.3 · tier ball | 0.65% | 1.11% | 1.56% | 2.27% |
| T 0.3 · tier kin | 0.30% | 0.31% | 0.40% | 0.51% |

## Record-only cell (never adopted in P9)

The IDSSE ball is frozen while dead, so on ball-visible frames the ball-at-rest feature already reads the dead-ball label; this row cannot judge a kinematic term there.

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| R_kin_all | -0.0026 (2/7) | -0.0022 (1/7) | -0.0021 (2/7) | -0.0023 (1/7) | +0.0045 (6/7) | +0.0067 (6/7) | +0.0073 (5/7) | +0.0118 (6/7) |

## IDSSE side of the target parity checks (per-match range)

- Q1 median per-frame median player speed: 1.28 … 1.46 m/s
- Q4 team-logit agreement with the ball-based decoder on ball-visible frames (no mask, leave-one-out model, clean tracks): 0.734 … 0.837
- Q5 mean visible slots: 21.1 … 22.0 (degraded for the kinematic features: 19.7 … 20.5)
- Q6 median ball speed on alive frames: 5.72 … 6.29 m/s

## Reproduce

```bash
.venv/bin/python -m tuning.stage9          # full pass (~25 min, 7 workers)
.venv/bin/python -m tuning.gate9           # this read + report
.venv/bin/python -m tuning.kin_fit         # shipped model -> src/kin_model.json
.venv/bin/python -m tuning.kin_parity      # target parity checks, PASS / FAIL only
```
