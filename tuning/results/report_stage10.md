# Stage 10 — P10 pass (kin_w_dead past the grid edge)

Pre-declared: `dossier.md` §7 P10 (2026-11-15). Shipped bundle cell as the reference (coalesce on · K 0.3 · T 0.03 · tier kin · delay 0.64 s · slope 0), four mask kinds × two label sets, 7 IDSSE matches; kinematic models fitted leave-one-match-out, kinematic features of the test match from dropout-degraded player tracks. Δ = 7-match mean of paired spellF1 differences against the 0.3 cell; (n/7) = matches where the difference is positive.

Harness pin: the 0.3 cell reproduces the stage-9 bundle row, max |Δ| = 0.0e+00.

## K weight (against: the shipped bundle, K 0.3)

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| K 0.4 | +0.0059 (6/7) | +0.0069 (6/7) | +0.0070 (6/7) | +0.0108 (7/7) | +0.0101 (7/7) | +0.0121 (7/7) | +0.0148 (7/7) | +0.0165 (7/7) |
| K 0.5 | +0.0071 (6/7) | +0.0081 (6/7) | +0.0063 (6/7) | +0.0098 (7/7) | +0.0146 (7/7) | +0.0170 (6/7) | +0.0221 (7/7) | +0.0251 (7/7) |
| K 0.6 | +0.0078 (6/7) | +0.0092 (6/7) | +0.0065 (5/7) | +0.0067 (6/7) | +0.0154 (6/7) | +0.0169 (6/7) | +0.0268 (7/7) | +0.0303 (7/7) |
| K 0.8 | +0.0106 (6/7) | +0.0116 (7/7) | +0.0066 (5/7) | +0.0044 (6/7) | +0.0199 (6/7) | +0.0197 (6/7) | +0.0340 (7/7) | +0.0366 (7/7) |
| K 1 | +0.0110 (6/7) | +0.0124 (6/7) | +0.0066 (4/7) | +0.0019 (4/7) | +0.0223 (7/7) | +0.0226 (6/7) | +0.0377 (7/7) | +0.0412 (7/7) |

### Pinned rule (smallest w: primary > +0.02, ≥ 5/7, no-harm ≥ −0.02)

- K 0.4: primary +0.0108 (7/7), worst other table +0.0059 — **REJECT**
- K 0.5: primary +0.0098 (7/7), worst other table +0.0063 — **REJECT**
- K 0.6: primary +0.0067 (6/7), worst other table +0.0065 — **REJECT**
- K 0.8: primary +0.0044 (6/7), worst other table +0.0066 — **REJECT**
- K 1: primary +0.0019 (4/7), worst other table +0.0066 — **REJECT**

- **Step verdict: REJECT — `kin_w_dead = 0.3` ships unchanged**

## Levels, predicted dead share and changed share (7-match means)

| cell | uniform aligned | phase_a2 aligned | cover_q50 aligned | cover_q90 aligned | uniform D | phase_a2 D | cover_q50 D | cover_q90 D | uniform changed | phase_a2 changed | cover_q50 changed | cover_q90 changed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K 0.3 | 0.5204 | 0.5191 | 0.4960 | 0.4744 | 0.4214 | 0.4176 | 0.3914 | 0.3531 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| K 0.4 | 0.5273 | 0.5299 | 0.5081 | 0.4909 | 0.4355 | 0.4345 | 0.4100 | 0.3759 | 0.0156 | 0.0182 | 0.0204 | 0.0242 |
| K 0.5 | 0.5285 | 0.5289 | 0.5130 | 0.4995 | 0.4449 | 0.4434 | 0.4209 | 0.3900 | 0.0252 | 0.0277 | 0.0317 | 0.0392 |
| K 0.6 | 0.5297 | 0.5258 | 0.5129 | 0.5047 | 0.4510 | 0.4508 | 0.4291 | 0.3995 | 0.0316 | 0.0353 | 0.0404 | 0.0489 |
| K 0.8 | 0.5320 | 0.5235 | 0.5157 | 0.5110 | 0.4590 | 0.4574 | 0.4379 | 0.4096 | 0.0401 | 0.0421 | 0.0494 | 0.0598 |
| K 1 | 0.5328 | 0.5210 | 0.5186 | 0.5156 | 0.4634 | 0.4620 | 0.4429 | 0.4151 | 0.0443 | 0.0475 | 0.0543 | 0.0656 |

## External firing — share of in-match frames changed vs the 0.3 cell, smallest … largest per-match value over the four mask kinds

- K 0.4: 1.3% … 2.7%
- K 0.5: 1.9% … 4.7%
- K 0.6: 2.5% … 6.0%
- K 0.8: 2.9% … 7.4%
- K 1: 3.5% … 8.0%

## Record-only rows (never adopted)

| cell | uniform aligned | phase_a2 aligned | cover_q50 aligned | cover_q90 aligned | uniform D | phase_a2 D | cover_q50 D | cover_q90 D |
|---|---|---|---|---|---|---|---|
| R_K05_all | 0.5021 | 0.4984 | 0.4849 | 0.4696 | 0.4498 | 0.4484 | 0.4338 | 0.4124 |
| R_K10_T0 | 0.4254 | 0.4234 | 0.4726 | 0.5184 | 0.5557 | 0.5400 | 0.4864 | 0.4307 |

`R_K05_all` puts the dead term on every frame (ball-visible included) and `R_K10_T0` reads the K×T interaction at K 1.0 with T 0 — both record-only; the bed cannot judge dead-ball features on ball-visible frames.
