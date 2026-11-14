# Stage 8 — P8 pass (G switch gate, S team-switch delay, M dead-ball minimum)

Pre-declared: `dossier.md` §7 P8 (2026-11-14). Stage-5 reference, four mask kinds ×
two label sets, 7 IDSSE matches. Δ = 7-match mean of paired spellF1 differences;
(n/7) = matches where the difference is positive.

Harness pin: the reference cell reproduces the stage-5 reference rows, max |Δ| = 0.0e+00.

## G — switch gate (against: gate revert · gap placement 0 · delay 0 s · D minimum 0 s)

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| gate move | +0.0146 (7/7) | +0.0192 (7/7) | +0.0143 (7/7) | +0.0190 (7/7) | +0.0098 (7/7) | +0.0117 (7/7) | +0.0040 (7/7) | +0.0041 (7/7) |
| gate move · gap placement 0.5 | +0.0293 (7/7) | +0.0316 (7/7) | +0.0341 (7/7) | +0.0328 (7/7) | +0.0255 (7/7) | +0.0259 (7/7) | +0.0046 (5/7) | +0.0023 (4/7) |

- candidate (maximin over the four mask kinds, aligned labels): **gate move** — g = +0.0041 (best +0.0041)
- primary (phase_a2 × aligned): **+0.0190**, positive 7/7 (needs > 0.005 and ≥ 5/7) · no-harm on the other seven tables: **yes**
- **Step verdict: KEEP**

## S — team-switch delay (against: gate move · gap placement 0 · delay 0 s · D minimum 0 s)

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| delay 0.16 s | +0.0048 (7/7) | +0.0043 (7/7) | +0.0039 (7/7) | +0.0047 (7/7) | +0.0076 (7/7) | +0.0070 (7/7) | +0.0093 (7/7) | +0.0081 (7/7) |
| delay 0.32 s | +0.0085 (7/7) | +0.0087 (7/7) | +0.0064 (6/7) | +0.0070 (7/7) | +0.0121 (7/7) | +0.0118 (7/7) | +0.0159 (7/7) | +0.0144 (7/7) |
| delay 0.48 s | +0.0111 (7/7) | +0.0119 (7/7) | +0.0092 (6/7) | +0.0100 (7/7) | +0.0166 (7/7) | +0.0171 (7/7) | +0.0221 (7/7) | +0.0202 (7/7) |
| delay 0.64 s | +0.0138 (7/7) | +0.0138 (7/7) | +0.0114 (6/7) | +0.0119 (7/7) | +0.0203 (7/7) | +0.0207 (7/7) | +0.0266 (7/7) | +0.0262 (7/7) |
| delay 0.8 s | +0.0169 (7/7) | +0.0172 (7/7) | +0.0133 (6/7) | +0.0153 (7/7) | +0.0236 (7/7) | +0.0254 (7/7) | +0.0313 (7/7) | +0.0304 (7/7) |

- candidate (maximin over the four mask kinds, aligned labels): **delay 0.64 s** — g = +0.0119 (best +0.0153)
- primary (phase_a2 × aligned): **+0.0119**, positive 7/7 (needs > 0.005 and ≥ 5/7) · no-harm on the other seven tables: **yes**
- **Step verdict: KEEP**

## M — dead-ball minimum (against: gate move · gap placement 0 · delay 0.64 s · D minimum 0 s)

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| D minimum 3 s | -0.0006 (2/7) | +0.0027 (4/7) | -0.0004 (4/7) | +0.0001 (5/7) | +0.0005 (4/7) | +0.0034 (5/7) | +0.0006 (4/7) | +0.0046 (6/7) |
| D minimum 4 s | -0.0022 (2/7) | +0.0009 (3/7) | -0.0012 (3/7) | +0.0008 (4/7) | +0.0015 (5/7) | +0.0056 (6/7) | +0.0026 (4/7) | +0.0085 (7/7) |
| D minimum 5 s | -0.0036 (2/7) | -0.0003 (3/7) | -0.0023 (3/7) | -0.0008 (3/7) | -0.0002 (4/7) | +0.0051 (5/7) | +0.0032 (5/7) | +0.0104 (7/7) |

- candidate (maximin over the four mask kinds, aligned labels): **D minimum 3 s** — g = +0.0001 (best +0.0008)
- primary (phase_a2 × aligned): **+0.0001**, positive 5/7 (needs > 0.005 and ≥ 5/7) · no-harm on the other seven tables: **yes**
- **Step verdict: DROP — the step keeps its current value**

## Bundle (the cell that would ship) against the reference

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| gate move · gap placement 0 · delay 0.64 s · D minimum 0 s | +0.0284 (7/7) | +0.0330 (7/7) | +0.0256 (7/7) | +0.0309 (7/7) | +0.0301 (7/7) | +0.0324 (7/7) | +0.0305 (7/7) | +0.0303 (7/7) |

- primary: **+0.0309**, positive 7/7 (needs > 0.02 and ≥ 5/7) · no-harm: **yes**
- **Verdict: ADOPT**

## Levels and external firing (7-match means)

| cell | uniform spellF1 aligned | phase_a2 spellF1 aligned | cover_q50 spellF1 aligned | cover_q90 spellF1 aligned | uniform changed | phase_a2 changed | cover_q50 changed | cover_q90 changed |
|---|---|---|---|---|---|---|---|---|
| reference | 0.3765 | 0.3661 | 0.3486 | 0.3517 | 0.0% | 0.0% | 0.0% | 0.0% |
| after G | 0.3957 | 0.3852 | 0.3603 | 0.3559 | 2.4% | 2.5% | 1.8% | 0.4% |
| after S | 0.4095 | 0.3971 | 0.3810 | 0.3821 | 3.4% | 3.5% | 3.4% | 2.6% |

Largest per-match share of in-match frames changed by a kept step on its own (the parity reference of the pinned hold):

- G alone (gate move · gap placement 0 · delay 0 s · D minimum 0 s): 0.1% … 5.0%
- S alone (gate revert · gap placement 0 · delay 0.64 s · D minimum 0 s): 0.8% … 2.7%

## Team-switch latency (aligned labels, predicted − true, frames; G as kept, D minimum off)

| delay | uniform | phase_a2 | cover_q50 | cover_q90 |
|---|---|---|---|---|
| 0 s | -11.4 | -12.4 | -13.0 | -14.5 |
| 0.16 s | -9.4 | -11.5 | -11.3 | -12.8 |
| 0.32 s | -7.4 | -9.9 | -9.5 | -10.6 |
| 0.48 s | -4.8 | -7.5 | -7.2 | -8.2 |
| 0.64 s | -2.6 | -5.9 | -5.4 | -6.4 |
| 0.8 s | -1.0 | -4.3 | -2.9 | -3.8 |

## Team-swap equivariance (share of in-match frames not mirrored; 0 = symmetric)

| gate cell | uniform | phase_a2 | cover_q50 | cover_q90 |
|---|---|---|---|---|
| revert, gap placement 0 | 4.41% | 4.98% | 8.74% | 11.93% |
| move, gap placement 0 | 3.58% | 4.40% | 7.48% | 11.48% |
| move, gap placement 0.5 | 0.77% | 0.66% | 1.26% | 1.96% |

I4 — a missed stoppage inherits the team that had the ball: share of missed dead frames given to home vs home's share of live frames (reference, 7-match means): uniform 0.459 vs 0.532 · phase_a2 0.491 vs 0.532 · cover_q50 0.580 vs 0.532 · cover_q90 0.610 vs 0.532

## Record-only cells (operator idea I2 — never adopted in P8)

The IDSSE ball is frozen while dead, so `stationary` reads the dead-ball label there; these rows show the size of that dependence, not the value of the feature on real tracking.

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| R_stat0 | -0.0665 (0/7) | -0.0940 (0/7) | -0.0651 (0/7) | -0.0943 (0/7) | -0.0280 (0/7) | -0.0366 (0/7) | +0.0052 (6/7) | +0.0114 (5/7) |
| R_noD | -0.0900 (0/7) | -0.1224 (0/7) | -0.0867 (0/7) | -0.1241 (0/7) | -0.0412 (0/7) | -0.0502 (0/7) | -0.0004 (3/7) | +0.0053 (3/7) |

| cell | uniform predicted D share | phase_a2 predicted D share | cover_q50 predicted D share | cover_q90 predicted D share |
|---|---|---|---|---|
| reference | 0.491 | 0.465 | 0.262 | 0.073 |
| R_stat0 | 0.135 | 0.135 | 0.064 | 0.024 |
| R_noD | 0.000 | 0.000 | 0.000 | 0.000 |

## Reproduce

```bash
.venv/bin/python -m tuning.stage8          # full pass (~3 min, 6 workers)
.venv/bin/python -m tuning.gate8           # this read + report
```
