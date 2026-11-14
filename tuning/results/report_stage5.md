# Stage 5 — P7 audit follow-up pass (U1, D1, B1)

Pre-declared: `dossier.md` §7 P7 (2026-11-14). Corrected units (m/s), frozen decoder
cell, four mask kinds × two label sets, 7 IDSSE matches. Δ = 7-match mean of paired
spellF1 differences; (n/7) = matches where the difference is positive.

## Mask statistics (7-match means)

| kind | masked | P(missing given dead) | P(missing given alive) | P(dead given missing) |
|---|---|---|---|---|
| uniform | 0.433 | 0.428 | 0.436 | 0.444 |
| phase_a2 | 0.432 | 0.391 | 0.465 | 0.407 |
| cover_q50 | 0.432 | 0.638 | 0.264 | 0.662 |
| cover_q90 | 0.436 | 0.898 | 0.058 | 0.924 |

## U1 — low-activity weight (OFF − as shipped)

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| OFF − ON | +0.0442 (7/7) | +0.0607 (7/7) | +0.0403 (7/7) | +0.0492 (7/7) | +0.0287 (7/7) | +0.0396 (7/7) | +0.0249 (7/7) | +0.0368 (7/7) |

- primary (phase_a2 × aligned): **+0.0492**, positive 7/7 (needs > 0.02 and ≥ 5/7)
- no-harm on the other seven tables: **yes**
- **Verdict: ADOPT — d_w_low_activity = 0**

Corrected bed with the weight off vs the committed stage-4 frozen-cell rows:
- uniform × orig: mean |Δ| 0.0002, max 0.0008 (corrected 0.2760 vs committed 0.2759)
- uniform × aligned: mean |Δ| 0.0003, max 0.0011 (corrected 0.3765 vs committed 0.3765)
- phase_a2 × orig: mean |Δ| 0.0015, max 0.0054 (corrected 0.2619 vs committed 0.2618)
- phase_a2 × aligned: mean |Δ| 0.0012, max 0.0050 (corrected 0.3661 vs committed 0.3667)

## D1 — dead-ball term (reference: long-missing 0.5 / 10 s · low-activity off)

### R1 — regime-agnostic (automatic)

- regimes: uniform, cover_q50, cover_q90 (aligned labels); best maximin gain -0.0001; 4 cell(s) within 0.005 of it
- candidate: **long-missing 0.5 / 3 s · low-activity off** — maximin gain **-0.0001**
- gain > 0.02: **no** · ≥ 5/7 in every regime: **no** · no-harm: **yes**
- **Verdict: REJECT**

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| long-missing 0.5 / 3 s · low-activity off | -0.0002 (0/7) | -0.0001 (0/7) | +0.0010 (3/7) | +0.0018 (3/7) | +0.0003 (3/7) | +0.0005 (3/7) | +0.0001 (1/7) | +0.0001 (1/7) |
| long-missing off · low-activity off | +0.0001 (1/7) | -0.0001 (0/7) | +0.0007 (3/7) | +0.0013 (3/7) | +0.0010 (4/7) | +0.0014 (4/7) | +0.0008 (3/7) | +0.0011 (3/7) |
| long-missing off · low-activity 0.5 @ 1 m/s | -0.0004 (2/7) | -0.0020 (0/7) | +0.0021 (6/7) | +0.0039 (6/7) | -0.0003 (4/7) | -0.0013 (3/7) | +0.0003 (5/7) | +0.0001 (4/7) |
| long-missing 0.5 / 10 s · low-activity 0.5 @ 1 m/s | -0.0016 (1/7) | -0.0037 (0/7) | +0.0014 (6/7) | +0.0020 (5/7) | -0.0028 (1/7) | -0.0030 (3/7) | -0.0024 (3/7) | -0.0017 (3/7) |
| long-missing 1 / 10 s · low-activity off | -0.0031 (2/7) | -0.0052 (1/7) | -0.0032 (2/7) | -0.0045 (2/7) | -0.0016 (2/7) | -0.0020 (2/7) | -0.0027 (2/7) | -0.0004 (3/7) |
| long-missing 0.5 / 3 s · low-activity 0.5 @ 1 m/s | -0.0045 (2/7) | -0.0070 (1/7) | +0.0008 (4/7) | +0.0013 (5/7) | +0.0072 (6/7) | +0.0072 (7/7) | +0.0101 (7/7) | +0.0104 (7/7) |
| long-missing 2 / 10 s · low-activity off | -0.0063 (1/7) | -0.0111 (1/7) | -0.0051 (2/7) | -0.0047 (2/7) | -0.0066 (1/7) | -0.0077 (1/7) | -0.0082 (3/7) | -0.0062 (3/7) |
| long-missing off · low-activity 1 @ 1 m/s | -0.0085 (1/7) | -0.0117 (1/7) | -0.0036 (2/7) | -0.0036 (2/7) | +0.0094 (7/7) | +0.0120 (7/7) | +0.0140 (6/7) | +0.0173 (7/7) |

### R2 — coupled-regime reading (operator option)

- regimes: cover_q50, cover_q90 (aligned labels); best maximin gain +0.0518; 4 cell(s) within 0.005 of it
- candidate: **long-missing 2 / 3 s · low-activity off** — maximin gain **+0.0513**
- gain > 0.02: **yes** · ≥ 5/7 in every regime: **yes** · no-harm: **yes**
- **Verdict: PASS**

| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | cover_q90 × aligned |
|---|---|---|---|---|---|---|---|---|
| long-missing 2 / 3 s · low-activity 0.5 @ 1 m/s | -0.0082 (1/7) | -0.0271 (0/7) | -0.0054 (2/7) | -0.0297 (1/7) | +0.0487 (7/7) | +0.0518 (7/7) | +0.0771 (7/7) | +0.0862 (7/7) |
| long-missing 2 / 3 s · low-activity off | -0.0074 (2/7) | -0.0249 (0/7) | -0.0032 (2/7) | -0.0274 (1/7) | +0.0483 (7/7) | +0.0513 (7/7) | +0.0758 (7/7) | +0.0855 (7/7) |
| long-missing 2 / 3 s · low-activity 1 @ 1 m/s | -0.0113 (1/7) | -0.0309 (0/7) | -0.0099 (2/7) | -0.0353 (1/7) | +0.0475 (7/7) | +0.0503 (7/7) | +0.0753 (7/7) | +0.0850 (7/7) |
| long-missing 2 / 3 s · low-activity 2 @ 1 m/s | -0.0115 (1/7) | -0.0313 (0/7) | -0.0100 (2/7) | -0.0348 (1/7) | +0.0462 (7/7) | +0.0486 (7/7) | +0.0741 (7/7) | +0.0838 (7/7) |
| long-missing 2 / 3 s · low-activity 0.5 @ 1.5 m/s | -0.0264 (1/7) | -0.0517 (0/7) | -0.0277 (0/7) | -0.0558 (0/7) | +0.0397 (7/7) | +0.0412 (7/7) | +0.0672 (7/7) | +0.0774 (7/7) |
| long-missing off · low-activity 1 @ 1 m/s | -0.0085 (1/7) | -0.0117 (1/7) | -0.0036 (2/7) | -0.0036 (2/7) | +0.0094 (7/7) | +0.0120 (7/7) | +0.0140 (6/7) | +0.0173 (7/7) |
| long-missing 0.5 / 3 s · low-activity 1 @ 1 m/s | -0.0090 (0/7) | -0.0130 (0/7) | -0.0045 (2/7) | -0.0047 (1/7) | +0.0092 (6/7) | +0.0117 (7/7) | +0.0150 (6/7) | +0.0174 (7/7) |
| long-missing 0.5 / 10 s · low-activity 2 @ 1 m/s | -0.0101 (0/7) | -0.0154 (0/7) | -0.0042 (1/7) | -0.0051 (3/7) | +0.0100 (7/7) | +0.0116 (7/7) | +0.0156 (7/7) | +0.0179 (7/7) |

## Predicted dead share (7-match mean; true IDSSE dead share ≈ 0.45)

| cell | uniform | phase_a2 | cover_q50 | cover_q90 |
|---|---|---|---|---|
| as shipped | 0.525 | 0.500 | 0.344 | 0.191 |
| reference | 0.491 | 0.465 | 0.262 | 0.073 |
| R1 candidate | 0.491 | 0.465 | 0.262 | 0.073 |
| R2 candidate | 0.422 | 0.351 | 0.320 | 0.236 |

## B1 — organisers' baselines on the same bed (spellF1 / macroF1, aligned labels)

| system | uniform | phase_a2 | cover_q50 | cover_q90 | published full match |
|---|---|---|---|---|---|
| baseline_raw | 0.037 / 0.426 | 0.031 / 0.402 | 0.047 / 0.581 | 0.060 / 0.773 | 0.049 / 0.653 |
| baseline_smooth25 | 0.154 / 0.446 | 0.149 / 0.426 | 0.221 / 0.598 | 0.299 / 0.780 | 0.200 / 0.675 |
| baseline_smooth50 | 0.193 / 0.453 | 0.189 / 0.435 | 0.270 / 0.601 | 0.355 / 0.769 | 0.243 / 0.672 |
| baseline_smooth125 | 0.224 / 0.439 | 0.228 / 0.427 | 0.270 / 0.573 | 0.304 / 0.705 | 0.224 / 0.636 |
| ours, as shipped | 0.316 / 0.673 | 0.317 / 0.674 | 0.309 / 0.593 | 0.315 / 0.489 | — |
| ours, reference | 0.376 / 0.722 | 0.366 / 0.708 | 0.349 / 0.612 | 0.352 / 0.462 | — |

The published column is the organisers' own figure for the provided match (EVALUATION.md); it is shown for orientation and selects nothing (dossier §7, Q5).

## Reproduce

```bash
.venv/bin/python -m tuning.stage5          # full pass (~15 min, 6 workers)
.venv/bin/python -m tuning.gate5           # this read + report
```
