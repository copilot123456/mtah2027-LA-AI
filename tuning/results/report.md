# P5 sweep report (IDSSE, CC BY 4.0 — external only)

- grid: 243 cells x 2 scenarios x 7 matches
- metric: spellF1 (vendored official metric code) on algorithmically derived labels
- selection scenario: kinematic; rule: within 0.01 of best, maximize worst neighbour

**chosen**: switch_pen=4/dead_pen=8/sigma_m=3/min_dwell_s=2/switch_support_s=1
- mean spellF1 (kinematic): 0.2460  | best cell: 0.2500  | gap to best: 0.0040
- match-level bootstrap 95% CI of chosen mean (n=7 matches): [0.2270, 0.2639]
- LOMO: mean(fold pick - global pick) on held-out = -0.0005 (optimism; positive means LOMO picks beat global on held-out)
- no-harm on `none` scenario: chosen 0.2572 vs best-none 0.2576

## LOMO detail
| held_out   | fold_pick                                                          |   fold_score |   global_score |
|:-----------|:-------------------------------------------------------------------|-------------:|---------------:|
| J03WMX     | switch_pen=4/dead_pen=8/sigma_m=3/min_dwell_s=2/switch_support_s=1 |     0.254971 |       0.254971 |
| J03WN1     | switch_pen=4/dead_pen=8/sigma_m=3/min_dwell_s=2/switch_support_s=1 |     0.207262 |       0.207262 |
| J03WOH     | switch_pen=4/dead_pen=5/sigma_m=3/min_dwell_s=2/switch_support_s=1 |     0.251465 |       0.255307 |
| J03WOY     | switch_pen=4/dead_pen=8/sigma_m=3/min_dwell_s=2/switch_support_s=1 |     0.231988 |       0.231988 |
| J03WPY     | switch_pen=4/dead_pen=8/sigma_m=3/min_dwell_s=2/switch_support_s=1 |     0.261914 |       0.261914 |
| J03WQQ     | switch_pen=4/dead_pen=8/sigma_m=3/min_dwell_s=2/switch_support_s=1 |     0.297199 |       0.297199 |
| J03WR9     | switch_pen=4/dead_pen=8/sigma_m=3/min_dwell_s=2/switch_support_s=1 |     0.252889 |       0.252889 |

## Top 10 cells by mean (kinematic)
|   switch_pen |   dead_pen |   sigma_m |   min_dwell_s |   switch_support_s |     mean |       std |   worst_neighbour | selected   |
|-------------:|-----------:|----------:|--------------:|-------------------:|---------:|----------:|------------------:|:-----------|
|            4 |          8 |         3 |             1 |                1   | 0.250044 | 0.0203536 |          0.181375 | False      |
|            4 |          8 |         3 |             1 |                0.6 | 0.248944 | 0.0190438 |          0.175933 | False      |
|            4 |          3 |         3 |             2 |                1   | 0.247341 | 0.029081  |          0.175693 | False      |
|            4 |          5 |         3 |             2 |                1   | 0.246634 | 0.024764  |          0.185532 | False      |
|            4 |          5 |         3 |             1 |                1   | 0.24649  | 0.0202482 |          0.177843 | False      |
|            4 |          5 |         3 |             1 |                0.6 | 0.246221 | 0.0190855 |          0.172591 | False      |
|            4 |          8 |         3 |             2 |                1   | 0.246044 | 0.0268805 |          0.187659 | True       |
|            4 |          3 |         3 |             2 |                0.6 | 0.245641 | 0.0285693 |          0.175445 | False      |
|            4 |          5 |         3 |             2 |                0.6 | 0.245593 | 0.0250987 |          0.184775 | False      |
|            4 |          8 |         3 |             2 |                0.6 | 0.244044 | 0.0266023 |          0.183296 | False      |

## Documented deviations
- cache format pickle instead of parquet (no pyarrow; same-env cache only)
- bootstrap resamples MATCHES only (no within-match blocks: spellF1 is not block-decomposable without edge artifacts — plan R-U3 caveat)
- third fill scenario (simulated detector) dropped with the detector tier (stretch)
- mask placement uniform (real missingness correlates with phases we may not quantify)
