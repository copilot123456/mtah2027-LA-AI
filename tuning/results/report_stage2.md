# P5 stage-2 extension report (IDSSE, CC BY 4.0 — external only)

- grid: 5 cells x ['none'] x 7 matches
- metric: spellF1 (vendored official metric code) on algorithmically derived labels
- selection scenario: none; rule: within 0.01 of best, maximize worst neighbour
- integrity: 1701 overlapping cells reproduce stage 1 (max |Δ| 0.00e+00)

**chosen**: switch_pen=2/dead_pen=8/sigma_m=2/min_dwell_s=2/switch_support_s=1
- mean spellF1 (none): 0.2736  | best cell: 0.2812  | gap to best: 0.0077
- match-level bootstrap 95% CI of chosen mean (n=7 matches): [0.2555, 0.2924]
- LOMO: mean(fold pick - global pick) on held-out = +0.0001
- scenario at chosen: none = 0.2736

## LOMO detail
| held_out   | fold_pick                                                            |   fold_score |   global_score |
|:-----------|:---------------------------------------------------------------------|-------------:|---------------:|
| J03WMX     | switch_pen=2/dead_pen=8/sigma_m=2/min_dwell_s=2/switch_support_s=1   |     0.264535 |       0.264535 |
| J03WN1     | switch_pen=4/dead_pen=8/sigma_m=2/min_dwell_s=2/switch_support_s=1.5 |     0.241096 |       0.235008 |
| J03WOH     | switch_pen=2/dead_pen=5/sigma_m=2/min_dwell_s=2/switch_support_s=1.5 |     0.263946 |       0.260068 |
| J03WOY     | switch_pen=4/dead_pen=8/sigma_m=2/min_dwell_s=2/switch_support_s=1.5 |     0.255573 |       0.260188 |
| J03WPY     | switch_pen=2/dead_pen=8/sigma_m=2/min_dwell_s=2/switch_support_s=1   |     0.290789 |       0.290789 |
| J03WQQ     | switch_pen=2/dead_pen=5/sigma_m=2/min_dwell_s=2/switch_support_s=1.5 |     0.321036 |       0.326023 |
| J03WR9     | switch_pen=2/dead_pen=8/sigma_m=2/min_dwell_s=2/switch_support_s=1   |     0.278327 |       0.278327 |

## Top 10 cells by mean (none)
|   switch_pen |   dead_pen |   sigma_m |   min_dwell_s |   switch_support_s |     mean |       std |   worst_neighbour | selected   |
|-------------:|-----------:|----------:|--------------:|-------------------:|---------:|----------:|------------------:|:-----------|
|            2 |          8 |         2 |             1 |                1   | 0.281246 | 0.0218079 |          0.186765 | False      |
|            2 |          8 |         2 |             1 |                1.5 | 0.280903 | 0.0200092 |          0.187952 | False      |
|            4 |          8 |         2 |             1 |                1.5 | 0.280066 | 0.0222923 |          0.191083 | False      |
|            2 |          5 |         2 |             1 |                1   | 0.279425 | 0.0209268 |          0.182844 | False      |
|            4 |          8 |         2 |             1 |                1   | 0.279379 | 0.0241795 |          0.190658 | False      |
|            2 |          5 |         2 |             1 |                1.5 | 0.279323 | 0.0195044 |          0.183968 | False      |
|            2 |          8 |         2 |             1 |                0.6 | 0.278854 | 0.0236029 |          0.184974 | False      |
|            4 |          5 |         2 |             1 |                1.5 | 0.276613 | 0.021079  |          0.186134 | False      |
|            2 |          5 |         2 |             1 |                0.6 | 0.27628  | 0.0228425 |          0.181565 | False      |
|            4 |          5 |         2 |             1 |                1   | 0.275749 | 0.0224854 |          0.185077 | False      |

## Documented deviations
- cache format pickle instead of parquet (no pyarrow; same-env cache only)
- bootstrap resamples MATCHES only (spellF1 not block-decomposable — plan R-U3)
- third fill scenario (simulated detector) dropped with the detector tier (stretch)
- mask placement uniform (real missingness correlates with phases we may not quantify)
