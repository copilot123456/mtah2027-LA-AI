# P5 stage-3 final report (IDSSE, CC BY 4.0 — external only)

- grid: 930 unique cells x ['none'] x 7 matches
- metric: spellF1 (vendored official metric code) on algorithmically derived labels
- selection scenario: none; rule: within 0.01 of best, maximize worst neighbour
- integrity: 1134 overlapping cells reproduce the prior stage (max |Δ| 0.00e+00)

**chosen**: switch_pen=2/dead_pen=5/sigma_m=1.5/min_dwell_s=2/switch_support_s=1.5
- mean spellF1 (none): 0.2759  | best cell: 0.2854  | gap to best: 0.0095
- match-level bootstrap 95% CI of chosen mean (n=7 matches): [0.2578, 0.2932]
- LOMO: mean(fold pick - global pick) on held-out = -0.0026
- scenario at chosen: none = 0.2759

## LOMO detail
| held_out   | fold_pick                                                              |   fold_score |   global_score |
|:-----------|:-----------------------------------------------------------------------|-------------:|---------------:|
| J03WMX     | switch_pen=2/dead_pen=8/sigma_m=1.5/min_dwell_s=2/switch_support_s=1.5 |     0.271698 |       0.274566 |
| J03WN1     | switch_pen=2/dead_pen=5/sigma_m=1.5/min_dwell_s=2/switch_support_s=1.5 |     0.234399 |       0.234399 |
| J03WOH     | switch_pen=2/dead_pen=8/sigma_m=1.5/min_dwell_s=2/switch_support_s=1.5 |     0.263714 |       0.264395 |
| J03WOY     | switch_pen=2/dead_pen=5/sigma_m=1.5/min_dwell_s=2/switch_support_s=1   |     0.269592 |       0.273552 |
| J03WPY     | switch_pen=2/dead_pen=5/sigma_m=1/min_dwell_s=2/switch_support_s=1.5   |     0.29404  |       0.304636 |
| J03WQQ     | switch_pen=2/dead_pen=5/sigma_m=1.5/min_dwell_s=2/switch_support_s=1.5 |     0.311111 |       0.311111 |
| J03WR9     | switch_pen=2/dead_pen=5/sigma_m=1.5/min_dwell_s=2/switch_support_s=1.5 |     0.268939 |       0.268939 |

## Top 10 cells by mean (none)
|   switch_pen |   dead_pen |   sigma_m |   min_dwell_s |   switch_support_s |     mean |       std |   worst_neighbour | selected   |
|-------------:|-----------:|----------:|--------------:|-------------------:|---------:|----------:|------------------:|:-----------|
|            2 |          8 |       1.5 |             1 |                1.5 | 0.2854   | 0.0262609 |          0.187952 | False      |
|            2 |          8 |       1.5 |             1 |                1   | 0.285388 | 0.0278066 |          0.186765 | False      |
|            4 |          8 |       1.5 |             1 |                1.5 | 0.283428 | 0.0244273 |          0.191083 | False      |
|            4 |          8 |       1.5 |             1 |                1   | 0.283408 | 0.0261999 |          0.190658 | False      |
|            2 |          8 |       1.5 |             1 |                0.6 | 0.283222 | 0.0288244 |          0.184974 | False      |
|            4 |          8 |       1   |             1 |                1.5 | 0.282258 | 0.0242617 |          0.191083 | False      |
|            4 |          8 |       1   |             1 |                1   | 0.282252 | 0.0246595 |          0.190658 | False      |
|            2 |          5 |       1.5 |             1 |                1   | 0.281767 | 0.0253854 |          0.182844 | False      |
|            2 |          8 |       1   |             1 |                1   | 0.281623 | 0.0229374 |          0.186765 | False      |
|            2 |          8 |       1   |             1 |                1.5 | 0.281484 | 0.022389  |          0.187952 | False      |

## Documented deviations
- cache format pickle instead of parquet (no pyarrow; same-env cache only)
- bootstrap resamples MATCHES only (spellF1 not block-decomposable — plan R-U3)
- third fill scenario (simulated detector) dropped with the detector tier (stretch)
- mask placement uniform (real missingness correlates with phases we may not quantify)
