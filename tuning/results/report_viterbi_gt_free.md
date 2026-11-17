# Viterbi GT-free calibration (IDSSE-anchored, no challenge GT)

Grid: 8 cells x [beta, p_switch_play, p_stay_dead]
Selection: feasible within IDSSE ranges (0.407, 0.517) D-share, (2.07, 3.12) changes/min, fast-D <= 5%; then min no-control-switch fraction. Relax level: 0

**chosen**: beta=1.0  p_switch_play=0.008  p_stay_dead=0.9975

| beta | p_switch_play | p_stay_dead | d_share | changes_per_min | fast_d_frac | no_control_switch_frac | mean_possession_s | d_runs_mean_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.0080 | 0.9975 | 0.4095 | 2.3040 | 0.0270 | 0.4422 | 9.7467 | 10.9319 |
| 1.0000 | 0.0080 | 0.9985 | 0.4095 | 2.3040 | 0.0270 | 0.4422 | 9.7467 | 10.9319 |
| 1.0000 | 0.0050 | 0.9975 | 0.4095 | 2.2662 | 0.0270 | 0.4437 | 9.8748 | 10.9321 |
| 1.0000 | 0.0050 | 0.9985 | 0.4095 | 2.2662 | 0.0270 | 0.4437 | 9.8748 | 10.9321 |
| 0.7000 | 0.0050 | 0.9985 | 0.4071 | 2.1529 | 0.0276 | 0.4526 | 10.4359 | 11.5488 |
| 0.7000 | 0.0050 | 0.9975 | 0.4071 | 2.1529 | 0.0276 | 0.4526 | 10.4359 | 11.5488 |
| 0.7000 | 0.0080 | 0.9975 | 0.4082 | 2.1529 | 0.0276 | 0.4599 | 10.3881 | 11.5280 |
| 0.7000 | 0.0080 | 0.9985 | 0.4082 | 2.1529 | 0.0276 | 0.4599 | 10.3881 | 11.5280 |
