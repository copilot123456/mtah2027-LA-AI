# IDSSE LOMO selection -- DEGRADED: ATD ball gaps injected (C2, seed 20261117) (spellF1 on IDSSE labels, challenge GT untouched)

- grid: 32 cells x 7 matches; metric: spellF1 mean over IoU 0.3..0.7
- current notebook config held-out mean: 0.3905 spellF1
- global pick held-out mean: 0.4039 spellF1
- fold-pick held-out mean: 0.4039 spellF1 (LOMO generalization)

**global pick**: beta=0.7  p_switch_play=0.005  min_possession_frames=25  w_control=2.6  w_duel_pulse=1

## LOMO detail

| held_out | fold_pick | fold_score | global_score | incumbent_score |
| --- | --- | --- | --- | --- |
| J03WMX | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.3919 | 0.3919 | 0.3847 |
| J03WN1 | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.3762 | 0.3762 | 0.3412 |
| J03WOH | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4457 | 0.4457 | 0.4335 |
| J03WOY | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4022 | 0.4022 | 0.3686 |
| J03WPY | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.3922 | 0.3922 | 0.3945 |
| J03WQQ | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4404 | 0.4404 | 0.4519 |
| J03WR9 | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.3790 | 0.3790 | 0.3594 |

## Top 12 cells by mean spellF1

| beta | p_switch_play | min_possession_frames | w_control | w_duel_pulse | spell_f1 | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.7000 | 0.0050 | 25.0000 | 2.6000 | 1.0000 | 0.4039 | 0.8417 |
| 0.7000 | 0.0080 | 25.0000 | 2.6000 | 1.0000 | 0.4028 | 0.8407 |
| 1.0000 | 0.0080 | 25.0000 | 2.6000 | 1.0000 | 0.4016 | 0.8361 |
| 1.0000 | 0.0050 | 25.0000 | 2.6000 | 1.0000 | 0.4004 | 0.8372 |
| 0.7000 | 0.0050 | 40.0000 | 2.6000 | 1.0000 | 0.3983 | 0.8415 |
| 0.7000 | 0.0080 | 25.0000 | 1.8000 | 1.0000 | 0.3981 | 0.8391 |
| 1.0000 | 0.0080 | 25.0000 | 1.8000 | 1.0000 | 0.3974 | 0.8350 |
| 0.7000 | 0.0050 | 25.0000 | 1.8000 | 1.0000 | 0.3973 | 0.8397 |
| 0.7000 | 0.0080 | 40.0000 | 2.6000 | 1.0000 | 0.3964 | 0.8404 |
| 1.0000 | 0.0050 | 25.0000 | 1.8000 | 1.0000 | 0.3957 | 0.8358 |
| 1.0000 | 0.0080 | 40.0000 | 2.6000 | 1.0000 | 0.3955 | 0.8357 |
| 1.0000 | 0.0050 | 40.0000 | 2.6000 | 1.0000 | 0.3944 | 0.8368 |

Closing step: gt_free_diagnostics.py vs IDSSE. Public-GT scoreboard = final report only.
