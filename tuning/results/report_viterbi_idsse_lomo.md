# IDSSE LOMO selection (spellF1 on IDSSE possession labels, challenge GT untouched)

- grid: 32 cells x 7 matches; metric: spellF1 mean over IoU 0.3..0.7
- current notebook config held-out mean: 0.4300 spellF1
- global pick held-out mean: 0.4488 spellF1
- fold-pick held-out mean: 0.4460 spellF1 (LOMO generalization)

**global pick**: beta=0.7  p_switch_play=0.005  min_possession_frames=25  w_control=2.6  w_duel_pulse=1

## LOMO detail

| held_out | fold_pick | fold_score | global_score | incumbent_score |
| --- | --- | --- | --- | --- |
| J03WMX | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4310 | 0.4310 | 0.4170 |
| J03WN1 | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4194 | 0.4194 | 0.3858 |
| J03WOH | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4773 | 0.4773 | 0.4554 |
| J03WOY | beta=1/p_switch_play=0.008/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4461 | 0.4542 | 0.4202 |
| J03WPY | beta=0.7/p_switch_play=0.005/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4484 | 0.4484 | 0.4495 |
| J03WQQ | beta=1/p_switch_play=0.008/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4877 | 0.4925 | 0.4796 |
| J03WR9 | beta=1/p_switch_play=0.008/min_possession_frames=25/w_control=2.6/w_duel_pulse=1 | 0.4120 | 0.4189 | 0.4025 |

## Top 12 cells by mean spellF1

| beta | p_switch_play | min_possession_frames | w_control | w_duel_pulse | spell_f1 | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.7000 | 0.0050 | 25.0000 | 2.6000 | 1.0000 | 0.4488 | 0.8579 |
| 1.0000 | 0.0080 | 25.0000 | 2.6000 | 1.0000 | 0.4482 | 0.8559 |
| 0.7000 | 0.0080 | 25.0000 | 2.6000 | 1.0000 | 0.4480 | 0.8574 |
| 1.0000 | 0.0050 | 25.0000 | 2.6000 | 1.0000 | 0.4480 | 0.8564 |
| 1.0000 | 0.0080 | 25.0000 | 1.8000 | 1.0000 | 0.4421 | 0.8527 |
| 1.0000 | 0.0050 | 25.0000 | 1.8000 | 1.0000 | 0.4412 | 0.8530 |
| 0.7000 | 0.0050 | 40.0000 | 2.6000 | 1.0000 | 0.4408 | 0.8574 |
| 0.7000 | 0.0080 | 40.0000 | 2.6000 | 1.0000 | 0.4397 | 0.8568 |
| 1.0000 | 0.0080 | 40.0000 | 2.6000 | 1.0000 | 0.4393 | 0.8554 |
| 1.0000 | 0.0050 | 40.0000 | 2.6000 | 1.0000 | 0.4388 | 0.8558 |
| 0.7000 | 0.0080 | 25.0000 | 1.8000 | 1.0000 | 0.4388 | 0.8537 |
| 0.7000 | 0.0050 | 25.0000 | 1.8000 | 1.0000 | 0.4383 | 0.8542 |

Closing step: gt_free_diagnostics.py vs IDSSE. Public-GT scoreboard = final report only.


## Parameter provenance (final, frozen config)

| parameter | value | source |
|---|---|---|
| beta | 0.7 | LOMO IDSSE (spellF1, 7 folds); ablation: flat 0.7->1.0, 0.35 hurts |
| p_switch_play | 0.005 | LOMO IDSSE (flat vs 0.008); prior aligned to raw IDSSE change rate |
| min_possession_frames | 25 | LOMO IDSSE (all 7 folds); D1: 40 ate A-claims 5.1x more than L |
| w_control | 2.6 | LOMO IDSSE (all 7 folds) |
| w_duel_pulse | 1.0 | LOMO IDSSE (all 7 folds); ablation -0.016 without it |
| duel_gap_m | 0.7 | design spec (duel = no evidence); ablation +-0.008 = noise, kept |
| min_dead_frames | 50 | IDSSE D-run durations (0% < 2 s); ablation -0.020 at 25 = largest effect |
| p_stay_dead | 0.9975 | IDSSE D-run dwell prior (p50 19.7 s) |
| ball_slow / fast / flight / stat_thr | 4.765 / 5.558 / 8.512 / 0.273 | quantile match: IDSSE operating point -> ATD scale (part E; ball speed 2.6x, frac_stationary 2.7x shift) |
| dead_players_floor | 0.4 | design hierarchy ball >> players; IDSSE physics-dead anchor 41.1% |
| w_dead_ball_slow | 7.0 | GT-free rate sweep (D-share anchor) |
| rolling_median / slow_gate / fast_gate | 7500 / 50 / 12 | design spec (5 min baseline; 2 s entry; 0.5 s exit), centered (offline) |
| p_stay_play / p_play_to_dead / p_dead_to_play / ema_alpha / contest_soft / control_radius / w_dead_oob / w_dead_missing / w_dead_fast | pre-existing | original notebook design |

## Final public-GT report (single evaluation, config frozen, run after items 1-4)

- macro-F1 0.6754 (bootstrap 0.568..0.714) | spell-F1 0.3250 (0.208..0.425) | accuracy 0.7136
- per class: H F1 0.8099 | A F1 0.6672 | D F1 0.5490 | alive/dead 0.8900
- possessions: GT 17, pred 31 (+14), mean 7.8 s vs 14.2 s | abstentions 0
- board: #1 macro-F1 (0.6754, +0.055 over best baseline), #3 spell-F1 (0.3250, tied with
  baseline_smooth50 0.3256 inside noise)
- vs previous cycle (0.7605 / 0.4118): LOWER. Reported without retouching, per protocol.
  Both configs were selected without the challenge GT; the delta is this cycle's LOMO
  re-selection + latency fixes + quantile rescale. Bootstrap CIs overlap (old 0.578..0.796),
  but the point drop is beyond the ~0.01 macro noise floor -- consistent with the domain-gap
  risk flagged in item 3 (ball availability 100%->79%, confidence channel has no IDSSE
  counterpart, quantile matching preserves rates but not which frames). No parameter was
  changed after seeing these numbers.

## Round 2026-11-16b: C1/C2/C3 outcomes and final report (report only, no change after)

Pre-declared plan: tuning/results/plan_round_2026-11-16b.md (written before execution).

- C1 (equalize processing): smoothing did NOT move the transfer ratios (ball 2.62 -> 2.67
  with med5+0.24s window; frac_stationary 2.91 -> 2.91). Distributional shift, not frame
  jitter -> quantile-rescaled thresholds KEPT, no smoothing added (measured outcome in the
  plan file).
- C2 (degrade IDSSE to ATD conditions): 1,268 injected gaps, 21.4% missing, seed 20261117.
  Global pick and ALL 7 fold picks IDENTICAL to clean IDSSE (0.7/0.005/25/2.6/1.0);
  absolute spellF1 0.4488 -> 0.4039 (gaps cost ~0.045), still above incumbent 0.3905.
  Selection is robust to the availability domain gap.
- C3 (D-L-D absorption): IDSSE has 5 such runs in 7 matches (41 frames); we had 38/1006.
  Rule added (min_d_poss_frames=50, external run-duration justification). Post-C3: D-L-D
  0 runs; D runs mean 10.8 -> 13.2 s; D share 42.01% (anchor 38.2-45.7); changes/min 3.08
  (inside the design-compatible 2.07-3.12); fast-D 4.46% (guard <=5%).

Final public-GT report (single run, frozen config, no decision taken from it):

| metric | previous frozen | this round |
|---|---:|---:|
| macro-F1 | 0.6754 | 0.7069 (CI 0.579..0.753) |
| spell-F1 | 0.3250 | 0.3478 (CI 0.226..0.452) |
| D F1 | 0.5490 | 0.6230 |
| possessions (GT 17) | 31 | 29 |
| board | #1 macro, #3 spell | #1 macro (0.7069, +0.086 over best baseline), #2 spell (0.3478, above baseline_smooth50 0.3256) |

Still below the pre-clarification 0.7605/0.4118 -- those were expectation-inflated by GT
peeks (review section 1); ~0.35 spellF1 with CI [0.23, 0.45] is the clean estimate. No
parameter changed after seeing this report.
