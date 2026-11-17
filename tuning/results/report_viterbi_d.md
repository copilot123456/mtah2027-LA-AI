# Viterbi D-channel tuning report (public GT window, in-sample)

> **COMPLIANCE NOTE (2026-11-11) -- SELECTED WITH THE CHALLENGE GT; NOT A VALID TUNING RUN.**
> The organizer clarification states that training, tuning or calibrating against the possession
> ground truth is not permitted. This sweep selected `stat_threshold_dead`, `speed_threshold_dead`,
> `w_dead_low_speed`, `w_dead_active_penalty` and `min_dead_frames` **on the public GT window**,
> and those values were written into `notebooks/viterbi.ipynb` and shipped in
> `evaluation/results/viterbi.csv`. The macro-F1 0.7315 below is therefore in-sample and
> optimistically biased; it must not be reported as a generalization estimate or used to argue
> for these parameter values. Remediation: the D channel is being recalibrated on IDSSE
> (external, algorithmic labels) and on tracking-only statistics, per the clarification; this
> file is kept as the historical record of what happened, not as a tuning run to repeat.
> GT may still be used for *evaluation* (reporting), never for selection.

- grid: 720 cells x [stat_threshold_dead, speed_threshold_dead, w_dead_low_speed, w_dead_active_penalty, min_dead_frames]; selection: macro_f1 plateau (tol 0.01) constrained by the spell guard (band 0.04)
- window: frames 11185..18684 (7500 graded frames, 17 GT possessions)
- notebook defaults (baseline cell): stat_threshold_dead=0.44  speed_threshold_dead=1.65  w_dead_low_speed=2.9  w_dead_active_penalty=2.8  min_dead_frames=25
- spell guard: PASS

**chosen**: stat_threshold_dead=0.2  speed_threshold_dead=1.95  w_dead_low_speed=6.0  w_dead_active_penalty=2.8  min_dead_frames=40

| metric | baseline | chosen |
|---|---:|---:|
| macro_f1 | 0.6115 | 0.7315 |
| spell_f1_mean | 0.2836 | 0.2642 |
| D recall | 0.2308 | 0.6521 |
| D precision | 0.8188 | 0.7386 |
| D F1 | 0.3601 | 0.6927 |
| alive/dead F1 | 0.8862 | 0.9126 |
| macro_f1 1st half | 0.4552 | 0.6167 |
| macro_f1 2nd half | 0.5300 | 0.6432 |

## Top 10 cells by macro_f1

| stat_threshold_dead | speed_threshold_dead | w_dead_low_speed | w_dead_active_penalty | min_dead_frames | macro_f1 | spell_f1 | D_recall | D_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3300 | 1.6500 | 8.0000 | 2.8000 | 40.0000 | 0.7414 | 0.2667 | 0.6474 | 0.7860 |
| 0.3300 | 1.6500 | 8.0000 | 1.0000 | 40.0000 | 0.7414 | 0.2667 | 0.6474 | 0.7860 |
| 0.3300 | 1.6500 | 8.0000 | 1.9000 | 40.0000 | 0.7414 | 0.2667 | 0.6474 | 0.7860 |
| 0.2600 | 1.6500 | 8.0000 | 2.8000 | 40.0000 | 0.7408 | 0.2667 | 0.6481 | 0.7829 |
| 0.2600 | 1.6500 | 8.0000 | 1.0000 | 40.0000 | 0.7408 | 0.2667 | 0.6481 | 0.7829 |
| 0.2600 | 1.6500 | 8.0000 | 1.9000 | 40.0000 | 0.7408 | 0.2667 | 0.6481 | 0.7829 |
| 0.2000 | 1.6500 | 8.0000 | 2.8000 | 40.0000 | 0.7389 | 0.2667 | 0.6603 | 0.7572 |
| 0.2000 | 1.6500 | 8.0000 | 1.0000 | 40.0000 | 0.7389 | 0.2667 | 0.6603 | 0.7572 |
| 0.2000 | 1.6500 | 8.0000 | 1.9000 | 40.0000 | 0.7389 | 0.2667 | 0.6603 | 0.7572 |
| 0.4400 | 1.6500 | 8.0000 | 1.0000 | 40.0000 | 0.7328 | 0.2566 | 0.6147 | 0.7839 |

## Bootstrap (1000 draws, GT segments)
- macro_f1 baseline [0.4421, 0.7120] vs chosen [0.5779, 0.7957]
- spell_f1 baseline [0.1400, 0.3267] vs chosen [0.1474, 0.3509]
