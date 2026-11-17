# Pre-declared changes, round 2026-11-16b (written BEFORE execution)

Protocol (review section 4): each change below states what, why, and its GT-free
justification. Nothing here was chosen from public-GT numbers; the public GT is passed
at most once at the end, as a report next to the previous number, never to pick between
versions. The evaluation that counts is the hidden full match.

## C1. Equalize processing instead of rescaling thresholds (review section 2)

- What: smooth the ATD ball trajectory (median, 5 frames) and compute ball speed over a
  0.25 s window instead of frame-to-frame gradient; smooth the player-speed inputs
  (mean_speed, frac_stationary) the same way. Re-measure the part-E transfer ratios.
- Why: ratios ball speed 2.62x and frac_stationary 2.7-2.9x with *similar tails*
  (IDSSE p90 13.06 vs ATD 14.80 m/s) look like measurement noise in the body of the
  distribution, not a game difference. The quantile rescale (ball_slow 4.765 m/s) leaves
  the "slow ball" threshold at a clearly-rolling ball -- a patch, and a candidate
  explanation for fragmented D.
- Success criterion: ratios move toward 1.0. If ball-speed ratio does, the four
  quantile-rescaled thresholds revert to their IDSSE-scale originals
  (ball_slow 3.0, fast 4.0, flight 7.0, stat_thr 0.15) -- no rescale patch.
- Risk if ratios do not move: keep the rescaled thresholds (they are rate-preserving by
  construction) and record that the shift is not explained by frame noise.

## C2. Degrade IDSSE to ATD conditions before re-selection (review section 2)

- What: inject the provided match's ball-gap pattern into IDSSE features -- gap-length
  distribution and 78.6% availability, seeded -- plus jitter matched to C1's residual,
  then re-run the LOMO selection (min_dead, pulse, beta) on degraded IDSSE.
- Why: ball availability 100% vs 78.6% is an unmeasured domain risk with no IDSSE
  counterpart; the missing-ball channel is ATD-specific. Selection on degraded IDSSE
  measures the parameters under the conditions they will run in.
- Precedent: dossier section 7 already injects seeded missingness from coarse structural
  statistics of this match (150 runs, median 268 frames) -- same pattern, same rules.

## C3. D-L-D absorption, conditional on measurement (review section 3)

- What: count short L/A runs (<2 s) sandwiched between D runs, in IDSSE and in the match.
  If they are rare in IDSSE (as D<2 s runs are), extend `_absorb_short_segments` to
  absorb them too.
- Why: the current rule absorbs a short L/A run only when both neighbours are L/A; a
  D -> short L -> D pattern survives and splits long stoppages, a candidate mechanism
  for the 2x D-entry rate (runs mean 10.8 s vs IDSSE 25.3 s).
- Guard: the rule changes only if the external run-duration statistics justify it --
  same justification style as min_dead_frames=50 (IDSSE: 0% of D runs < 2 s).

## Outcome of C1 (recorded after execution, measurement-only)

Smoothing did NOT move the transfer ratios: ball speed raw 2.62 -> med5+0.24s-window 2.67,
window-only 2.65, med5-only 2.81; frac_stationary 2.91 -> 2.91 (identical after med5).
Player mean_speed transfers fine (0.91). Conclusion per the pre-declared criterion: the
shift is distributional, not frame jitter (similar tails, displaced body), so the
quantile-rescaled thresholds are KEPT -- they preserve each system's operating point
regardless of cause -- and the "rescale patch" characterization is replaced by
"distribution shift, rate-matched". No smoothing is added to the notebook: it would add
complexity without changing the ratios.

## Not to be done this round

- Any change justified by public-GT numbers, or choosing between versions by GT result.
- Touching min_dead_frames (reviewed last round: IDSSE 0% < 2 s -- correctly set).
- Section 6 (supervised two-stage, X review): out of scope, noted as future work.
