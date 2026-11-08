# Technical report — LA+AI (draft)

Frame-level team possession labeling (`H`/`A`/`D`/`X`) for FC Barcelona vs Athletic Club
(BAR-ATH), More Than A Hack 2027 Initial Challenge. Ranking metric: `spellF1`.

## Approach — what we built and why

A four-stage probabilistic pipeline over the provided tracking, tuned exclusively on
external data (IDSSE), with every parameter carrying an explicit provenance
(`prior` | `fixed_default` | `external_fit` — enforced by a test):

1. **Ball provenance tiers** (`src/ball_fill.py`). Every frame is tagged `atd` (raw
   observation) or `none` (no ball evidence), with `age_s` (seconds to the nearest raw
   observation) and an `out_of_pitch` flag computed before any nulling. Raw ATD positions
   are never modified (bit-exact, tested).
   A kinematic cubic-Hermite fill for bracketed gaps ≤ 10 s was implemented and measured
   on external data: it **hurt** (see *Layer decisions*) and is disabled in the shipped
   configuration (`fill_long_gap_s = 0`, origin `external_fit`).

2. **Emission model** (`src/emission.py`). Per-frame log-emissions over `H/A/D`:
   - team evidence: per-team top-k kernel `Σ exp(-(d/σ)²)` over *visible* slots, distances
     in metres (normalized coordinates × pitch dimensions), σ widened for estimated
     positions;
   - confidence tempering: raw observations carry full weight, frames without a ball
     carry **zero** team weight — their H/A columns are exactly equal, so evidence-free
     stretches cannot by themselves flip a possession (loose balls stay with the last
     team that had them);
   - a dead-ball term `p_d = sigmoid(bias + Σ wᵢfᵢ)` over five features: ball outside the
     pitch margin, ball at rest (< 1 m/s), ball near a boundary, long no-ball run, low
     team activity. The bias is an a-priori logit(≈1/3) — an external re-fit to the
     IDSSE dead share was tried and **rejected** (it collapsed performance; see dossier).

3. **Segment decoder** (`src/decoder.py`). Per-half Viterbi in log space with switch and
   dead-ball penalties, followed by two guards the metric analysis motivated:
   - **switch gating**: a boundary without nearby raw-ATD support is reverted — evidence
     from estimated/absent ball alone never switches a possession;
   - **min-dwell**: segments shorter than the dwell merge into the longer neighbour.
   The decoder directly targets `spellF1`'s segment semantics: the raw nearest-player
   baseline fragments a match into hundreds of possessions; segment-coherent decoding
   yields realistic possession counts with 0 % sub-second segments.

4. **Abstention policy** (`src/x_policy.py`). `X` is written outside the graded spans
   (free) and **not** on graded frames: for a dwell- and gated-smoothed path, emitting the
   decoded label produces the same forward-filled array as abstaining (so `spellF1` cannot
   improve) while `macroF1` can only lose a true positive. The alternative policy
   (abstain in stable low-confidence interiors, per the docs' guidance for per-frame
   systems) was implemented and evaluated on external data: −0.011 spellF1 — kept off.

## Assets used — and which were tried and dropped

| Asset | Use |
|---|---|
| Tactical ATD tracking | primary input (positions, speeds, slot teams from the EPTS metadata) |
| Tactical homography | available; unused so far (positions already pitch-normalized; video not needed) |
| Smart Tagging | inspected; **not used** — phase intervals are not possessions, and no external labels exist to calibrate a weight (weak prior, default off) |
| Broadcast ATD/video | **dropped** — separate timeline with cuts/replays and no mapping to the submission index; solving alignment was not worth the risk |
| Kinematic ball fill | implemented, ablated externally, **dropped** (see above) |
| Official evaluator | format validation + one frozen final check (guarded, logged) |

## Training data

- **IDSSE** — 7 full Bundesliga 2022/23 matches (Sportec TRACAB tracking + official DFL
  events), *An integrated dataset of synchronized spatiotemporal and event data in elite
  soccer* (Bassek et al., Scientific Data 2025), **CC BY 4.0**, mirrored at
  `huggingface.co/datasets/pysport/idsse-data`.
- Labels for tuning were derived **algorithmically** from provider fields
  (`ball_state` → D, `ball_owning_team_id` → side) — no human annotation anywhere.
- Used for: parameter selection (grid search, 930 unique cells × 7 matches), layer
  ablations, and two adoption checks (X-policy, dead-bias re-fit). Never for the provided
  game. All of it external; the provided game's ground truth was never used to choose,
  tune or calibrate anything.

## Possession rules implemented

- A possession is a maximal run of one team while the ball is alive; dead ball is `D`,
  never a possession; each half is decoded independently (no state across the break).
- Loose/contested phases stay with the last team that had the ball: evidence-free frames
  have exactly equal H/A emissions, so the decoder carries the previous team.
- Change of possession requires the decoder to overcome the switch penalty with
  supporting raw tracking nearby — a single deflection or duel win (a few frames of
  flickering evidence) cannot pass both the gate and the dwell filter.
- Restarts produce dead-ball stretches via the D features (stationary/boundary/
  out-of-pitch); D blips shorter than the dwell merge into the surrounding state.

## Layer decisions (external evidence, IDSSE)

| Decision | Evidence | Outcome |
|---|---|---|
| Kinematic fill tier | paired `kinematic − none` = **−0.0112** spellF1, 95 % CI [−0.017, −0.007], 7/7 matches negative; all-cell mean −0.0089 | **dropped** |
| Dead-bias re-fit to the external D share | **−0.177** spellF1 | rejected; prior kept |
| X-policy on graded frames | −0.0113, negative on 6/7 matches | kept off |

## Limitations

- Tuning runs under a simplified missingness model: mask runs placed uniformly at random
  on IDSSE using the provided match's *coarse* run-length distribution. The real
  missingness correlates with match phases this simulation does not reproduce.
- Broadcast footage is unused (alignment cost); a video-based ball detector tier was
  scoped as stretch and is not in the shipped pipeline.
- The provider's dead-ball share on external matches (41–52 %) differs from what the
  prior encodes; the feature-driven D term compensates but was not re-tuned per league.
- Parameter selection sits on plateau edges for σ and switch penalty; further grid
  extension had diminishing returns within the time budget.

## Reproduction

```bash
pip install -r requirements-starter.txt pytest tabulate
# match data in data/tactical (organizer bundle; gitignored)
python -m pytest tests/ -q              # alignment, invariants, official format check
python -m src.submit                    # writes LA+AI.csv and runs the official validator
python evaluation/run_evaluation.py     # official scorer (optional; runs are logged)
```

External tuning evidence ships in `tuning/results/` (matrices + reports);
`tuning/sweep.py` and `tuning/post_checks.py` regenerate it given `IDSSE_DIR`.

## Final configuration (frozen 2026-11-07)

| parameter | shipped | origin |
|---|---|---|
| `switch_pen` | 2.0 | `external_fit` (IDSSE) |
| `dead_pen` | 5.0 | `prior` (external surface flat) |
| `sigma_m` | 1.5 | `external_fit` (IDSSE) |
| `min_dwell_s` | 2.0 | `external_fit` (IDSSE) |
| `switch_support_s` | 1.5 | `external_fit` (IDSSE) |
| `fill_long_gap_s` | 0 (tier off) | `external_fit` (layer ablation) |
| `x_use` | False | evaluated, rejected |
| `d_bias` | −0.70 | `prior` (re-fit tried, rejected) |

External selection: 930-cell union × 7 matches, mean spellF1 **0.2759** (95 % CI
[0.2578, 0.2932]), LOMO optimism −0.0026. The submission regenerates byte-identically
(md5-verified) and passes the official validator.
