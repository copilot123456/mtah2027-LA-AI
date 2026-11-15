# Technical report — LA+AI

> **New to the project?** Start at [README.md](README.md) → the guided path in
> [docs/](docs/01-the-challenge.md). This document is the submission-facing summary.

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
   - a dead-ball term `p_d = sigmoid(bias + Σ wᵢfᵢ)` over four active features: ball
     outside the pitch margin, ball at rest (< 1 m/s), ball near a boundary, long no-ball
     run. A fifth, low team activity, is implemented and **switched off by external
     evidence** (see *Layer decisions*).
   - **player-only kinematic terms** (`src/kinematics.py`), active only on frames with
     no ball observation: a dead-ball log-likelihood ratio from how the players move
     (median speed, share of slow and fast players, collective drift) is added to the
     dead logit, and a team log-odds from four home-minus-away features (collective
     direction of movement, territory, difference of the teams' length and width) is added
     to the H / A columns. Both are logistic regressions fitted on the external matches;
     the team model has no intercept, so swapping the teams mirrors it exactly. The bias is an a-priori logit(≈1/3) — an external
     re-fit to the IDSSE dead share was tried and **rejected** (it collapsed performance;
     see dossier).

3. **Segment decoder** (`src/decoder.py`). Per-half Viterbi in log space with switch and
   dead-ball penalties, followed by two guards the metric analysis motivated:
   - **switch gating**: a boundary without nearby raw-ATD support does not stand — the
     previous label holds until the first raw-ATD frame that supports the new one, and a
     segment with no such frame is merged away. Evidence from an estimated or absent ball
     alone never switches a possession;
     On ball-less frames a frame carrying a kinematic term counts as support too;
   - **min-dwell**: segments shorter than the dwell merge into the longer neighbour, and
     equal neighbours are joined after every merge;
   - **team-switch delay**: a change of team that the ball evidence placed is moved 0.64 s
     later. The kernel hands the ball over as soon as it is nearer to the receiving team —
     in mid-flight — while a change of possession is the moment the receiver controls it.
   The decoder directly targets `spellF1`'s segment semantics: the raw nearest-player
   baseline fragments a match into hundreds of possessions; segment-coherent decoding
   yields realistic possession counts with 0 % sub-second segments.

4. **Abstention policy** (`src/x_policy.py`). `X` is written outside the graded spans
   (free) and **not** on graded frames: for a dwell- and gated-smoothed path, emitting the
   decoded label produces the same forward-filled array as abstaining (so `spellF1` cannot
   improve) while `macroF1` can only lose a true positive. The alternative policy
   (abstain in stable low-confidence interiors, per the docs' guidance for per-frame
   systems) was implemented and evaluated on external data under the official scoring:
   spellF1 identical to the shipped path on every match, macroF1 −0.11 — kept off.

## Assets used — and which were tried and dropped

| Asset | Use |
|---|---|
| Tactical ATD tracking | primary input (positions, speeds, slot teams from the EPTS metadata) |
| Tactical homography | available; unused so far (positions already pitch-normalized; video not needed) |
| Smart Tagging | the `Kick Off (Start)` tag only (dead ball before a half's kick-off); phase and set-piece tags **not used** — phase intervals are not possessions, and no external tagging exists to calibrate a weight |
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
  ablations, and two adoption checks (X-policy, dead-bias re-fit) — run at the stage-1
  cell and re-confirmed at the shipped cell. Never for the provided game. All of it
  external; the provided game's ground truth was never used to choose, tune or calibrate
  anything.

## Possession rules implemented

- A possession is a maximal run of one team while the ball is alive; dead ball is `D`,
  never a possession; each half is decoded independently (no state across the break).
- Loose/contested phases stay with the last team that had the ball: evidence-free frames
  have exactly equal H/A emissions, so the decoder carries the previous team.
- Change of possession requires the decoder to overcome the switch penalty with
  supporting raw tracking nearby — a single deflection or duel win (a few frames of
  flickering evidence) cannot pass both the gate and the dwell filter.
- The change is timed where the receiving team controls the ball, not where the ball
  becomes nearer to it: ball-placed team switches are moved 0.64 s later.
- Frames of a graded half that precede its Smart Tagging `Kick Off (Start)` tag are `D`
  (the second half is graded before it kicks off).
- Restarts produce dead-ball stretches via the D features (stationary/boundary/
  out-of-pitch); D blips shorter than the dwell merge into the surrounding state.

## Layer decisions (external evidence, IDSSE)

| Decision | Evidence | Outcome |
|---|---|---|
| Kinematic fill tier | paired `kinematic − none` = **−0.0112** spellF1, 95 % CI [−0.017, −0.007], 7/7 matches negative; all-cell mean −0.0089 | **dropped** |
| Dead-bias re-fit to the external D share | **−0.177** spellF1 at the stage-1 cell; **−0.085** at the shipped cell, 7/7 matches negative | rejected; prior kept |
| X-policy on graded frames | spellF1 **identical** at both cells (the scorer forward-fills `X` inside a possession); macroF1 **−0.079** / **−0.110**, 7/7 matches negative | kept off |
| Low-activity dead-ball feature | weight 0 minus 0.5 = **+0.049** spellF1 on the primary table, 7/7 matches; positive on all 8 tables (+0.025 … +0.061), units-corrected bed | **switched off** (2026-11-14) |
| Dead-ball term re-sweep (49 cells) | no cell gains under every ball-loss regime (best worst-case gain −0.0001) | reference kept |
| Set-piece tag as dead-ball prior | −0.0075 on the primary table, 2/7 matches | kept off |
| Switch gate: move the boundary instead of merging the segment | the merge deleted whole possessions that had their own ball evidence (33–121 over seven matches, by ball-loss model); move − merge = **+0.019** spellF1 on the primary table, 7/7 matches; positive on all 8 tables (+0.004 … +0.019) | **on** (2026-11-14) |
| Team-switch delay, 0.64 s | external team switches are 11–15 frames early on 7/7 matches; **+0.012** on top of the gate change, 7/7; positive on all 8 tables. Both together: **+0.031**, 7/7, +0.026 … +0.033 on all 8 tables | **on** (2026-11-14) |
| Mid-gap placement of a switch the ball cannot time | +0.026 … +0.033 under three ball-loss models, +0.002 (4/7) under the strongly dead-tied one | kept off (worst-case rule) |
| Longer minimum duration for dead-ball segments (3–5 s) | +0.0001 on the primary table | kept off |
| Min-dwell: join equal neighbours after a merge | without it a short segment could be merged into the other team although its twin made it long enough; **+0.014** on the primary table, 7/7; positive on all 8 tables | **on** (2026-11-14) |
| Player-only dead-ball term on ball-less frames (weight 0.3) | leave-one-match-out model, player tracks degraded by 7 % dropout; **+0.017** on the primary table, 7/7; all 8 tables positive, up to +0.095 when ball loss is tied to dead ball | **on** (2026-11-14) |
| Player-only team term on ball-less frames (weight 0.03) | **+0.091** on the primary table, 7/7; +0.019 … +0.091 on six tables, −0.009 / −0.014 on the two strongly dead-tied tables (inside the no-harm bound). All three together: **+0.122**, 7/7, +0.083 … +0.122 on all 8 tables | **on** (2026-11-14) |
| Longer or ball-speed-scaled team-switch delay | best cell +0.004 on the primary table once the kinematic terms are in | kept off |
| Kick-off tag rule (`D` before a half's kick-off tag) | documents-only rule (FAQ + Smart Tagging `Kick Off (Start)`); no external test possible; second-half tag within 10 s of the documented kick-off; changes < 2 % of graded labels | **on** (2026-11-14, check amended by team decision — see dossier) |

(`tuning/results/post_checks.csv`, `post_checks_final.csv`, `report_stage4.md`,
`report_e7.md`, `report_stage5.md`, `report_stage8.md`, `report_stage9.md`, scored with the official forward-fill semantics;
adoption rule: paired gain > 0.02 on at least 5 of 7 matches.)

**Units correction (2026-11-14).** The external tracking reports player speed in km/h and
the provided match in m/s. Until this date the test bed applied the m/s activity threshold
to km/h values, so the low-activity feature was effectively inactive in every external
experiment while it was active on the provided match. The loader now converts to m/s
behind a self-check, and the feature was re-evaluated on the corrected bed (table above).
Earlier matrices are kept as the record of what ran; they describe the weight-off
configuration to within 0.0015 spellF1 at the shipped cell.

## Limitations

- Tuning runs under simulated ball loss on IDSSE, using the provided match's *coarse*
  run-length distribution and four placement models: uniform, weighted towards duels and
  possession changes, and two that tie ball loss to dead ball (moderate and strong). How
  strongly the real tracker's ball loss is tied to dead ball is not measured here.
- **Player-only terms are fitted on another league and another tracker.** The kinematic
  models come from seven Bundesliga matches. Six coarse checks on the provided match
  (player speed scale, share of frames the model calls dead, team balance, agreement with
  the ball-based output where the ball is visible, visible players, ball speed scale) all
  passed, as PASS / FAIL only; they show the match looks alike to the model, not that the
  external gain transfers. The terms change more than 20 % of the graded labels. The
  dead-ball weight sits at the edge of the tested range, and the team term costs about
  0.01 on the bed when the ball is lost almost only during stoppages.
- **Frozen ball on the test bed.** On the external matches the ball position is frozen
  while play is stopped (speed exactly 0 on every dead frame, under 1.1 % of live frames).
  The "ball at rest" feature therefore finds dead ball almost perfectly there whenever
  the ball is not masked, which automatic tracking of a real match does not guarantee.
  Dead-ball results of the bed are optimistic in proportion to how often the ball is
  visible during a stoppage; the strongly dead-tied placement is the least affected. For
  the same reason the bed cannot say whether the ball-at-rest feature helps on real
  tracking, and no delay was fitted for stoppage onsets.
- Swapping the two teams does not exactly mirror the output where neither the ball nor the
  players decide; the team term removes most of it on the external bed (3.6–11.4 % of
  frames before; 0.3–0.5 % at the largest tested weight, not measured at the shipped one).
- Broadcast footage is unused (alignment cost); a video-based ball detector tier was
  scoped as stretch and is not in the shipped pipeline.
- The provider's dead-ball share on external matches (41–52 %) differs from what the
  prior encodes; the feature-driven D term compensates but was not re-tuned per league.
- The switch penalty sits at the low edge of the swept range (2 vs 4 within noise, both
  above 8); σ has an interior optimum after the stage-3 extension (1.0: 0.2627,
  1.5: 0.2644, 2.0: 0.2417) and the dead penalty surface is flat. Further grid extension
  had diminishing returns within the time budget.

## Reproduction

From the repository root (CPython 3.14.4; `requirements-pipeline.txt` pins the versions that
produced the committed file):

```bash
python3 -m virtualenv .venv && .venv/bin/pip install -r requirements-dev.txt
# match data in data/tactical + data/broadcast (organiser bundle; gitignored) or MTAH_DATA_DIR
.venv/bin/python -m pytest -q           # 182 tests: alignment, invariants, format, byte-identity
.venv/bin/python -m src.submit          # writes LA+AI.csv and runs the official validator
md5sum LA+AI.csv                        # c121715836ed603fc5792b85daf6f279
```

The first run builds a local cache from the raw ATD (about 1m40s); a clean clone with a cold
cache reproduces the committed file byte for byte (see `AUDIT.md`). External tuning evidence
ships in `tuning/results/` (matrices + reports); `tuning/sweep.py` and
`tuning/post_checks.py` regenerate it given `IDSSE_DIR` (see README).

## Final configuration (decoder cell frozen 2026-11-07; dead-ball term revised 2026-11-14)

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
| `d_w_low_activity` | 0 (feature off) | `external_fit` (units-corrected bed) |
| `kickoff_rule` | True | `prior` (documents-only rule) |
| `gate_mode` / `switch_delay_s` | move / 0.64 | `external_fit` (IDSSE) |
| `dwell_coalesce` | True | `external_fit` (IDSSE) |
| `kin_w_dead` / `kin_w_team` / `gate_kin_support` | 0.3 / 0.03 / True | `external_fit` (IDSSE; coefficients in `src/kin_model.json`) |

External selection: 930-cell union × 7 matches, mean spellF1 **0.2759** (95 % CI
[0.2578, 0.2932]) on the provider's per-touch labels and uniform masks, LOMO optimism
−0.0026; with the 2026-11-14 changes **0.52** on challenge-style labels with
phase-weighted masks and 0.47 when ball loss is strongly tied to dead ball. The submission regenerates byte-identically (md5-verified) and
passes the official validator.
