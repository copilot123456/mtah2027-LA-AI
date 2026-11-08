# Compliance dossier — MTAH 2027 possession decoder

Updated continuously (plan P0–P6). This file is part of the submission package.

## 1. Data lineage

| Dataset | Version / source | License | Used for |
|---|---|---|---|
| Provided match assets (BAR-ATH tactical + broadcast) | organizer bundle, local only, gitignored | competition license (no redistribution) | inputs to the pipeline (ATD, homography, smart tagging) |
| IDSSE (7 Bundesliga matches) | HF mirror `pysport/idsse-data`, paper doi:10.1038/s41597-025-04505-y | CC BY 4.0 | parameter fitting/selection (P5), simulated-missingness evaluation |
| Public GT window (frames 11185–18684) | challenge repo | competition docs | **pipeline correctness only** (format/alignment), never selection |

Statements:
- No provided-video frame enters any train/validation split (there is no training on the provided game).
- No human-produced data about the provided game (event data, corrected tracking, annotations) is used.

## 2. Parameter provenance

Authoritative table: `config.py` → `PROVENANCE` (origin ∈ prior | fixed_default | external_fit).
`tests/test_config.py` fails the build if a key is undocumented or marked `provided_*`.

## 3. Ground-truth looks (append-only)

| date | what ran | by | result used for |
|---|---|---|---|
| 2026-11-07 | `evaluation/run_evaluation.py` on shipped baselines (repo smoke test) | assistant | environment validation only — no design decision taken from it |
| 2026-11-07 | `evaluation/run_evaluation.py` after baseline regeneration (byte-identical check) | assistant | reproducibility check only |

Any future evaluator run on the provided match: operator only, via the guard, result archived and never fed back into development.

## 4. AI-service statement

- No provided-match per-frame values, coordinates, labels or video frames are sent to any AI service.
- Diagnostics run locally; only coarse aggregates (rounded percentages, binned counts) appear in local documents.
- Background research agents performed web/GitHub search only; their model endpoint issues are logged in the session history.
- Organizers' clarification (2026-11-03): no uploading provided data to AI services — our operating rule is stricter than needed (coarse aggregates only) pending organizer Q3.

## 5. Video statement

The video may be inspected to understand possession (allowed). Outputs are never checked against the video; no output-dependent adjustment ever references video content.

## 6. Open organizer questions

1. Does "tune on external data" exclude fitting on the provided tracking (no GT/video)?
2. May the public GT be used to evaluate (not fit) a frozen pipeline?
3. May cloud AI assistants read aggregate diagnostics computed from the provided match?

Answers will be appended here verbatim when received.

## 7. External tuning record (P5, closed 2026-11-07)

**Data.** IDSSE, 7 full Bundesliga 2022/23 matches, CC BY 4.0 (HF mirror `pysport/idsse-data`;
files outside the repo, `IDSSE_DIR`). Labels derived algorithmically from provider fields
(`ball_state` → D, `ball_owning_team_id` → side); `owner_filled = 0` on all matches (no
imputation needed). No human annotation anywhere. Missingness simulation: uniform seeded
placement of run lengths sampled from the provided match's *coarse* structural distribution
(150 runs, median 268 frames; missing-count/gap-distribution diagnostics = GREEN), target 43.1%.

**Matrices** (all under `tuning/results/`, committed):
| stage | cells | rows | integrity |
|---|---|---|---|
| 1 | 243 × 2 scenarios × 7 | 3,402 | — |
| 2 (extended) | 768 × 7 (`none`) | 5,376 | 1,701 overlaps reproduce stage 1, max abs diff 0.00e+00 |
| 3 (σ floor) | 324 × 7 (`none`) | 2,268 | 1,134 overlaps reproduce stage 2, max abs diff 0.00e+00 |
| final union | **930** | — | deterministic end-to-end |

**Selection (pre-declared rule):** within 0.01 of best mean spellF1, maximize the worst
grid-neighbour; scenario `none`. Final cell: `switch_pen=2, dead_pen=5, sigma_m=1.5,
min_dwell_s=2.0, switch_support_s=1.5` — mean **0.2759** (best 0.2854, gap 0.0095),
match-level bootstrap 95% CI [0.2578, 0.2932], **LOMO optimism −0.0026** (folds pick the
same neighborhood: switch 2 in 7/7, σ 1.5 in 6/7, dwell 2 in 7/7). Marginal turns verified
for σ (1.0: 0.2627, 1.5: 0.2644, 2.0: 0.2417) and dwell (2.0 peak); switch/support/dead flat
at noise level.

**Layer / adoption decisions (all external, all evidence-gated):**
| Decision | Evidence | Outcome |
|---|---|---|
| Kinematic fill tier | paired −0.0112 spellF1, CI [−0.017, −0.007], 7/7 negative; all-cell −0.0089 | **dropped** (`fill_long_gap_s=0`, external_fit) |
| d_bias re-fit to IDSSE D share (45%) | −0.177 spellF1 | **rejected**, prior −0.70 retained with evidence |
| X-policy on graded frames | −0.0113, 6/7 negative | **off** |

**Shipped config provenance:** `sigma_m`, `switch_pen`, `min_dwell_s`, `switch_support_s` →
`external_fit` (evidence above); `dead_pen` stays `prior` (surface flat — value immaterial);
everything else `prior`/`fixed_default`. No `provided_*` entries (test-enforced).

**Reproduction checks:** full suite 64/64 green on the frozen config; `python -m src.submit`
passes the official `validate_submission.py`; two consecutive runs produce **byte-identical**
`LA+AI.csv` (md5 verified). Own-output counts: 467 segments / 337 possessions / median 7.0 s /
0% sub-second / D share 26.6% of graded — inspected for sanity only, never used to select.

**Evaluator status:** no run of the official evaluator on the provided match during P5.
The single frozen `submission_eval` run (section 10 of the plan) remains pending the
operator, per the guard protocol.
