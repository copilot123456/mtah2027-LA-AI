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

Audit session 2026-11-07 (section 8): no evaluator run, no ground-truth read, no video inspection — checked by grep over the code and by the local `evaluation/scoreboard.csv`, which still lists the four baselines only.

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
| d_bias re-fit to IDSSE D share (45%) | −0.177 spellF1 at the stage-1 cell; −0.0847 at the shipped cell, 7/7 negative | **rejected**, prior −0.70 retained with evidence |
| X-policy on graded frames | −0.0113 at the stage-1 cell, 6/7 negative; −0.0338 at the shipped cell, 6/7 negative | **off** |

**Post-selection checks — where they ran.** `tuning/results/post_checks.csv` (2026-11-07, before
stage 2) evaluated both candidates at the **stage-1** chosen cell (`switch_pen=4, dead_pen=8,
sigma_m=3.0, min_dwell_s=2.0, switch_support_s=1.0`) under the `kinematic` scenario, i.e. with
the fill tier still on. The audit re-ran the same checks at the **shipped** cell with the shipped
fill setting (`python -m tuning.post_checks --final` → `post_checks_final.csv`): baseline 0.2759
(identical to the sweep's cell mean), d_bias re-fit −0.0847, X-policy −0.0338, both −0.1269 — all
three decisions stand under the pre-declared > 0.02 adoption rule. The default mode of
`post_checks.py` reproduces the committed file to the last digit.

**Selection rule — implementation note.** The plateau rule's "worst grid-neighbour" took its
candidate values from the stage-1 grid literal; on the stage-2/3 extension this ignored the new
notches. Fixed in the audit to use the values actually swept; re-running the final selection on
the committed stage-2+3 union gives the same cell, mean, CI, LOMO gap and worst-neighbour score
(0.2081), so the frozen configuration is unchanged (test-enforced in `tests/test_tuning.py`).

**Shipped config provenance:** `sigma_m`, `switch_pen`, `min_dwell_s`, `switch_support_s` →
`external_fit` (evidence above); `dead_pen` stays `prior` (surface flat — value immaterial);
everything else `prior`/`fixed_default`. No `provided_*` entries (test-enforced).

**Reproduction checks:** full suite green on the frozen config (64 tests at the freeze, 83
after the audit); `python -m src.submit` passes the official `validate_submission.py`; two
consecutive runs produce **byte-identical** `LA+AI.csv` (md5 `aeef0058a08860c9e2407423b1172684`),
and so does a clean clone with a cold cache and the pinned requirements. Own-output counts:
467 segments / 337 possessions / median 7.0 s / 0% sub-second / D share 26.6% of graded —
inspected for sanity only, never used to select.

**Evaluator status:** no run of the official evaluator on the provided match during P5 or
the audit. The single frozen `submission_eval` run (section 10 of the plan) remains pending
the operator, per the guard protocol.

## 8. Independent audit (2026-11-07)

Scope: correctness, compliance, reproducibility, tests, docs of the frozen P5 state
(`a6bb75a`). Full record in `AUDIT.md`. Summary of what it established:

- **Compliance:** no code path outside the vendored evaluator reads `evaluation/gt/`; the
  scorer was never run on `LA+AI.csv`; no match data, no secrets and no operator paths are
  tracked (the one hard-coded IDSSE path was removed). Vendored `evaluation/`, `starter/`
  and `games.py` untouched.
- **Label outputs:** unchanged by every audit commit — `LA+AI.csv` regenerates to the same
  md5 after each one; the committed file is now covered by a byte-identity test.
- **Frozen decisions:** re-confirmed at the shipped cell with new external evidence
  (section 7); nothing re-opened.
- **Fixes without output impact:** validator path independent of the cwd; data-driven
  neighbour rule in the selection code (same outcome); X-policy edge clause on a
  switch-free half (dormant, x_use=False); pinned requirements; README made true for a
  clean clone.
