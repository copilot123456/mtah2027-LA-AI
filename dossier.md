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
| 2026-11-09 | guarded run via `tuning/guarded_eval.py` → `run_evaluation.py --bootstrap 1000` on the frozen `LA+AI.csv` (public GT); full board archived `evaluation/reports/guarded_board_*.txt`, operator eyes only | operator | **band C**; flags `count 23–30 · f1_D low · balance normal · latency none` → **Gate 0 proceed** (sanity clean, band ∈ {A, B, C}); orders the P6 backlog per plan §1 — E2/E1 promoted, E3 demoted, D-feature audit (E7 variant) after E2; diagnosis: dead-ball mis-calibration (not over-merging, not latency); no design decision taken from the score; a second run only if a code path changes before the freeze |
| 2026-11-11 | assistant opened the local `evaluation/scoreboard.csv` (gitignored, untracked) while preparing a submission-simulation answer — the LA+AI row from the 2026-11-09 guarded run (full board) entered the AI session | assistant (logged by operator decision 2026-11-11) | **incident — no use**: numbers never quoted back and never used in any decision; file verified never in git (`.gitignore:27`; `evaluation/reports/` also ignored) and a single run only (board mtimes 2026-11-09 14:05 — no re-run); no rule changed and nothing re-run — standing fix: board files are never opened assistant-side, band + flags by operator report only |

Any future evaluator run on the provided match: operator only, via the guard, result archived and never fed back into development.

2026-11-14 (P7, §7): no evaluator run. `config.py` and a code path changed, so a second guarded run is owed before submission — operator only, band + flags, correctness gate only.

Audit session 2026-11-07 (section 8): no evaluator run, no ground-truth read, no video inspection — checked by grep over the code and by the local `evaluation/scoreboard.csv`, which still lists the four baselines only.

## 4. AI-service statement

- No provided-match per-frame values, coordinates, labels or video frames are sent to any AI service.
- Diagnostics run locally; only coarse aggregates (rounded percentages, binned counts) appear in local documents.
- Background research agents performed web/GitHub search only; their model endpoint issues are logged in the session history.
- Organizers' clarification (2026-11-03): no uploading provided data to AI services — our operating rule is stricter than needed (coarse aggregates only) pending organizer Q3.
- **Audit + P7 session 2026-11-14 (cloud assistant):** the provided match data, the ground truth, the evaluator board files and the video were not opened. What crossed into the session from the provided match: three categorical checks (speed units = m/s; kick-off PASS/FAIL clauses; label-change bucket) and the own-output summary `src.submit` prints (label counts, possession count and mean length).
- **Incident 2026-11-11 (disclosed, no use):** while preparing a local submission simulation the assistant read the gitignored `evaluation/scoreboard.csv`, which carries the full board of the 2026-11-09 guarded run; those numbers entered one AI session. They were never quoted, never used for a decision, and the file has never been in git or in any shared artifact. Standing fix: evaluator/board artifacts are never opened assistant-side — only the band letter and the categorical flags cross to the assistant.

## 5. Video statement

The video may be inspected to understand possession (allowed). Outputs are never checked against the video; no output-dependent adjustment ever references video content.

## 6. Open organizer questions

1. Does "tune on external data" exclude fitting on the provided tracking (no GT/video)?
2. May the public GT be used to evaluate (not fit) a frozen pipeline?
3. May cloud AI assistants read aggregate diagnostics computed from the provided match?
4. May a pretrained model run with a-priori parameters on the provided tactical video feed the pipeline (video as an input source, beyond "inspection to understand")? And are weights fine-tuned on provided-match material usable? (Jose's notebooks 02/04 assume video inference is allowed; `ball_ft_v3` training provenance unknown — asked 2026-11-09 after his notebook review.)

5. May the published full-match baseline scores (EVALUATION.md) be used to sanity-check an external test bed — to judge which simulated ball-loss regime resembles the match — without counting as calibration on the ground truth? (Added 2026-11-14; decides D1 R2, §7 P7.)

**Status 2026-11-14: Q1–Q5 not yet sent.** The clarification of 2026-11-03 already settles half of Q4: a pretrained model may run on the provided video as input; weights trained on the provided video may not be used.

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
| X-policy on graded frames | spellF1 **identical** to the shipped path at both cells (the scorer forward-fills `X` inside a possession, so a stable-interior `X` reproduces the decoded label); macroF1 −0.079 at the stage-1 cell, −0.110 at the shipped cell, 7/7 negative | **off** (tiebreak loses, nothing to gain) |

**Post-selection checks — where they ran.** `tuning/results/post_checks.csv` (2026-11-07, before
stage 2) evaluated both candidates at the **stage-1** chosen cell (`switch_pen=4, dead_pen=8,
sigma_m=3.0, min_dwell_s=2.0, switch_support_s=1.0`) under the `kinematic` scenario, i.e. with
the fill tier still on. The audit re-ran the same checks at the **shipped** cell with the shipped
fill setting (`python -m tuning.post_checks --final` → `post_checks_final.csv`): baseline 0.2759
(identical to the sweep's cell mean), d_bias re-fit −0.0847, X-policy ±0.0000 spellF1 /
−0.110 macroF1, both −0.0847 — all three decisions stand under the pre-declared > 0.02 adoption
rule. The default mode of `post_checks.py` reproduces the stage-1 record (`post_checks.csv`).

**Correction (2026-11-07, found by the P6 planning pass, verified against the vendored scorer).**
Until this date `tuning/sweep.py::_score` and `tuning/post_checks.py` scored predicted `X` without
the forward-fill that the official `evaluate()` applies (`fill_abstentions`), so the X-policy
ablation counted `H X X H` as two possessions. The P5 sweep never writes `X` on graded frames and
is unaffected (verified: no `X` inside any graded run at any cell); the earlier X-policy figures
(−0.0113 stage-1, −0.0338 shipped) were artifacts. Both post-check matrices were regenerated with
the corrected scorer (previous versions in git history at `a6bb75a` and `d60b66d`); the d_bias
rows are unchanged to the last digit, the X rows now show the spellF1 tie and the macroF1 loss.
The harness scorer is pinned to `evaluate()`'s semantics by a test.

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

### P6 pre-declarations

#### E10 — generic no-play rule (pre-declared 2026-11-09, before implementation and before the run)

- **Authorization:** operator approval 2026-11-09 ("if we can confirm the theory … then yes");
  condition confirmed by `reference-docs/FAQ.md` (second half graded from 110,808, kick-off at
  113,742, "those 2,934 frames … are all `D`"). The check itself was measured 2026-11-07
  (plan §0): our output is 37 % `D` on those frames.
- **Rule (exact, generic):** inside `decode()`, after gating and min-dwell, mark a frame `D`
  iff it lies in a maximal run of ≥ `no_play_s` = **10.0 s** (a priori: same scale as the
  existing `d_missing_run_s` prior; 10 s of unbroken low activity is unambiguously a stoppage)
  of consecutive frames whose per-frame mean player speed is < `d_activity_thr_ms` = 2.0
  (existing prior). The whole qualifying run is marked, including its first `no_play_s`.
  Runs are computed within one decoded half only — never across the halftime hole. New cfg
  keys: `no_play_rule` (bool; default OFF until the gate below passes), `no_play_s` (10.0,
  origin `prior`). No frame numbers, no match identifiers, no ball-position conditions. The
  centre-spot clause of plan §2 is dropped because DATA.md documents the camera/homography as
  unreliable across halftime (ball position cannot be verified exactly where the rule must
  fire); the FAQ fact is a match-state fact, not a ball-state fact.
- **Why this is a compliance fix, not a tune:** the FAQ states the labels for this situation
  outright; the rule implements a documented property of the challenge using existing priors
  plus one a-priori dwell — no number is fitted to any data.
- **Gate (a priori):** no-harm on IDSSE — frozen cell (config as shipped), all 7 matches,
  paired rule-off vs rule-on, same seeds; **pass iff** mean paired Δ spellF1 ≥ −0.005 **and**
  no single match worse than −0.02 (mean within the bootstrap noise floor of the frozen cell's
  CI; per-match bound = the standard evidence-rule harm bound). Fail ⇒ the flag stays OFF and
  the result is reported as-is.
- **Expected own-output effect:** the pre-kickoff window moves from 37 % to ≈ 95 % `D`;
  plan §2's estimate stands (0 … +0.005 spellF1).
- **If adopted:** `LA+AI.csv` is regenerated in the same branch commit (new md5 documented in
  the branch report); adoption into the final branch remains a team decision.

#### E1 — challenge-definition labels (pre-declared 2026-11-09, operator: "proceed with E1")

- **Rule (exact; lands as `derive_letters_controlled()` in `tuning/idsse.py`):** process the
  maximal alive runs (H/A) of the original IDSSE-derived letters left to right; `D`/`X` is a
  stoppage. The first alive run after a stoppage starts a **fresh possession — never
  absorbed**. An alive run **shorter than `T`** that is directly preceded by another alive run
  (no stoppage between) is relabelled to the preceding team ("loose or contested ball stays
  with the last team"; CHALLENGE.md: "a player keeps the ball for a few seconds"); the
  absorbing team persists — **chaining merges**. An alive run **≥ `T`** is a genuine
  possession change. `D`/`X` frames never change (dead share invariant by construction).
- **`T` = 3 s primary, a priori** (CHALLENGE.md "a few seconds"; plan §2; operator approval
  2026-11-09). **Sensitivity `T` ∈ {2, 5} s** — reported in the statistics table, never used
  to select: plan §0 correction and do-not-do 1 forbid choosing `T` by matching the public
  window.
- **Purpose now:** the label-statistics table only. Scoring happens solely inside the
  pre-declared E1→E2 combined 144-cell pass (plan §2 protocol: primary table aligned labels ×
  phase masks, plateau rule, adopt only > 0.02 paired and not worse > 0.02 elsewhere) — that
  pass's grid and rule go into this section *before* it runs.
- **Fidelity probe (pre-registered):** statistics at `T` = 3 s must land near the plan §0
  audited probe (mean 2.49 possessions/min, 13.4 s mean length, 0.03 share under 2 s over the
  7 matches). A material mismatch means this implementation differs from the probed rule and
  must be reconciled before any compute.

#### E10 redesign — feasibility probes and closure (2026-11-09, operator decision: A, close as infeasible)

- **Trigger:** operator proposed re-deriving E10 from kickoff geometry — ball at the centre
  spot, a kicker beside it, players near no movement, formation stable/static (motivated by
  public broadcast stills of a kick-off; no eval video used).
- **Method:** four read-only probes of the target window (frames 110,808–113,741) on our own
  arrays — no GT, no video, no scoring run. Every tested value was an existing prior
  (`sigma_m` 1.5, `ball_stationary_thr_ms` 1.0, `d_activity_thr_ms` 2.0, `no_play_s` 10.0);
  no threshold was chosen from any measurement.
- **Findings (own-output aggregates):**
  - ball within 1.5 m of the centre spot: **0.0 %** of window frames; ball at rest < 1 m/s:
    6.1 %. Camera is not the obstacle — reprojection error at the window is 0.20 (play-quality).
  - a player within 1.5 m of the centre: **0.0 %** (closest approach 1.93 m, final 0.5 s).
    At 9.15 m (centre circle): 57.6 % of the window vs **96.2 %** of a live-play contrast
    span — centre proximity fires *more* in live play than in the window (anti-discriminative).
  - formation static: median per-slot displacement across the window **29.9 m** vs 36.3 m on
    the live-play contrast — the window is warm-up milling, not a formed-up restart; the
    kick-off-stills geometry exists only in the last ~10 s (89.6 % of the final 250 frames
    within 5 m of centre), i.e. ~4 % of the window, and the window ends before formation.
  - players quiet (< 2 m/s): 95.5 % of the window vs 56.6 % live — the only clause that
    covers the window is the round-1 D-hammer on its own (30.9 % of graded frames, retracted).
  - the four-clause conjunction (quiet ∧ ball at rest ∧ centre ∧ kicker): **0.0 %**.
- **Conclusion:** the plan §2 sanctioned fix (ball at rest near the centre spot, no player
  movement) has no firing opportunity in the target window; the FAQ's all-D window is a
  match-state fact that the allowed generic, frame-number-free signals do not expose in this
  tracking. Signatures derived from the window itself (e.g. an "unattended warm-up ball")
  are barred by plan §4.1/§4.5 (no fitting to the public window) and were not pursued.
- **Disposition A (operator, 2026-11-09):** E10 closed as infeasible under the rules —
  `no_play_rule` stays OFF, the 37 % deficiency stays documented as unobservable from the
  allowed signals, **no gate run taken** (a run on a rule measured inert would be theater).
  Reopening requires either an organizer answer or a rule shaped from documents alone.

#### E2 + E3 + E11 combined pass — grid G6, tables, simulator, rule (pre-declared 2026-11-09, before implementation and before the run)

- **Authorization / trigger:** plan §2 combined protocol (pre-existing), operator NEXT
  instruction (E2 → pin → run → Gate 1 2026-11-10), Gate 0 read (band C: E2/E1 promoted,
  E3 demoted, dossier §3). E1 is already pinned (T = 3 s) and implemented; scoring of E1
  happens only inside this pass.
- **Grid G6 = 144 cells** (plan §2, verbatim): `switch_pen` {1, 2, 4} × `dead_pen` {5, 8, 12}
  × `sigma_m` {1.5, 2.0} × `min_dwell_s` {1.0, 1.5, 2.0, 3.0} × `switch_support_s` {1.0, 1.5},
  parameter order `tuning.sweep.PARAMS`. The frozen cell (2 / 5 / 1.5 / 2.0 / 1.5) is in the
  grid. E3's demotion (band C) means: its notches stay for neighbour continuity, no separate
  E3 adoption path — the uniform rule below decides everything.
- **Simulator (exact, all cues external to the provided match):**
  - cue 1 — *possession change*: frame `t` where an alive (`H`/`A`) frame of the **original**
    IDSSE letters follows a non-alive frame or a direct team change; dilated ±2 s (50 frames
    at 25 fps). Original letters are used (the provider-faithful change points; E1's aligned
    set is a scoring interpretation and is not coupled to the mask).
  - cue 2 — *crowded*: ≥ 3 finite players within 3 m of a finite raw-ball frame (metres via
    `CFG` pitch dimensions — the same convention `emission._team_evidence` uses on IDSSE);
    ball missing → false.
  - cue 3 — *tackle*: ±2 s (same a-priori window) around `TacklingGame` parent events from
    `data/idsse/DFL_03_02_events_raw_*.xml`, mapped to tracking rows through per-period
    timestamp offsets (tracking `timestamp` = µs since period start) anchored on the events'
    `StartFrame` attributes — per-period median offset, anchor residual reported per match
    (J03WPY: std 1.3 s / 1.7 s; all 7 matches carry 12–23 anchors and 173–229 tackles;
    validation: 100 % of mapped J03WPY tackle rows have finite raw ball). Unmappable events
    are dropped and counted.
  - weight: `cue_t = c1 + c2 + c3 ∈ {0..3}`, `w_t = 1 + α·cue_t`, **α = 2 primary** (docs:
    missingness "disproportionately on crowded boxes, aerial duels, tackles"), **α = 3
    sensitivity** (never selects). Placement: `mask_ball` mechanics unchanged (empirical run
    lengths from the provided match's graded missingness — the P5 structural diagnostic,
    target 43.1 %, non-overlap rejection); only the start-frame sampling changes from uniform
    to `∝ w_t`. Same seed per match as P5 (`MASK_SEED + Σ ord(match_id)`) for every mask kind.
- **Tables — six decode runs, four decision tables + two reported twins**, all 144 cells ×
  7 matches, fill off unless stated, each run scored on **both** label sets (original
  `derive_letters`, aligned `derive_letters_controlled` T = 3 s) with `sweep._score`
  (official X-fill semantics):

  | run | mask | fill | scored on | role |
  |---|---|---|---|---|
  | T1 | uniform (P5 mechanics) | off | orig + aligned | decision table |
  | T2 | phase α = 2 | off | orig + aligned | decision table; **primary = T2 × aligned** |
  | T3 | phase α = 2 | **on** (kinematic tier) | orig + aligned | E11 re-check, reported only |
  | T4 | phase α = 3 | off | orig + aligned | sensitivity, reported only, never selects |

- **Selection rule (verbatim plan §2):** plateau rule on the primary table (within 0.01 of
  the best mean, maximise the worst swept grid-neighbour, ties by mean) → candidate. Adopt the
  candidate only if its primary-table mean beats the frozen cell's by **> 0.02 paired with
  ≥ 5/7 matches positive** AND it is not worse than frozen by > 0.02 on any of the other three
  decision tables. Delta 0.01–0.02 → tie → keep frozen; < 0.01 → reject. Otherwise the frozen
  cell ships and the tables go into the report as robustness evidence. LOMO and the
  match-level bootstrap CI are reported for whichever cell ships. Fill (E11) adopts only
  through the same > 0.02 bar on its own parameter.
- **Pre-registered mask statistics** (`tuning/results/stage4_masks.csv`, reported in
  `report_stage4.md`): per match × mask kind — overall masked share; masked share inside vs
  outside the cue-1 window (the plan's "external switches vs overall"); the same inside/outside
  split for cue 2, cue 3 and any-cue.
- **Outputs:** `tuning/results/stage4_uniform.csv`, `stage4_phase_a2.csv`,
  `stage4_phase_a3.csv`, `stage4_phase_a2_fill.csv`, `stage4_masks.csv`,
  `report_stage4.md`; runner `tuning/stage4.py`; 7 workers (plan §0 compute note), ≈ 45–60 min.
- **Boundaries honored:** no provided-match value enters the simulator beyond the already-
  accepted run-length structural diagnostic (P5, GREEN); α is a priori (Q1 pending — calibrating
  α on provided aggregates stays barred); no window/GT/video input; nothing is selected on the
  public window; second evaluator run only if a code path changes before the freeze.

##### Gate 1 read (2026-11-09, same day — matrices `tuning/results/stage4_*.csv`, report `report_stage4.md`)

- **Run:** 1921 s wall, 7 workers; 144 cells × 7 matches × 4 runs, both label sets scored;
  TacklingGame mapped 100 % on every match (173–229 events, 12–23 anchors each).
  Mask statistics confirm the simulator did its job: uniform flat (anycue 0.428 in /
  0.435 out — the control), phase α=2 lifted on every cue (0.475 / 0.410), α=3 harder
  (0.488 / 0.404), consistent across the 7 matches.
- **Primary (phase × aligned) plateau:** candidate `switch_pen=1 / dead_pen=12 /
  σ=1.5 / dwell=2 / support=1` mean **0.3699** vs frozen **0.3667** (frozen rank 12/144;
  best 0.3731; 23/144 cells within 0.01 of best).
- **Conditions:** paired Δ **+0.0032** (needs > 0.02) ✗ · positive **3/7** (needs ≥ 5/7) ✗ ·
  no-harm on the other three decision tables ✓ (+0.0014 / +0.0032 / +0.0044) ·
  bootstrap 95 % CI **[−0.0078, +0.0179]**.
- **Verdict: REJECT → the frozen cell ships** (Δ ≤ 0.01 under the pinned rule);
  `LA+AI.csv` md5 `aeef0058a08860c9e2407423b1172684` unchanged, no config touched.
- **LOMO:** fold pick = global pick on **all 7 folds** — the selection is stable; the
  candidate is simply not better than frozen by the bar.
- **Twins (reported only):** fill-on (E11) = **+0.0000** for both cells — the fill tier
  adds nothing under realistic missingness, P5's drop stands; α=3 picks the **same** cell
  and scores slightly worse (−0.0087 candidate / −0.0040 frozen) — sensitivity confirms
  the shape, not a different choice.
- **E2 hypothesis read:** the plateau moved exactly as predicted (lower switch penalty,
  shorter support) but at noise magnitude — under phase-correlated missingness the gating
  bottleneck is worth ≤ +0.003, far under the adoption bar. E3's notches (folded in,
  demoted by band C) did not clear it either. The four tables + twins + mask statistics
  are the pre-declared robustness evidence for the frozen cell.
- **Reproduce:** `.venv/bin/python -m tuning.stage4` (≈35 min) →
  `.venv/bin/python -m tuning.gate1` (read + report) → tests green.

#### E7 — set-piece dead-ball prior + band-C D-feature audit (pre-declared 2026-11-09, before implementation and before the gate)

- **Trigger:** plan §2 rank 6 (pre-existing) + band-C read (dossier §3: `f1_D` low →
  D-feature audit folded into E7) + Gate 1 finished a day early, so E7 pulls forward
  per plan §3 ("E7 only if early").
- **Feature (exact):** `f_sp(t) = 1` iff a set-piece tag start lies in
  `[start − sp_pre_s, start + sp_post_s]` with `sp_pre_s = 2.0`, `sp_post_s = 1.0`
  (plan §2 E7 window, a priori). It enters **only** the dead logit:
  `z += d_w_set_piece · f_sp`, `d_w_set_piece` = **0.0 (OFF)** with the ON value pinned
  at **1.0** (plan §2 "weight prior 1.0"). Timing only — no team, no side, no type
  distinction (`Team` is `N/A` on throw-ins and is ignored entirely).
- **Inputs:**
  - *Provided match (inference input, allowed):* smart tagging `code == "SET PIECES"` —
    59 instances (Throw-in 26, Corner 12, Goal Kick 10, Free Kick 5, Kick Off 6),
    `start_frame = round(seconds · 25)`.
  - *IDSSE (gate + audit):* event children `ThrowIn`/`FreeKick`/`GoalKick`/`CornerKick`/
    `KickOff` — 90–108 per match (J03WPY 93 = plan's ≈93) — parent `EventTime` mapped to
    rows by the same StartFrame-anchored per-period alignment as E2 cue 3, then
    **simulated tagging noise, all a priori** (plan §2 E7): jitter ±1 s uniform, drop 25 %,
    add `round(0.10 × n_real)` spurious events at uniformly random frames; one seeded
    stream per match, `default_rng(MASK_SEED + 700 + Σ ord(match_id))`.
- **Gate (a priori):** frozen cell only, `d_w_set_piece` 0 → 1, 7 matches, the four
  decision tables (uniform/phase × orig/aligned), **primary = phase α=2 × aligned**
  (the standing Gate-1 primary). Pass iff paired Δ > 0.02 **and** ≥ 5/7 positive **and**
  no-harm (≥ −0.02) on the other three tables. Fail → stays OFF. The plan's own prior
  (+0.00 … +0.01) sits **below** the bar: the expected outcome is OFF unless surprised.
- **D-feature audit (band C, read-only):** frozen cell, feature OFF, both masks × both
  labels — per-frame D confusion and each dead feature's firing rate conditioned on
  {D-correct, missed-D, false-D}, plus `f_sp` coverage of missed-D frames (the external
  upside estimate). Report only; the audit never adjusts a weight — the gate decides alone.
- **Implementation surface:** `config.py` (+`d_w_set_piece`, `sp_pre_s`, `sp_post_s`,
  provenance), `src/emission.py` (`f_sp` + z term), `src/data.py` (`MatchArrays.setpiece`),
  `src/steps.py` (one pass-through line), `tuning/e7.py` (gate + audit + report). The
  `pipeline_spec.py` contract is **unchanged** (the field rides inside `arrays`; stage
  names and requires/provides untouched). With `d_w_set_piece = 0` the z-vector is
  bit-identical → `LA+AI.csv` regeneration stays byte-identical. If the gate ever passes,
  the adoption follows the standard's external-evidence rule and is shared with the team
  at the Wednesday convergence.
- **Outputs:** `tuning/results/e7_gate.csv`, `tuning/results/report_e7.md`, dossier entry.
- **Boundaries:** tagging is an inference input only (timing), never a fitting signal;
  noise parameters a priori from plan §2; no window/GT/video; nothing here re-opens the
  frozen cell's other parameters.

##### E7 gate + D-feature audit (2026-11-09, same day — `e7_gate.csv`, `e7_audit.csv`, `e7_noise.csv`, `report_e7.md`)

- **Run:** 14 ON decodes + a fresh-OFF cross-check; set pieces mapped **100 %** on all
  7 matches (90–108 raw → noisy mean 84 + 9 spurious each); **OFF fresh-vs-committed
  drift 0.000000** — masks, seeds and the code path reproduce the committed stage-4
  rows exactly.
- **Verdict: REJECT → `d_w_set_piece` stays 0.0.** Primary (phase × aligned) Δ
  **−0.0075**, 2/7 positive; all four tables negative (−0.0057 … −0.0112). With noisy
  tag timing the ON value pushes D where the Viterbi path disagrees — the plan's
  +0.00 … +0.01 prior was optimistic and the > 0.02 bar did its job. `LA+AI.csv`
  untouched (md5 `aeef0058a08860c9e2407423b1172684`).
- **Audit (band-C `f1_D` low, external picture, phase masks, 7-match means):**
  missed-D is a **ball-absent** phenomenon — `long_missing` fires on **53.0 %** of
  missed-D frames (base 19.2 %) while `stationary` (1.3 %) and `near_boundary`
  (1.2 %) cannot speak without ball evidence; `set_piece` covers only **6.9 %** of
  missed-D (base 4.4 %) — a real but tiny upside, consistent with the gate. False-D
  also rides `long_missing` (28.3 %). Reading: the band-C weakness is dead-feature
  *availability* under a missing ball, not missing priors. Recorded as a freeze-time
  limitation candidate; any fix would need its own pre-declared > 0.02 gate — nothing
  changed today.
- **Reproduce:** `.venv/bin/python -m tuning.e7` (~20 min first run incl. event
  parsing, ~3 min with warm caches) → `python -m pytest -q`.

### P7 — audit follow-up (pre-declared 2026-11-14, before any stage-5 compute)

- **Trigger / authorization:** design-and-process audit of 2026-11-14 (operator-requested);
  operator instruction the same day: "do all identified fixes that are unblocked". This
  overrides Gate 2's "no new ideas after 11-12" cut-off by operator decision. Work stays on
  `ruiz-work`; `main` stays locked; no push without operator instruction.
- **Disclosure — what the audit saw before this pin (external data only, repository
  untouched, provided match / GT / video / board files never opened):** the frozen cell and
  the organisers' baselines on IDSSE under uniform masks, two dead-weighted masks (start
  weight 1 + 8·dead, 1 + 60·dead), a no-mask run, and the two dead-cover masks defined
  below; six D-term probe cells (low-activity weight 0, threshold 1.0 / 1.5, long-missing
  weight 0 / 2, missing-run 3 s). Those probes shaped the grid ranges below. The audit's
  lead came partly from the band-C read (§3) and from the own-output aggregates already in
  this section (E10 probes) — recorded here so the provenance of the idea is on file. The
  phase α = 2 tables of stage 5 were **not** seen before this pin.
- **Unit check on the provided match (2026-11-14):** categorical output only — reported
  player speed vs position-derived speed classifies as **m/s** on the provided match and
  as **km/h** on IDSSE. No value from the provided match was printed.

#### U0 — test-bed units defect and fix

- **Defect:** the IDSSE `*_s` columns are km/h (median per-frame mean speed in live play
  7.8–8.2 as reported vs 2.2–2.3 m/s from positions; ratio 3.6), while
  `d_activity_thr_ms = 2.0` is m/s and the provided match reports m/s. On the bed the
  `low_activity` dead-ball feature therefore fired on 0.0 % of live frames and 12–17 % of
  dead frames; in m/s it fires on 36–42 % / 95–96 %. Every P5/P6 matrix and gate was
  computed with the feature effectively off; the shipped file was produced with it on.
  The same defect explains E10 (no-play rule: 0–1 external matches vs 30.9 % of our
  graded frames) — the gate was not vacuous by football, it was vacuous by units.
- **Fix:** `tuning/idsse.py` converts `players_s` to m/s at parse time and migrates
  existing caches at load, guarded by `speed_unit_factor` (ratio of reported speed to the
  1-s chord speed: 0.7–1.4 → m/s, 2.9–4.5 → km/h, anything else is an error).
  `tests/test_units.py` pins both loaders (IDSSE → m/s after the fix; provided match →
  m/s, assertion only).
- **Historical matrices:** left as committed (they are the record of what ran). Stage 5
  reports the drift between the corrected bed with `d_w_low_activity = 0` and the
  committed stage-4 frozen-cell rows.

#### Stage 5 — one pass, four mask kinds, both label sets

- **Fixed:** frozen decoder cell (`switch 2 / dead 5 / σ 1.5 / dwell 2.0 / support 1.5`),
  fill off, corrected units, same per-match seed as P5 for every mask kind.
- **Mask kinds:** `uniform` (P5 mechanics), `phase_a2` (stage 4, α = 2),
  `cover_q50`, `cover_q90` — *dead-cover* placement: every maximal dead run of the
  original letters lasting ≥ 2 s is masked entirely with probability `q` (one draw per
  run, chronological, per-match seeded generator); P5 run lengths are then placed
  uniformly, rejecting overlaps, until 43.1 % of in-match frames are masked. `q` = 0.5 and
  0.9 are a priori (a moderate and a strong coupling of ball loss to dead ball); nothing
  from the provided match enters beyond the already-accepted run-length diagnostic.
- **Cells:** grid G7 over the dead-ball term —
  `d_w_long_missing` {0, 0.5, 1.0, 2.0} × `d_missing_run_s` {3, 10} ×
  `d_w_low_activity` {0, 0.5, 1.0, 2.0} × `d_activity_thr_ms` {1.0, 1.5}, a zero weight
  counted once (49 unique cells) — plus the **as-shipped** cell (`d_w_low_activity` 0.5,
  threshold 2.0, long-missing 0.5, run 10 s). `d_bias` and the other weights stay at their
  shipped priors (the bias re-fit was already rejected; the dead share is
  definition-specific).
- **Outputs:** `tuning/results/stage5_<kind>.csv`, `stage5_baselines.csv` (B1),
  `stage5_masks.csv`, `report_stage5.md`; runner `tuning/stage5.py`, reader
  `tuning/gate5.py`.

#### U1 — low-activity weight (gate)

- **OFF** = `d_w_low_activity = 0` (long-missing 0.5 / 10 s); **ON** = as shipped.
- **Rule:** adopt OFF iff on the standing primary table (phase α = 2 × aligned) the paired
  mean Δ(OFF − ON) > 0.02 with ≥ 5/7 matches positive, **and** Δ ≥ −0.02 on each of the
  other seven tables (4 kinds × 2 label sets). Otherwise the shipped value stays.
- **Not blind:** the audit already measured OFF − ON on uniform (+0.061 aligned) and on
  both cover masks (+0.040 / +0.037); the phase tables are the unseen part.

#### D1 — dead-ball term re-sweep (two readings, only R1 is automatic)

- **Reference** = the U1 outcome. Per cell, paired Δ vs the reference on aligned labels.
- **R1 — regime-agnostic (automatic):** `g(cell)` = minimum over the three coupling
  regimes {`uniform`, `cover_q50`, `cover_q90`} of the 7-match mean Δ. Candidate =
  argmax `g`; cells within 0.005 of the best are tied and the tie goes to the cell with
  the fewest parameters changed from the reference, then the smallest summed weight
  change. **Adopt iff** `g` > 0.02 **and** ≥ 5/7 positive in each of the three regimes
  **and** Δ ≥ −0.02 on `phase_a2` × aligned and on all four original-label tables.
  Rationale: the coupling of ball loss to dead ball on the provided match is not measured
  (strict stance, Q1 open), so the decision must hold whichever regime is true.
- **R2 — coupled-regime reading (operator option, never automatic):** the same with
  regimes {`cover_q50`, `cover_q90`} only; `uniform` and `phase_a2` are reported without
  a veto. R2 is only defensible if the published full-match baseline scores
  (EVALUATION.md) may be used to judge which bed regime resembles the match — on the bed
  the baselines' macroF1 is ≈ 0.45 under uniform masks and 0.60 / 0.77 under the cover
  masks, against the published 0.64–0.68. That is a compliance call for the operator
  (new organiser question Q5 below); nothing is adopted from R2 without it.
- **Own-output check after any adoption:** share of graded frames whose label changes,
  reported as a bucket only (< 2 % · 2–5 % · 5–10 % · 10–20 % · > 20 %). U1 has no veto
  (it restores the externally evidenced state); a D1 adoption above 20 % is held for the
  operator (E10 lesson).

#### K1 — kick-off tag rule (documents-only rule; reopens E10 under its closure clause)

- **Basis:** FAQ ("the second half is graded from frame 110,808 but does not kick off
  until 113,742 … those 2,934 frames are graded and all of them are `D`") and DATA.md
  (Smart Tagging `Type` = `Kick Off (Start)`, 2 instances; time base "checked against
  both kickoffs"). The E10 closure allowed reopening with "a rule shaped from documents
  alone".
- **Rule (exact, generic):** for each graded half, let `k` be the earliest start frame of
  a Smart Tagging `SET PIECES` instance whose `Type` is `Kick Off (Start)`
  (case-insensitive) and whose start lies inside that half's graded span. If `k` exists,
  frames from the half's first graded frame up to `k − 1` are labelled `D`, after
  decoding, gating and min-dwell, before the X policy. No such tag → no-op. New key
  `kickoff_rule` (bool). No frame numbers, no match identifiers.
- **Verification (categorical PASS / FAIL only, `tuning/kickoff_check.py`):** exactly one
  such tag inside each graded half; first-half span ≤ 250 frames; second-half span within
  ±250 frames of the FAQ's 2,934. Any FAIL → the flag stays OFF.
- **External gate:** none possible — IDSSE halves start at the kick-off and carry no
  Smart Tagging, so the rule cannot fire there (the E10 firing-opportunity lesson). The
  adoption rests on the two documents and today's operator instruction; expected effect
  0 … +0.005 spellF1 plus the tie-break.

#### B1 — baseline control (report only)

- The organisers' nearest-player baseline (raw, ±25, ±50, ±125 frames) scored on every
  stage-5 mask kind, both label sets, spellF1 and macroF1. No decision is taken from it.

#### New organiser question

5. May the published full-match baseline scores (EVALUATION.md) be used to sanity-check
   an external test bed — i.e. to judge which simulated missingness regime resembles the
   match — without counting as calibration on the ground truth?

##### P7 read (2026-11-14, same day — `stage5_*.csv`, `stage5_verdicts.csv`, `report_stage5.md`)

- **U0 done:** all seven IDSSE caches migrated to m/s (source factor 3.6 on each);
  `tests/test_units.py` pins the self-check and both loaders.
- **Run:** 904 s wall, 6 workers; 50 cells × 7 matches × 4 mask kinds, both label sets.
  Mask statistics (7-match means, P(missing | dead) / P(dead | missing)): uniform
  0.43 / 0.44 · phase_a2 0.39 / 0.41 · cover_q50 0.64 / 0.66 · cover_q90 0.90 / 0.92.
- **U1 — ADOPT, `d_w_low_activity` 0.5 → 0.0.** Primary (phase_a2 × aligned)
  **+0.0492**, 7/7; positive 7/7 on all eight tables (+0.0249 … +0.0607). Corrected bed
  with the weight off vs the committed stage-4 frozen-cell rows: mean |Δ| ≤ 0.0015 — the
  P5/P6 matrices describe the weight-off configuration, which is what now ships.
- **D1 R1 — REJECT.** No cell gains in all three regimes (best maximin −0.0001); the
  reference (long-missing 0.5 / 10 s, low-activity off) stays.
- **D1 R2 — PASS on its own terms, NOT adopted (operator option, Q5).** Candidate
  `d_w_long_missing = 2.0`, `d_missing_run_s = 3.0`: +0.0513 (cover_q50) and +0.0855
  (cover_q90), 7/7 on both; −0.0249 (uniform) and −0.0274 (phase_a2). The sign follows
  P(dead | missing): about −0.025 at 0.44, +0.05 at 0.66, +0.09 at 0.92 — break-even near
  one half. It pays only if more than about half of the ball-less frames of the match are
  dead ball, which the allowed signals do not establish.
- **B1 (aligned labels, spellF1 / macroF1):** smooth50 0.193 / 0.453 (uniform),
  0.270 / 0.601 (cover_q50), 0.355 / 0.769 (cover_q90) against the published 0.243 /
  0.672; ours after U1 0.376 / 0.722, 0.349 / 0.612, 0.352 / 0.462. Under strong coupling
  the decoder is level with smooth50 on spellF1 and far below it on macroF1: with the
  ball absent it has no dead-ball evidence (the P6 D-audit finding, now quantified).
- **Adoption record:** `config.py` (`d_w_low_activity = 0.0`, origin `external_fit`);
  `LA+AI.csv` regenerated, md5 `aeef0058a08860c9e2407423b1172684` →
  **`47b38c1f778ca2643f8f6429d027ee31`**; official validator PASS; 141 tests green; the
  byte-identity test passes on the new file.
- **Own-output check (pinned as a bucket, no veto for U1): "> 20 %" of graded frames
  change label.** Own-output counts, sanity only, never used to select: 324 possessions
  of mean 17.6 s; D share 10.0 % of graded frames (was 26.6 %). The retired feature was
  producing most of our `D` frames. The size of the shift matches the cover rows of
  stage 5 (predicted dead share 0.19 → 0.07 on cover_q90), where the weight-off cell
  still wins 7/7 — but it is a large change three days before the deadline and is
  flagged for the operator.
- **Evaluator:** not run. Config and a code path changed, so the protocol's second
  guarded run is owed (operator only, band + flags). It is a correctness gate; it must
  not be used to choose between the old and the new file (RULES.md: the public window
  never chooses between two versions).
- **K1 — pinned verification FAIL, flag stays OFF.** `tuning/kickoff_check.py`: the
  second-half clause passes (tag within ±250 frames of the FAQ's 2,934); the clause
  "one tag inside each graded half" fails. Diagnosis (categorical): two tags overall, none
  inside the first graded half, one within 250 frames *before* the first graded frame —
  i.e. the first half is graded from the kick-off itself, exactly as the FAQ says, and
  the rule is a no-op there. The pinned clause was mis-specified, not the rule. Changing
  a check after seeing it fail is the operator's decision, not the implementer's:
  proposed amendment — first half "no tag inside, or span ≤ 250"; second half as pinned.
  Own-output effect with the rule on: "< 2 %" of graded frames. Code, tests and the check
  are committed with `kickoff_rule = False`.
- **Historical matrices:** `sweep_results*`, `stage4_*`, `e7_*`, `post_checks*`,
  `no_harm_no_play.csv` were computed on the km/h bed and are kept as the record of what
  ran. Re-running those scripts now uses m/s and will not reproduce them to the last
  digit; `stage5` is the units-correct reference.

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
