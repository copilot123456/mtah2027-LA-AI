# 03 · How the decoder works — the pipeline tour

One page from raw data to the submission CSV, stage by stage. At the end you should be
able to open any file in `src/` and know why it exists. (Code references are
`file:line` — verified against `ruiz-work`.)

## The one-sentence version

> Per-frame player/ball evidence becomes per-frame log-scores (**emission**); a per-half
> **Viterbi** finds the best label *sequence*; a chain of **post-rules** (switch gate,
> team-switch delay, min-dwell + coalesce, kick-off rule) turns that sequence into
> `H`/`A`/`D`/`X` per frame; `src/submit.py` writes and validates the CSV.

## Stage 0 — data in ([src/data.py](../src/data.py))

`load_arrays()` ([src/data.py:174](../src/data.py#L174)) builds one `MatchArrays` bundle:
ball positions, player positions and speeds, per-slot team, half labels, graded-mask
spans, Smart Tagging starts (set pieces at [src/data.py:149](../src/data.py#L149),
kick-offs at [src/data.py:159](../src/data.py#L159)). The alignment contract
(`video_frame = frame_id − 1`) lives here — everything downstream works in frame index
space. A local pickle cache under `data/.cache/` makes re-runs 4 s instead of 2 min.

## Stage 1 — ball provenance ([src/ball_fill.py](../src/ball_fill.py))

`build_ball_track()` ([src/ball_fill.py:63](../src/ball_fill.py#L63)) produces a `BallTrack`
where every frame knows *how* its ball position is known: observed (`atd`) or absent.
Provenance matters later: the switch gate trusts observed frames, tolerates none, and the
kinematic fill tier (implemented, ships **off**) is the third tier that experiments used.

## Stage 2 — the emission ([src/emission.py](../src/emission.py))

`build_emission()` ([src/emission.py:173](../src/emission.py#L173)) computes, per frame,
log-scores for each label. The shape of it:

```text
e_H = conf · log p_team_H + log(1 − p_d)
e_A = conf · log p_team_A + log(1 − p_d)
e_D = log p_d
```

where `p_team` is the (tempered) team kernel and `p_d` a dead-ball sigmoid over weighted
features. The ingredients:

| Term | What it measures | Where |
|---|---|---|
| **team kernel** | each player's distance to the ball, Gaussian (σ 1.5 m) — "who is on the ball" | `src/emission.py` |
| **confidence tempering** | player-count and tracking-quality weighting | `team_confidence` [src/emission.py:68](../src/emission.py#L68) |
| **dead-ball terms** | inactivity, ball at rest, long missing runs, set-piece tags (last two weight 0 today) | `src/emission.py` |
| **kinematic terms (P9)** | player-only dead log-likelihood (K) and team log-odds (T) on **ball-less** frames, from externally fitted models ([src/kinematics.py](../src/kinematics.py), `src/kin_model.json`) | `src/emission.py` |

All weights live in [`config.py`](../config.py) with provenance — the emission is exactly
as configurable as the evidence allowed and not more.

## Stage 3 — the sequence decoder ([src/decoder.py](../src/decoder.py))

- `viterbi_paths()` ([src/decoder.py:378](../src/decoder.py#L378)) — per half, a standard
  Viterbi over states {H, A, D} with a transition log ([src/decoder.py:56](../src/decoder.py#L56))
  that prices switches and dead-ball entries. `X` is not a state: it is placed later,
  only outside graded spans.
- `finish_paths()` ([src/decoder.py:392](../src/decoder.py#L392)) — runs the post-rules
  below in order and returns the final labels.

## Stage 4 — the post-rules (in the exact order `finish_paths` applies them)

| # | Rule | Function | What it does | Ships |
|---|---|---|---|---|
| 1 | **Switch gate** | `apply_gating` [src/decoder.py:117](../src/decoder.py#L117) | an H↔A boundary needs ball support (or, since P9, kinematic support); unsupported ones **move** to the nearest supporting frame (P8 G; `revert` before) | ON |
| 2 | **Gap placement** | `place_gap_switches` [src/decoder.py:163](../src/decoder.py#L163) | place boundaries in the middle of ball gaps | OFF (rejected P8) |
| 3 | **Min-dwell + coalesce** | `enforce_min_dwell` [src/decoder.py:197](../src/decoder.py#L197) | no spell shorter than `min_dwell_s`; then join equal neighbours (P9 C) | ON |
| 4 | **D min-dwell** | `enforce_d_min_dwell` [src/decoder.py:240](../src/decoder.py#L240) | separate dwell for `D` (only when > `min_dwell_s`) | OFF (rejected P8 M) |
| 5 | **No-play rule** | `apply_no_play` [src/decoder.py:318](../src/decoder.py#L318) | sustained inactivity → `D` | OFF (E10 retraction) |
| 6 | **Team-switch delay** | `delay_team_switches` [src/decoder.py:275](../src/decoder.py#L275) | ball-timed H↔A boundaries land 0.64 s later (P8 S; capped at half the following segment) | ON |

After `finish_paths`, the decode step applies the **kick-off rule**
(`apply_kickoff_rule` [src/decoder.py:346](../src/decoder.py#L346), wired at
[src/steps.py:54](../src/steps.py#L54)): a graded half is `D` from its start up to its
`Kick Off (Start)` tag — the FAQ's graded-but-pre-kickoff trap (P7 K1).

The switch gate deserves a second look: it is where the project's hardest question lives —
what to do with a possession change the ball never showed. Before P8 it deleted such
boundaries (under-segmentation); since P8 it *moves* them to the first supporting frame;
since P9 a ball-less frame carrying kinematic support counts as support
(`gate_kin_support`).

## Stage 5 — output ([src/submit.py](../src/submit.py))

`python -m src.submit` → `main()` ([src/submit.py:66](../src/submit.py#L66)) → the pipeline
runner (below) → `write_submission()` ([src/submit.py:45](../src/submit.py#L45)) →
`validate()` ([src/submit.py:50](../src/submit.py#L50), the official validator as a
subprocess). One row per frame, `frame_id,label`. The committed `LA+AI.csv` is
byte-reproducible: `tests/test_submission.py` pins its md5.

## The pipeline framework (team standard)

Stages 0–5 are not call order in code — they are a declaration:

- [`pipeline_spec.py`](../pipeline_spec.py) — the ordered `SPEC` of six steps
  (`load · ball_fill · emission · decode · x_policy · summary`) with `requires`/`provides`
  tags. This is the one file to swap / add / reorder stages.
- [`src/pipeline.py`](../src/pipeline.py) — the contract and runner: `validate_spec`
  fails fast on a broken declaration before anything executes.
- [`src/steps.py`](../src/steps.py) — thin default handlers, each wiring the context to
  one independently-tested pure function.

See [PIPELINE_STANDARD.html](../PIPELINE_STANDARD.html). Framework changes are
team-shared artifacts; pipeline changes (config, emission, decoder rules) are branch work.

## The config surface ([config.py](../config.py))

Groups worth knowing:

| Group | Keys (shipped) | Meaning |
|---|---|---|
| decoder cell | `switch_pen` 2 · `dead_pen` 5 · `sigma_m` 1.5 · `min_dwell_s` 2.0 · `switch_support_s` 1.5 | the "frozen cell" selected at P6 and defended since |
| gate & delays | `gate_mode` "move" · `switch_delay_s` 0.64 · `gap_switch_frac` 0 | P8 |
| coalesce | `dwell_coalesce` True | P9 C |
| kinematics | `kin_w_dead` 0.3 · `kin_w_team` 0.03 · `gate_kin_support` True | P9 K/T |
| dead features | `d_w_low_activity` 0 · `d_w_set_piece` 0 · `no_play_rule` False | rejected/off, each with its story in config provenance |
| fill | `fill_long_gap_s` 0 | the kinematic fill tier, off |

Every key carries `origin` (`prior` / `fixed_default` / `external_fit`) and `evidence`
in the `PROVENANCE` dict — and a test enforces that nothing is fitted on the provided
match.

> **Gotcha:** a few module docstrings in `src/emission.py` / `src/decoder.py` still say
> "OFF by default" for components that the P8/P9 adoptions switched **on** — the docstrings
> predate the adoptions. `config.py` is the truth; the docstrings are history.

## The 30-minute code tour (do it in this order)

1. [`pipeline_spec.py`](../pipeline_spec.py) (37 lines) — the whole flow at a glance.
2. [`src/pipeline.py`](../src/pipeline.py) — `Ctx` / `Step` / `validate_spec` / `run`.
3. [`src/steps.py`](../src/steps.py) — how handlers wire the context to pure functions.
4. [`config.py`](../config.py) `CFG` block, then 3–4 `PROVENANCE` entries — the discipline
   in miniature.
5. [`src/data.py`](../src/data.py) — the alignment contract docstring first.
6. [`src/ball_fill.py`](../src/ball_fill.py) — provenance tiers.
7. [`src/emission.py`](../src/emission.py) — the docstring formulas, then `build_emission`.
8. [`src/decoder.py`](../src/decoder.py) — module docstring (order of operations),
   `viterbi`, `apply_gating`, `finish_paths`.
9. [`src/submit.py`](../src/submit.py) — one minute.
10. `tests/test_submission.py` — the byte-identity test that freezes everything.

## The tests ([tests/](../tests), 182)

What they protect: the alignment contract, emission/decoder invariants, brute-force
Viterbi equivalence, submission format + **byte-identity of the committed CSV**, the
external selection reproduction (`tests/test_tuning.py` re-applies the pinned rule to the
committed matrices and must yield exactly the shipped cell), and the pipeline-framework
contract.

## Now you can

- read the results story: **[04 · why these results](04-why-these-results.md)**
- or check yourself: **[08 · mastery checklist](08-mastery-checklist.md)**
