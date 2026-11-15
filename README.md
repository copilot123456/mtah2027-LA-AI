# MTAH 2027 — LA+AI possession decoder

Frame-level team-possession labeling (`H`/`A`/`D`/`X`) for FC Barcelona vs Athletic Club,
built for the **More Than A Hack 2027** Initial Challenge (Mobile World Capital / Talent
Arena). Ranked on **spellF1** (possession-segment F1); `macroF1` tiebreak.

**Status: frozen and submitted state.** The artifact is `LA+AI.csv` — regenerated
byte-identically by `python -m src.submit`, md5 `c121715836ed603fc5792b85daf6f279`,
182 tests green, guarded evaluator band **A**. Every decision that produced it is on
record.

---

## 🗺️ Start here — the guided path

This repository documents itself as a course. Follow the numbered path and you end up
able to *change* the decoder, not just run it. (≈ 4 h reading + 30 min hands-on; every
page has verify-checkpoints.)

| # | Page | You will learn | Time |
|---|---|---|---|
| 1 | **[docs/01-the-challenge.md](docs/01-the-challenge.md)** | what the problem is, how spellF1 really scores, and the three traps that decide the design | 15 min |
| 2 | **[docs/02-quickstart.md](docs/02-quickstart.md)** | reproduce the submission from zero — with checkpoints that prove it | 30 min |
| 3 | **[docs/03-how-the-decoder-works.md](docs/03-how-the-decoder-works.md)** | the pipeline stage by stage: emission, Viterbi, the post-rules, the config surface | 30 min |
| 4 | **[docs/04-why-these-results.md](docs/04-why-these-results.md)** | the decision story: 20 experiments, 3 adoptions, 5 honest rejections, one retraction | 30 min |
| 5 | **[docs/05-the-evidence-protocol.md](docs/05-the-evidence-protocol.md)** | the operating system: pre-declaration, the external bed, the +0.02 bar, guarded runs | 25 min |
| 6 | **[docs/06-limitations-and-open-questions.md](docs/06-limitations-and-open-questions.md)** | what we cannot do, what the evidence cannot tell us, and what to do next | 15 min |
| 7 | **[docs/07-glossary.md](docs/07-glossary.md)** | every term, when you meet it | reference |
| 8 | **[docs/08-mastery-checklist.md](docs/08-mastery-checklist.md)** | quiz + hands-on challenges that prove mastery | 30 min |

**Deep references (not part of the course):**
[`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) (the submission document) ·
[`dossier.md`](dossier.md) (compliance + decision record — append-only history) ·
[`AUDIT.md`](AUDIT.md) (independent audit of the frozen state) ·
[`reports/ruiz-work.html`](reports/ruiz-work.html) (the illustrated branch report) ·
[`tuning/results/`](tuning/results) (every matrix and read-out).

**Team standards:** [PIPELINE_STANDARD.html](PIPELINE_STANDARD.html) (human) /
[PIPELINE_STANDARD_AI.md](PIPELINE_STANDARD_AI.md) (AI agents) — the plug-and-play
pipeline framework (`pipeline_spec.py` is the one file to swap/add/reorder stages;
framework changes are shared with the team before merging) ·
[AI-SOP-SKILL.md](AI-SOP-SKILL.md) — the two-report standard every branch produces.

---

## The result, in five lines

- A classical, fully inspectable decoder: per-frame emission → per-half Viterbi →
  post-rules (switch gate, team-switch delay, min-dwell + coalesce, kick-off rule).
- Tuned **only** on 7 external public matches (IDSSE) under a written evidence protocol —
  never on the provided match, its ground truth, its window or the video.
- 20 pre-declared experiments: 3 adopted, 5 rejected, 1 retracted — all on record
  ([docs/04](docs/04-why-these-results.md)).
- Four guarded evaluator runs (operator-only, band + flags): **C → B → A → A**.
- Reproducible: `md5sum LA+AI.csv` → `c121715836ed603fc5792b85daf6f279` from a clean clone.

## Layout

| Path | What it is |
|---|---|
| `docs/` | the guided course (start at [docs/01](docs/01-the-challenge.md)) |
| `config.py` | every tunable parameter + provenance (`prior` \| `fixed_default` \| `external_fit`) — nothing is ever fitted on the provided match (test-enforced) |
| `src/data.py` | loaders + the alignment contract (`video_frame = frame_id - 1`, graded spans, local pickle cache) |
| `src/ball_fill.py` | ball provenance tiers (`atd` / `none`; the kinematic fill tier is implemented but ships disabled) |
| `src/emission.py` | per-frame H/A/D log-emissions: team kernel in metres, confidence tempering, dead-ball + kinematic terms |
| `src/decoder.py` | per-half Viterbi → switch gating → team-switch delay → min-dwell + coalesce → kick-off rule |
| `src/kinematics.py`, `src/kin_model.json` | player-only kinematic dead/team terms for ball-less frames (P9), fitted LOMO on IDSSE |
| `src/x_policy.py` | abstention rule (ships off: `X` only outside the graded spans) |
| `pipeline_spec.py`, `src/pipeline.py`, `src/steps.py` | the pipeline framework — see [PIPELINE_STANDARD.html](PIPELINE_STANDARD.html) |
| `src/submit.py` | CSV writer + the official validator |
| `src/diagnostics.py` | aggregate-only data diagnostics (no ground truth, no video) |
| `tests/` | 182 tests: alignment, provenance, fill/emission/decoder invariants, brute-force Viterbi equivalence, format + validator, byte-identity of the committed CSV, reproduction of the external selection, pipeline-framework contract |
| `tuning/` | the external evidence machinery: IDSSE loader, missingness simulator, stages 4/5/8/9/10 + gates, kinematic fitting + parity checks; matrices and reports committed under `tuning/results/` |
| `evaluation/`, `starter/`, `games.py` | official scorer, validator, the four published baselines and the starter helpers (vendored, MIT © Metrica Sports — never edited) |
| `dossier.md` | compliance dossier: data lineage, parameter provenance, GT-look log, tuning record |
| `TECHNICAL_REPORT.md` | submission document |
| `AUDIT.md` | independent audit of the frozen pipeline: findings, fixes, verification |
| `PIPELINE_STANDARD.html`, `PIPELINE_STANDARD_AI.md` | the pipeline framework standard (human / AI) |
| `AI-SOP-SKILL.md`, `reports/` | parallel-work report standard; per-branch reports live in `reports/` |
| `reference-docs/` | local-only copies of the challenge docs (not tracked) |

## Reproduce

Everything runs from the repository root. Tested with CPython 3.14.4 on Linux; the pins in
`requirements-pipeline.txt` are the versions that produced the committed `LA+AI.csv`.
Step-by-step with checkpoints: [docs/02-quickstart.md](docs/02-quickstart.md).

```bash
# 1. environment. `python3 -m venv` needs ensurepip, which some distributions strip out;
#    virtualenv works everywhere.
python3 -m virtualenv .venv              # or: python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

# 2. match data (gated, never committed): unpack the organiser bundle into data/, or symlink it:
#    ln -s /path/to/tactical_data data/tactical
#    ln -s /path/to/broadcast_data data/broadcast
#    (or export MTAH_DATA_DIR=/path/to/bundle — see data/README.md)

# 3. verify, then regenerate the submission
.venv/bin/python -m pytest -q            # 182 tests, ~30 s with a warm cache
.venv/bin/python -m src.submit           # writes LA+AI.csv and runs the official validator
md5sum LA+AI.csv                         # c121715836ed603fc5792b85daf6f279
```

The first run parses the ATD with kloppy and caches it under `data/.cache/` (about 1m40s and
260 MB; later runs take about 4 s). The cache is a local accelerator only: delete the folder to
rebuild it from the raw files. The output is identical either way (checked from a clean clone
with a cold cache, see `AUDIT.md`).

Optional:

```bash
.venv/bin/python -m src.diagnostics            # aggregate data diagnostics (no GT, no video)
.venv/bin/python evaluation/run_evaluation.py  # official scorer on the shipped baselines only
                                               # (never run on LA+AI.csv during development — dossier §3)
```

### External tuning (IDSSE, CC BY 4.0)

`tuning/results/` ships the committed matrices and reports, and `tests/test_tuning.py` checks
that the pre-declared selection rule applied to them yields exactly the cell in `config.py`.
To regenerate them, point `IDSSE_DIR` at the seven IDSSE matches (default `data/idsse/`,
gitignored; missing files are fetched from the HF mirror `pysport/idsse-data` on first use):

```bash
.venv/bin/python -m tuning.sweep                 # stage 1: 243 cells x 2 scenarios x 7 matches
.venv/bin/python -m tuning.sweep --stage2        # extended grid, `none` scenario
.venv/bin/python -m tuning.sweep --stage3        # final grid; selection on the stage-2+3 union
.venv/bin/python -m tuning.post_checks           # d_bias / X-policy checks at the stage-1 cell
.venv/bin/python -m tuning.post_checks --final   # the same checks at the shipped cell
.venv/bin/python -m tuning.stage5                # units-corrected dead-ball pass + baselines (~15 min)
.venv/bin/python -m tuning.gate5                 # applies the pre-declared rules, writes report_stage5.md
.venv/bin/python -m tuning.kickoff_check         # kick-off tag rule: categorical PASS/FAIL only
.venv/bin/python -m tuning.stage8 && .venv/bin/python -m tuning.gate8   # switch gate + delay (~3 min)
.venv/bin/python -m tuning.kin_fit               # player-only kinematic model -> src/kin_model.json
.venv/bin/python -m tuning.stage9 && .venv/bin/python -m tuning.gate9   # kinematic terms (~25 min)
.venv/bin/python -m tuning.stage10 && .venv/bin/python -m tuning.gate10 # K-weight edge check (REJECT)
.venv/bin/python -m tuning.kin_parity            # provided-match parity checks: PASS/FAIL only
```

The IDSSE speed column is km/h and is converted to m/s on load (dossier §7 P7). Matrices
committed before 2026-11-14 were computed on km/h and are kept as the record of what ran;
`stage5_*` is the units-correct reference.

A sweep stage takes between half an hour and an hour with 7 workers; the IDSSE parse (minutes
per match) is cached under `data/.cache/` as well.

## Rules we operate under (see dossier.md)

- Labels come from an algorithm only; no hand annotation of the provided game.
- Training/tuning on external data only (IDSSE, CC BY 4.0) or a-priori values — never on the
  provided match's ground truth, video, or per-frame statistics.
- The match data is not redistributable and must never be committed (`.gitignore` enforces it).

## Provenance

Starter code and evaluator vendored from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(MIT, © 2026–2027 Metrica Sports) at commit `f8f2629`.
