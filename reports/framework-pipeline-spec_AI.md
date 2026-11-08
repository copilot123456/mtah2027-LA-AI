---
branch: framework/pipeline-spec
slug: framework-pipeline-spec
baseline: 25207bc                   # fork point with main — the common ground this branch started from
                                    # (recorded when the branch was live; it has since been merged by ff,
                                    #  so a post-merge `git merge-base main <branch>` reads 95e6eca)
author: operator + Copilot (LA+AI)
date: 2026-11-08
status: ADOPTED
verdict: ADOPTED
kpi_band: PENDING                   # no guarded evaluator run yet — categorical only, never numbers
sample: true                        # produced under AI-SOP-SKILL.md as its own worked example;
                                    # committed on standard/report-skill together with the SOP
                                    # (the branch it describes is already merged — normal flow
                                    #  commits reports on the branch they describe)
---

# Branch report (AI) — plug-and-play pipeline framework

## 1. Purpose & scope

Replace the hard-wired 5-stage `run()` chain with a **declared list of stages**
(`pipeline_spec.py`) executed by a validating runner, so team branches can swap or add pipeline
stages with a one-line binding instead of editing the audited core. **Behaviour-preserving by
design**: the default declaration must reproduce the frozen submission byte-for-byte.

Scope: mechanism only — no scoring-behaviour change, no `config.py` change, no tuning.

## 2. Anchor

- **Baseline (common ground):** `25207bc` — *Add EXECUTIVE_REPORT.html: plain-language team-facing report*
- **Head:** `95e6eca` — *Add plug-and-play pipeline framework (team standard v1.0)*, merged to
  `main` (fast-forward) and pushed to `copilot123456/mtah2027-LA-AI` on 2026-11-08.
- Branch `framework/pipeline-spec` retained locally and on the remote.
- Clean tree at verification: `git status --porcelain` empty.

## 3. Verification batch (real output, 2026-11-08)

```text
$ .venv/bin/python -m pytest -q
92 passed in 19.02s

$ rm -f /tmp/v.csv && .venv/bin/python -m src.submit --out /tmp/v.csv && md5sum /tmp/v.csv
Valid. Note this checks the format only -- it says nothing about whether the labels are right.
Score it against the public ground truth with run_evaluation.py.
aeef0058a08860c9e2407423b1172684  /tmp/v.csv
aeef0058a08860c9e2407423b1172684  LA+AI.csv     # committed file, byte-identical

$ python -c "import pipeline_spec; from src.pipeline import validate_spec; print(validate_spec(pipeline_spec.SPEC))"
[]                                                   # default spec validates clean
```

Test count before/after: **84 → 92** (+8 in `tests/test_pipeline_spec.py`).
Submission md5 before/after: **`aeef0058a08860c9e2407423b1172684` = `aeef0058a08860c9e2407423b1172684`**.

## 4. Diff inventory

```text
$ git diff --stat 25207bc...95e6eca
 PIPELINE_STANDARD.html      | 271 ++++++++++++++++++
 PIPELINE_STANDARD_AI.md     | 202 +++++++++++++
 README.md                   |  13 ++-
 pipeline_spec.py            |  37 +++++
 src/pipeline.py             | 114 +++++++++++-----
 src/steps.py                |  62 +++++++
 tests/test_pipeline_spec.py | 107 +++++++++++++
 7 files changed, 781 insertions(+), 25 deletions(-)
```

| File | Role in the change |
|---|---|
| `pipeline_spec.py` (new) | the declaration: `SPEC` list of 6 `Step`s + quick-start docstring |
| `src/pipeline.py` (rewritten) | contract (`Ctx`, `Step`, `validate_spec`) + spec-driven `run()`; signature-compatible (`run(cfg)` preserved) |
| `src/steps.py` (new) | 6 default handlers — thin adapters over the **unchanged** pure stage functions |
| `tests/test_pipeline_spec.py` (new) | 8 framework tests: validation (requires/provides, docstrings, duplicates), ordering via real side-effects, swap mechanics, end-state guard |
| `PIPELINE_STANDARD.html` (new) | human standard (exec-report style) incl. the required share-with-the-team adoption step |
| `PIPELINE_STANDARD_AI.md` (new) | AI standard: contract field-by-field, invariants, recipes, §7 change process |
| `README.md` | standard links + layout rows + test count |

**Config deltas:** none — `config.py` does not appear in the diff; `PROVENANCE` untouched;
`tests/test_config.py` green unchanged.

## 5. Evidence

| Claim | Evidence | Artifact |
|---|---|---|
| Output unchanged | fresh run md5 == committed md5 (both directions checked) | `LA+AI.csv`, `tests/test_submission.py::test_committed_submission_is_what_the_code_produces` |
| Spec is valid by contract | `validate_spec(SPEC) == []` | probe above; `tests/test_pipeline_spec.py` |
| Default behaviour ≡ old `run()` | same 6 stages, same order, same arguments — adapters call the identical pure functions; byte-identity test end-to-end | `src/steps.py` vs pre-change `src/pipeline.py` |
| Score effect | **none by construction** — no emission/decoder/config change → external-evidence rule (>0.02 paired, dossier §7) **not triggered**; no IDSSE cell run needed or performed | — |
| P6 KPI | `PENDING` — no guarded evaluator run at report time (band letter + flags arrive from the operator) | — |

## 6. Reproduction from a clean clone

```bash
git clone git@github.com:copilot123456/mtah2027-LA-AI.git && cd more-than-a-hack-2027
python3 -m virtualenv .venv && .venv/bin/pip install -r requirements-dev.txt
# provide data/ per data/README.md (never committed)
.venv/bin/python -m pytest -q                      # expect 92 passed
.venv/bin/python -m src.submit --out /tmp/v.csv && md5sum /tmp/v.csv
# expect aeef0058a08860c9e2407423b1172684
```

## 7. Compliance log

- [x] **No provided-match per-frame values** in either report — aggregates/counts only.
- [x] **No guarded-evaluator numbers** — `kpi_band: PENDING`; no run performed.
- [x] **No GT/video-derived claim** — evidence is tests, md5, validator and git diff only.
- [x] **CFG deltas:** none → provenance rule trivially satisfied.
- [x] **Behaviour change:** none (byte-identical) → evidence rule not applicable; explicitly *not*
      used as a tuning vehicle.
- [x] **Framework touched: yes — this IS the framework change.** Adoption step followed: built on
      a branch, verified, operator reviewed and approved the merge (`merge_and_push`, 2026-11-08),
      merged + pushed; shared with the team via `PIPELINE_STANDARD.*` (README-linked) and this
      report pair.
- [x] **Forbidden paths:** diff contains none of `evaluation/`, `starter/`, `games.py`; no
      `data/*`; `.gitignore` untouched; no history rewrite (fast-forward only).
- [x] **Human ↔ AI consistency:** one-screen claims = this document's claims.
- [x] **Operator reviewed** before merge (explicit approval in session).

## 8. Standard-change detection

This branch changed: `src/pipeline.py` (contract/runner), `pipeline_spec.py` (defaults),
`src/steps.py` (handlers), and added two standard documents. Per `PIPELINE_STANDARD_AI.md` §7
these are exactly the artifacts whose change requires the share-with-the-team step — performed
as above. **No further framework change is pending.**

## 9. Open questions / handoff

- **SOP learning (for `AI-SOP-SKILL.md` v1.1):** `merge-base main <branch>` only yields the true
  baseline *before* the branch merges; for merged branches the baseline must be recorded at fork
  time (or taken as the first parent of the merge). Recorded here; SOP update goes through the
  share-with-the-team step.
- The first real handler swap (e.g. a ViT gap-filler) does not exist yet — Wednesday 2026-11-11
  is the first stress test of the declaration file under parallel edits.
- P6 band + flags still pending from the operator; when they arrive, they slot into the human
  report's evidence table and this file's `kpi_band` field (letter + categorical flags only).
