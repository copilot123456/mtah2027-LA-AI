# AI SOP SKILL — parallel work report standard (MTAH 2027, LA+AI)

**What this is:** the standard for the two reports every team branch produces before sharing its
work — a human **executive-style report** and a fully detailed **AI report**, produced by the
branch's AI agent together with its operator.

**Status:** v1.0 · 2026-11-08 · team-facing repo standard.
**Companions:** [`PIPELINE_STANDARD.html`](./PIPELINE_STANDARD.html) /
[`PIPELINE_STANDARD_AI.md`](./PIPELINE_STANDARD_AI.md) (the pipeline framework this reports
against) · [`EXECUTIVE_REPORT.html`](./EXECUTIVE_REPORT.html) (the style source and the
project-level story this extends).

> **Adoption rule (same as the pipeline standard):** this SOP is itself a common artifact —
> changes to *it* must be shared with the team before merging. A branch report, by contrast, is
> produced *by* this process and shared at the Wednesday sync.

## 0. Why reports exist here

Each member branches from `main` (the common ground), pursues one "break", and brings the result
to the **Wed 2026-11-11 sync**. The reports are the convergence artifact: they are what the team
reads to decide what enters the final branch. Two audiences, one content:

| Version | File | Audience | Bar |
|---|---|---|---|
| Human | `reports/<slug>.html` | the team, in the meeting | one-screen verdict, plain language, honest reading |
| AI | `reports/<slug>_AI.md` | any AI agent (and future us) picking the branch up | every command, diff and artifact path — reproducible without the author |

**Golden rule:** every claim in the human report must be backed by the AI report; neither may
claim anything the verification batch didn't show. Never fabricate a number — missing evidence is
written as `PENDING`, not guessed.

## 1. Inputs (operator-provided — never invented)

| Input | Source | Hard rule |
|---|---|---|
| branch name, author | operator / git | slug = branch name lowercased, `/[^a-z0-9-]+/` → `-` |
| baseline sha | `git merge-base main <branch>` | the common ground the report starts from |
| test / validator / md5 results | the verification batch (you run it) | paste real output only |
| P6 band + flags | **operator only** (guarded run) | **band letter A/B/C/D + the categorical flags of plan §1 — never the numbers** |
| IDSSE evidence | committed `tuning/results/` / dossier §7 | paired delta + CI — external data, allowed |
| verdict (ADOPT / ITERATE / DISCARD) | operator decision after seeing evidence | not yours to pick silently |

## 2. Workflow

1. **Anchor** — `git status --porcelain` (must be clean), record
   `BASELINE=$(git merge-base main <branch>)`.
2. **Verification batch** — run it and capture the real output:

   ```bash
   .venv/bin/python -m pytest -q
   rm -f /tmp/v.csv && .venv/bin/python -m src.submit --out /tmp/v.csv && md5sum /tmp/v.csv
   git diff --stat <BASELINE>...HEAD
   git diff <BASELINE>...HEAD -- config.py
   ```

   If the md5 differs from the baseline's, the change must be *intentional* and *documented*
   (regenerated CSV committed together with the change) — otherwise stop and fix before
   reporting.
3. **Assemble evidence** — P6 band + flags (from the operator), IDSSE results, test counts,
   config provenance deltas.
4. **Draft Template A (human)** → `reports/<slug>.html`.
5. **Draft Template B (AI)** → `reports/<slug>_AI.md`; cross-check every number against steps
   2–3.
6. **Compliance self-check** — §4; every box must pass before the operator sees it.
7. **Operator review → commit both files on the branch** (never on `main` directly).
8. **Share** — post the one-screen summary (or the file) to the team chat / bring it to the Wed
   sync. A report that isn't shared wasn't produced.

## 3. Templates

### Template A — human report (`reports/<slug>.html`)

Style: **copy the `<style>` block and component vocabulary from `EXECUTIVE_REPORT.html`** — it is
the single source of truth for the look (hero, chips, kpis, cards, callouts, badges, nav). Keep
its direct, plain-language register. Sections, in order, each an `<section>` with an emoji `<h2>`
(same pattern as the original):

1. **Hero** — title `Branch report — <slug>`; chips: `baseline main @ <sha>`, `branch <name>`,
   `author`, `date`, status badge (`PROPOSED / ADOPTED / REJECTED`).
2. **🖥️ One-screen summary** — the idea in one sentence; a headline delta table
   (`metric | common ground | this branch`); the **verdict card**: ADOPT / ITERATE / DISCARD with
   a one-line confidence.
3. **💡 The idea — what we tried to break** — hypothesis and why it could move spellF1 (2–4
   sentences).
4. **🧱 What changed from common ground** — table `area | common ground | this branch` (pipeline
   steps, config keys, files) plus the `git diff --stat` summary. The baseline sha appears in
   prose: everything below is measured *from that commit*.
5. **📈 Improvements — the evidence** — before/after tables: P6 KPI (band letter + flags only),
   IDSSE paired delta + 95% CI, test counts. Then **"Is it good? — three honest lenses"**
   (borrow the original's pattern), including what did *not* improve. If nothing improved, say so
   plainly — an honestly rejected idea is a valid Wednesday output.
6. **✅ Rules & compliance** — checklist with live results: no eval/GT tuning; evidence rule
   status (>0.02 paired on IDSSE, pre-declared dossier §7) for any behaviour change; forbidden
   files untouched; byte-identity or intentional regeneration; tests green.
7. **🙋 Verdict & the ask for Wednesday** — what the team must decide, risks, rollout note. If
   the branch touched the pipeline framework: adoption-step reminder (PIPELINE_STANDARD).
8. **🔁 Reproduce it yourself (3 commands)** — the verification batch, from a clean clone of the
   branch.

### Template B — AI report (`reports/<slug>_AI.md`)

Front matter (YAML):

```yaml
---
branch: <branch>
slug: <slug>
baseline: <merge-base sha>        # the common ground this branch started from
author: <human> + <ai agent>
date: 2026-MM-DD
status: PROPOSED | ADOPTED | REJECTED
verdict: ADOPT | ITERATE | DISCARD
kpi_band: A|B|C|D | PENDING       # categorical only — never evaluator numbers
---
```

Sections:

1. **Purpose & scope** — one paragraph: what this branch changes and for whom.
2. **Anchor** — baseline sha, branch, clean-tree proof, relationship to main
   (`git log --oneline <BASELINE>..HEAD`).
3. **Verification batch** — exact commands + pasted outputs (pytest line, md5, validator result).
4. **Diff inventory** — `git diff --stat <BASELINE>...HEAD`; key hunks quoted; config deltas as
   a table `key | baseline value | branch value | provenance origin | evidence`.
5. **Evidence** — IDSSE artifacts (paths, paired delta, CI, dossier §7 pre-declaration ref); P6
   band + flags exactly as delivered by the operator; test counts before/after.
6. **Reproduction from a clean clone** — numbered steps.
7. **Compliance log** — the §4 checklist with results, not just claims.
8. **Standard-change detection** — did the branch touch `pipeline_spec.py`, `src/pipeline.py`,
   `src/steps.py`, this SOP, or `config.py` provenance? If yes: adoption-step reminder and what
   must be shared with the team.
9. **Open questions / handoff** — what a downstream AI must know to continue or review this work.

## 4. Compliance self-check (both reports)

- [ ] No provided-match per-frame values anywhere (coarse aggregates only)
- [ ] No numbers from a guarded evaluator run — band letter + categorical flags only
- [ ] No claim derived from the video or the ground truth
- [ ] Every CFG delta carries its provenance origin + evidence
- [ ] Behaviour changes: >0.02 paired on IDSSE, pre-declared in dossier §7 — or explicitly *not*
      met → ITERATE, never ADOPT
- [ ] Framework touched → adoption step flagged (PIPELINE_STANDARD)
- [ ] Verification batch pasted with real output; md5 matches baseline, or the change is
      intentional and documented
- [ ] Human and AI reports agree, claim for claim
- [ ] Operator reviewed before commit

## 5. Naming & location

- `reports/<slug>.html` + `reports/<slug>_AI.md`, committed **on the branch they describe**
  (e.g. `Ruiz-parrallel-work-branch` carries its own pair: `reports/ruiz-parrallel-work-branch.html`
  and `reports/ruiz-parrallel-work-branch_AI.md`).
- The `reports/` folder accumulates one pair per branch — no name collisions when everything
  converges on Wed 2026-11-11.

## 6. What this SOP does not do

- It does not change the pipeline framework or the evidence rule — reports *document* them.
- It does not authorize running `evaluation/run_evaluation.py` on `LA+AI.csv` (operator-only; the
  band letter comes back).
- It does not replace `EXECUTIVE_REPORT.html` — that stays the project-level story; branch
  reports are its per-experiment extensions, written from the common ground each branch started
  from.
