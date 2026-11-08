# Audit — MTAH 2027 possession decoder (frozen P5 state)

Independent audit + improvement pass, 2026-11-07, on `a6bb75a` (P5 closed). Scope and rules as
briefed: correctness → compliance → reproducibility → tests → performance → docs, with no change
to label outputs, no ground-truth look, no evaluator run on the submission, no video, no data
commits, no edits to the vendored `evaluation/`, `starter/`, `games.py`.

**Headline.** The frozen state is sound: no leakage path, byte-identical regeneration from a
clean clone with a cold cache, every frozen decision re-confirmed at the shipped cell with new
external evidence. Nothing found changes `LA+AI.csv` (md5 `aeef0058a08860c9e2407423b1172684`
before and after every commit below). The real gaps were release-readiness — unpinned
requirements, a README that did not work on this host, a cwd-dependent validator call — plus one
dormant logic bug and one implementation slip in the selection code that, once fixed, picks the
same cell.

Baseline before any change: 64/64 tests in 11.1 s, validator OK, two runs md5-identical.

## 1. Findings

Severity: blocker / high / medium / low / info (info = verified clean, recorded so the next
auditor does not redo it). Line numbers refer to `a6bb75a`.

| id | sev | area | file:line | evidence | status |
|---|---|---|---|---|---|
| A1 | high | correctness / tests | `src/submit.py:43-46` | `validate()` spawned `evaluation/validate_submission.py` as a cwd-relative path; `pytest <repo>/tests` from `/tmp` → `can't open file '/tmp/evaluation/validate_submission.py'`, exit 2, `test_full_pipeline_validates_officially` FAILED (reproduced) | fixed `5c94af9` — path from `__file__`, submission path made absolute; regression test added |
| A2 | medium | correctness (dormant) | `src/x_policy.py:36-37` | `_distance_to_nearest_boundary` returned `+inf` for a switch-free span, skipping the "half start / half end" clause of the rule; with `x_use=True` such a half abstained up to its first and last frame. `test_x_policy_never_abstains_whole_half` passed only through this degenerate path | fixed `7d8d6ab`; shipped `x_use=False`, output unchanged; P5 X-policy ablation re-run after the fix is numerically identical (no IDSSE half decodes to zero switches) |
| A3 | low | correctness | `src/submit.py:31-33` | letter lookup by `code + 1`: an invalid `-2` would silently become `D`, a `5` would raise `IndexError` deep in numpy | fixed `5c94af9` — explicit code check, `ValueError`; test added |
| A4 | low | hygiene | `src/pipeline.py:13` | annotation names `np.ndarray` with numpy never imported (works only because of `from __future__ import annotations`) | fixed `5c94af9` |
| A5 | info | correctness | `src/submit.py`, `src/decoder.py`, `src/data.py`, `games.py` | scorer interface verified: internal `X=-1` → letter → scorer `X=3` round-trips (`test_official_reader_roundtrip`); per-half decode enforces contiguity (now tested); alignment `video_frame = frame_id - 1` and graded spans tested against the organisers' numbers; Viterbi ties resolve to the first argmax (deterministic); log floors at `EPS=1e-12` keep every emission finite (now tested for an invisible team); NaN ball-speed edges never count as stationary (now tested) | verified |
| B1 | info | compliance | repo-wide | `grep` for `_gt`, `possession_gt`, `run_evaluation`, `scoreboard` outside `evaluation/`: hits only in `tuning/*` on IDSSE-derived labels (`pm.label_codes(case.letters)`); no test reads `evaluation/gt/`; local `evaluation/scoreboard.csv` lists the four baselines only; `evaluation/results/` holds only the shipped baselines | verified |
| B2 | medium | compliance / portability | `tuning/idsse.py:33-34` | default `IDSSE_DIR` was a hard-coded path in the operator's home directory, in a public repo | fixed `d96b830` — default `data/idsse/` (gitignored), env override kept; a local symlink `data/idsse → <operator store>` preserves the operator's workflow |
| B3 | medium | compliance / dossier accuracy | `tuning/post_checks.py:1-14,40-41`, `dossier.md:75-80` | the d_bias and X-policy adoption checks ran at the **stage-1** chosen cell (`4/8/3.0/2.0/1.0`) with the fill tier **on**, before stages 2-3 and the fill ablation; the dossier said "on the CHOSEN grid cell" without saying which. Re-run at the shipped cell (`--final`, external IDSSE only): baseline 0.2759 (= the sweep's cell mean), d_bias fit −0.0847 (7/7 negative); X-policy: see B8 — the −0.0338 first recorded here was a scoring artifact. Default mode re-run reproduced the committed `post_checks.csv` with max diff 0.00e+00 | fixed `d96b830` (`--final` mode, default pins the fill tier on so it stays reproducible) + `d60b66d` (dossier §7, report table, `post_checks_final.csv`); numbers restated in the addendum (§6) |
| B8 | medium | correctness of the external evidence | `tuning/sweep.py:90-94`, `tuning/post_checks.py:112` | **found after the audit by the P6 planning pass, verified here:** the harness scored predicted `X` with `spell_scores` on the raw codes, while the official `evaluate()` forward-fills `X` inside graded runs first (`fill_abstentions`). The P5 sweep is unaffected (the decoder never writes `X` on graded frames), but the X-policy ablation counted `H X X H` as two possessions. Re-scored with `evaluate()` on the 7 matches at the shipped cell: X on = 0.2759 spellF1 on every match (delta 0.0000 ×7), macroF1 0.7167 → 0.6069 | fixed (addendum commit): `_score` applies the fill; `post_checks.py` shares it; both matrices regenerated; equality test added |
| B4 | medium | compliance / selection rule | `tuning/sweep.py:149-167` | `_neighbour_means` drew the "other values" of each parameter from the stage-1 `GRID` literal, so on the stage-2/3 extension it ignored the new notches (σ 1.0/1.5/2.0, dwell 3.0, support 1.5, switch 2.0) and probed stage-1 notches never swept at the new values. Re-running the pre-declared rule on the committed stage-2+3 union with the values actually swept: same cell, mean 0.2759, CI [0.2578, 0.2932], LOMO −0.0026, worst neighbour 0.2081 — unchanged | fixed `d96b830`; `test_final_selection_reproduces_frozen_config` + `test_neighbour_means_use_the_swept_values` added |
| B5 | low | compliance / hygiene | `.gitignore` | `tuning/results/smoke.csv` (output of `--smoke`) was neither tracked nor ignored; the real matrices are tracked as intended | fixed `33b2f2e` |
| B6 | info | compliance | `config.py`, `tests/test_config.py` | provenance table complete, no `provided_*` origins; the checker itself was untested — a negative test now proves it reports each failure mode | verified + test added `694d95f` |
| B7 | info | compliance | `dossier.md` vs `git log` | commit history matches the dossier's narrative (P0-P5, vendoring at `f8f2629`); "930 unique cells × 7 matches" verified from the CSVs (stage 2: 768 cells, stage 3: 324, union 930; overlaps reproduce with max abs diff 0.0 for both 1,701 and 1,134 rows); no secrets (`grep` for key/token/secret/password: none) | verified |
| C1 | high | reproducibility | `requirements.txt:10-11`, `requirements-starter.txt` | runtime unpinned (`numpy`, `pandas`) while byte-identical regeneration is a hard rule; `kloppy`, `lxml`, `pytest`, `tabulate` absent from the pipeline's requirement story (they live in the starter's optional file or nowhere) | fixed `33b2f2e` — `requirements-pipeline.txt` (numpy 2.5.4, pandas 3.1.1, kloppy 3.18.0, lxml 6.1.3) + `requirements-dev.txt` (pytest 9.2.0, tabulate 0.10.0); the organisers' two files left as shipped |
| C2 | high | reproducibility / docs | `README.md:14-15,21-29` | `python3 -m venv .venv` fails on this host (`No module named 'ensurepip'`; `virtualenv 21.15.6` works — the existing `.venv` was made with it); layout table still P0-era ("decoder — in progress", "alignment + provenance tests"); no data wiring, no cache note, no `src.submit` step, no md5 | fixed `d60b66d` |
| C3 | info | reproducibility | clean clone | fresh `git clone` + `virtualenv` + pinned install (21 s) + data symlinks, cold cache: first `python -m src.submit` 1m41s (parses the ATD, writes 260 MB under `data/.cache/`), second run 4 s, both md5 `aeef0058…` = committed file; 64/64 at `a6bb75a`, 83/83 at `d60b66d` | verified |
| C4 | low | reproducibility / docs | `src/data.py:9`, `tuning/idsse.py:5-7` | pickle caches are pandas-version-bound and were undocumented for organisers; with pinned pandas they are safe, and rebuilding is automatic | documented `d60b66d` (README) |
| C5 | info | reproducibility | `src/data.py:22-27`, `tests/conftest.py` | `sys.path` insertion of the repo root is what lets `from games import …` work; `python -m src.submit` must run from the repo root (the `src` package is resolved from cwd) — now stated in the README; `validate()` no longer depends on it (A1) | documented |
| D1 | medium | tests | `tests/` | untested: X-policy distances and its inclusive `>=`/`<=` edges, `derive_letters` with NaN owners / owner-free periods, validator from another cwd, invalid label codes, the provenance checker's negatives, gating window half-openness, all-short dwell collapse, non-contiguous half rejection, log-floor numerics, the link between the committed matrices and `config.py`, the link between the committed CSV and the code | fixed `694d95f` (64 → 83 tests; the byte-identity test is on our own output, no GT snapshot) |
| E1 | info | performance | `src/decoder.py:47-51` | full pipeline 4 s warm, suite ~20 s; the Viterbi loop is the hot path but there is nothing to win that would justify touching the decoder of a frozen submission | not pursued (by design) |
| F1 | low | docs | `TECHNICAL_REPORT.md:102-103` | "selection sits on plateau edges for σ and switch penalty" was stale: σ has an interior optimum after stage 3 (1.0: 0.2627, 1.5: 0.2644, 2.0: 0.2417); the switch penalty is the one at the low edge of the swept range (2 vs 4 within noise); dead penalty flat | fixed `d60b66d` |
| F2 | low | docs | `TECHNICAL_REPORT.md:105-116` | reproduction block installed the starter's optional file and skipped the venv/data/md5 story | fixed `d60b66d` |
| F3 | low | docs | `tuning/sweep.py:12-15`, `src/__init__.py:1` | sweep docstring said selection used the `kinematic` scenario (true for stage 1 only); package docstring named a branch that does not exist | fixed `d96b830`, `5c94af9` |
| F4 | low | docs | `dossier.md:86` | test count and reproduction evidence now stale by construction; §3 did not state that the audit made no GT look | fixed `d60b66d` (§3 note, §7 update, new §8) |

Candidate issues from the brief, each checked: unpinned requirements (C1, real); README venv (C2, real); `tuning/results/smoke.csv` (B5, not present locally, not ignored — now ignored); `grid_used` (real, removed in `d96b830`) and `data.load_team_names` (real, no caller, removed in `5c94af9`); `submit.py` cwd dependence (A1, real, reproduced); pickle caches (C4, documented, safe); "930 unique cells × 7 matches" (B7, correct).

## 2. Fixes made (commits on `main`, oldest first; tests green after each)

| commit | rationale |
|---|---|
| `5c94af9` Anchor the validator call to the repo and harden the submission writer | A1, A3, A4, `load_team_names` removal, package docstring |
| `d96b830` Tidy the tuning package: data-driven neighbours, no operator paths | B4 (same outcome, now provable), B2, `grid_used`, `load_union()`, `post_checks --final` + reproducible default (B3), docstrings |
| `7d8d6ab` Make the X-policy edge rule hold on a switch-free half | A2 (dormant path; output and ablation unchanged) |
| `694d95f` Add targeted tests for untested paths (64 -> 83) | D1 |
| `33b2f2e` Pin the pipeline runtime; ignore the tuning smoke matrix | C1, B5 |
| `d60b66d` Docs: make the README true for a clean clone; record the audit evidence | C2, C4, F1, F2, F4, B3 evidence (`post_checks_final.csv`) |
| (this commit) Add AUDIT.md | this report |

Every commit was followed by `python -m src.submit`; the md5 never moved from
`aeef0058a08860c9e2407423b1172684`.

## 3. Deferred suggestions (ranked)

1. **The "adjacent-neighbour" reading of the plateau rule** would select
   `switch_pen=2 / dead_pen=8 / sigma_m=1.5 / min_dwell_s=1.0 / switch_support_s=1.0`
   (mean 0.2854, the best cell; worst adjacent neighbour 0.2780) instead of the frozen cell
   (0.2759). Not adopted: the rule was pre-declared and implemented as "any other swept value",
   and changing the reading after seeing the matrix is post-hoc selection. It is evidence that
   the dwell-1.0 / dead-8 neighbourhood deserves a *new* external experiment if P6 has the
   budget — operator decision, with a dossier entry either way.
2. **`switch_pen=2` is the low edge of the swept range.** A stage-4 notch at 1.0-1.5 is cheap
   (one sigma, `none` scenario, ~15-20 min) and would close the only open marginal.
3. **Missingness realism.** The uniform placement of mask runs is the largest documented gap
   between the test bed and the provided match; a phase-correlated simulator is the P6 item
   most likely to change which cell wins. (Planning handed to a separate session.)
4. **`tuning/post_checks.py` default mode overwrites the committed `post_checks.csv`** when run
   without `--out`; it reproduces it exactly today, but a safer default would write next to it.
5. **One letter table.** `src/submit.py` and `tuning/sweep.py` each carry a `["X","H","A","D"]`
   lookup; move it to `src/decoder.py` next to `CODE_NAMES`.
6. **Cache robustness.** A pandas upgrade would make `pd.read_pickle` fail with an opaque error;
   a `try/except` that says "delete `data/.cache/`" would cost five lines. Moot while pinned.
7. **`tests/test_submission.py::test_official_reader_roundtrip`** mutates `sys.path` for the
   whole session; use `monkeypatch.syspath_prepend`.
8. **Title of `TECHNICAL_REPORT.md` still says "(draft)".** Wording, not audit; the operator
   should finalise before the freeze.
9. **The organisers' `requirements.txt`** stays unpinned by design (vendored); the README now
   points readers to the pinned files, which is enough.

## 4. Final verification block (actual outputs, HEAD `d60b66d`, before this commit)

Working repo:

```
$ .venv/bin/python -m pytest -q
83 passed in 22.58s

$ .venv/bin/python -m src.submit
wrote LA+AI.csv
  segments 467 (possessions 337)  median 7.0s  sub-second 0.0%  D share 26.6%  gated switches 82
OK    LA+AI.csv
      199319 rows, frame 0..199318
      label counts
        H    69460   34.8%
        A    47189   23.7%
        D    42208   21.2%
        X    40462   20.3%
      reports 337 possessions of mean 13.8 s
Valid. Note this checks the format only -- ...
exit=0
aeef0058a08860c9e2407423b1172684  LA+AI.csv

$ .venv/bin/python -m src.submit --no-validate
aeef0058a08860c9e2407423b1172684  LA+AI.csv

$ git log --oneline
d60b66d Docs: make the README true for a clean clone; record the audit evidence
33b2f2e Pin the pipeline runtime; ignore the tuning smoke matrix
694d95f Add targeted tests for untested paths (64 -> 83)
7d8d6ab Make the X-policy edge rule hold on a switch-free half
d96b830 Tidy the tuning package: data-driven neighbours, no operator paths
5c94af9 Anchor the validator call to the repo and harden the submission writer
a6bb75a P5 closed: frozen config (930-cell external selection) + dossier + report

$ git status --short --branch
## main...origin/main [ahead 6]          (clean; push pending the operator)
```

Clean clone (`git clone` of this repo, `python3 -m virtualenv`, `pip install -r
requirements-dev.txt`, data symlinked, cold cache at `a6bb75a`, then pulled to `d60b66d`):

```
kloppy==3.18.0 lxml==6.1.3 numpy==2.5.4 pandas==3.1.1 pytest==9.2.0 tabulate==0.10.0
first run (cold cache):   real 1m41.028s   aeef0058a08860c9e2407423b1172684  LA+AI.csv
second run:               real 0m3.985s    aeef0058a08860c9e2407423b1172684  LA+AI.csv
committed file:                            aeef0058a08860c9e2407423b1172684
83 passed in 22.63s
git status: clean
```

External re-checks (IDSSE only): `post_checks.py` default mode vs committed `post_checks.csv`:
28 rows, max |Δ spellF1| 0.00e+00; `--final` run before vs after the X-policy fix: 28 rows,
max |Δ| 0.00e+00.

## 6. Addendum (2026-11-07, after the audit was pushed)

The P6 planning session read this audit and the harness and found B8: the external X-policy
numbers quoted in dossier §7, the technical report, the vault note and finding B3 above were
artifacts of scoring abstentions without the official forward-fill. Verified by re-scoring the
shipped cell on IDSSE with the vendored `evaluate()`:

| variant | harness spellF1 (old) | official spellF1 | official macroF1 |
|---|---|---|---|
| X off (shipped) | 0.2759 | 0.2759 | 0.7167 |
| X on | 0.2421 | 0.2759 | 0.6069 |

The decision (X off) stands — for the reason `src/x_policy.py` already gives: under the fill a
stable-interior `X` reproduces the decoded label, so spellF1 cannot move and the macroF1 tiebreak
can only lose. What changed in the addendum commit: `tuning/sweep.py::_score` now fills before
`spell_scores` (exactly as `evaluate()`), `tuning/post_checks.py` uses the same scorer and also
prints the macroF1 delta, both post-check matrices were regenerated (d_bias rows identical to the
last digit; X rows: spellF1 tie at both cells, macroF1 −0.079 stage-1 / −0.110 shipped), the
`x_use` provenance note, dossier §7, the report's layer table and this file were restated, and
`test_harness_scorer_matches_the_official_evaluate` pins the equality. Label outputs unchanged
(md5 `aeef0058a08860c9e2407423b1172684`).

## 5. Compliance statement

During this audit I did **not**: read `evaluation/gt/*` (neither the public window nor anything
else; the only GT-related reads were `grep`s of the code and the `git ls-files` inventory), run
`evaluation/run_evaluation.py` on anything (not even the baselines), open or inspect any video,
commit or stage anything under `data/` (the three symlinks there are gitignored), weaken
`.gitignore` (one ignore added), rewrite or force-push history, edit `evaluation/`, `starter/`,
`games.py`, or `config.py` values, or put any provided-match per-frame values, coordinates or
labels into any AI/network channel (every number quoted here is a coarse aggregate already in
the repository's documents or a statistic of our own output). Label outputs are unchanged; every
frozen decision stands, now with evidence at the shipped cell.
