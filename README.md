# MTAH 2027 — LA+AI possession decoder

Frame-level team-possession labeling (`H`/`A`/`D`/`X`) for the FC Barcelona vs Athletic Club
match, built for the **More Than A Hack 2027** Initial Challenge (Mobile World Capital /
Talent Arena). Ranked on **spellF1** (possession-segment F1); `macroF1` tiebreak.

## Layout

| Path | What it is |
|---|---|
| `config.py` | every tunable parameter + provenance (`prior` \| `fixed_default` \| `external_fit`) — nothing is ever fitted on the provided match |
| `src/data.py` | loaders + the alignment contract (`video_frame = frame_id - 1`, graded spans, caching) |
| `src/diagnostics.py` | aggregate-only data diagnostics (no ground truth, no video) |
| `src/…` | pipeline stages (ball fill, emissions, decoder — in progress) |
| `tests/` | alignment + provenance tests |
| `evaluation/` | official scorer and the four published baselines (vendored, MIT © Metrica Sports) |
| `dossier.md` | compliance dossier: data lineage, parameter provenance, GT-look log |
| `TECHNICAL_REPORT.md` | submission document |
| `reference-docs/` | local-only copies of the challenge docs (not tracked) |

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-starter.txt pytest
# match data: unpack the organizer bundle to data/tactical and data/broadcast (gitignored)
.venv/bin/python -m pytest tests/ -q          # alignment + provenance
.venv/bin/python -m src.diagnostics           # aggregate data diagnostics
.venv/bin/python evaluation/run_evaluation.py # official scorer on evaluation/results/*.csv
```

## Rules we operate under (see dossier.md)

- Labels come from an algorithm only; no hand annotation of the provided game.
- Training/tuning on external data only (IDSSE, CC BY 4.0) or a-priori values — never on the
  provided match's ground truth, video, or per-frame statistics.
- The match data is not redistributable and must never be committed (`.gitignore` enforces it).

## Provenance

Starter code and evaluator vendored from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(MIT, © 2026–2027 Metrica Sports) at commit `f8f2629`.
