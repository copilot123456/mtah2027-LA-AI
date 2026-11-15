"""Gate 5 — apply the dossier §7 P7 rules (U1, D1 R1/R2, B1) to the stage-5 matrices.

    python -m tuning.gate5        # prints the read, writes tuning/results/report_stage5.md

Read-only over `stage5_*.csv`. It never writes config: verdicts are recorded in the
dossier by hand, and only R1 can be adopted without a further operator decision.
"""

from __future__ import annotations  # postponed annotation evaluation

import sys  # sys.path bootstrapping below
from pathlib import Path  # filesystem paths

import numpy as np  # not used in this file
import pandas as pd  # committed matrix tables

ROOT = Path(__file__).resolve().parents[1]  # repo root (above tuning/)
if str(ROOT) not in sys.path:  # make the repo importable
    sys.path.insert(0, str(ROOT))  # prepend repo root

# stage-5 constants (frozen decoder, params, kinds) reused verbatim
from tuning.stage5 import AS_SHIPPED, D_PARAMS, FROZEN_DECODER, KINDS  # noqa: E402

RES = ROOT / "tuning" / "results"  # committed matrices live here (read-only)
OFF = {"d_w_long_missing": 0.5, "d_missing_run_s": 10.0,  # U1 candidate cell:
       "d_w_low_activity": 0.0, "d_activity_thr_ms": 1.0}  # low-activity weight off
PRIMARY_U1 = ("phase_a2", "aligned")  # primary table for U1
R1_REGIMES = ("uniform", "cover_q50", "cover_q90")  # regimes R1 must win in
R2_REGIMES = ("cover_q50", "cover_q90")  # regimes for the coupled R2 reading
BAR, MIN_WINS, HARM, TIE = 0.02, 5, -0.02, 0.005  # pinned adoption thresholds
PUBLISHED = {"baseline_raw": (0.0486, 0.6532), "baseline_smooth25": (0.1996, 0.6752),  # organisers'
             "baseline_smooth50": (0.2433, 0.6724), "baseline_smooth125": (0.2242, 0.6356)}  # figures (B1)


def key(cell: dict) -> tuple:
    return tuple(float(cell[p]) for p in D_PARAMS)  # hashable cell identity


def load_tables(res: Path = RES) -> dict[str, pd.DataFrame]:
    tables = {}  # collected tables
    for kind in KINDS:
        df = pd.read_csv(res / f"stage5_{kind}.csv")  # one committed matrix
        n_cells = df.groupby(D_PARAMS).ngroups  # distinct cells present
        if len(df) != n_cells * 7:  # every cell needs its 7 matches
            raise ValueError(f"{kind}: {len(df)} rows for {n_cells} cells x 7 matches")  # matrix truncated
        tables[kind] = df.set_index(D_PARAMS + ["match_id"]).sort_index()  # index for xs()
    return tables  # all four loaded


def cell_scores(df: pd.DataFrame, cell: dict, metric: str) -> pd.Series:
    """7 per-match values of `metric` for one cell, indexed by match_id."""
    return df.xs(key(cell), level=D_PARAMS)[metric]  # slice that cell's column


def paired(df: pd.DataFrame, a: dict, b: dict, label: str) -> tuple[float, int]:
    d = (cell_scores(df, a, f"spell_f1_{label}") - cell_scores(df, b, f"spell_f1_{label}"))  # per-match delta
    return float(d.mean()), int((d > 0).sum())  # mean delta, matches positive


def all_deltas(tables: dict, a: dict, b: dict) -> dict[tuple[str, str], tuple[float, int]]:
    return {(kind, lab): paired(tables[kind], a, b, lab)  # same paired delta
            for kind in KINDS for lab in ("orig", "aligned")}  # over all eight tables


def u1_verdict(tables: dict) -> dict:
    deltas = all_deltas(tables, OFF, AS_SHIPPED)  # U1 candidate vs shipped
    d, wins = deltas[PRIMARY_U1]  # primary-table values
    others_ok = all(v[0] >= HARM for k, v in deltas.items() if k != PRIMARY_U1)  # no-harm
    return {"deltas": deltas, "primary": d, "wins": wins, "no_harm": others_ok,
            "adopt": bool(d > BAR and wins >= MIN_WINS and others_ok)}  # all three hold


def n_changed(cell: dict, ref: dict) -> tuple[int, float]:
    """(effective parameters changed, summed weight change) — the pinned tie-break."""
    changed = 0  # running change count
    w_lm, r_lm = cell["d_w_long_missing"], ref["d_w_long_missing"]  # long-missing weights
    w_la, r_la = cell["d_w_low_activity"], ref["d_w_low_activity"]  # low-activity weights
    changed += w_lm != r_lm  # toggling a weight counts as one change
    changed += w_la != r_la  # low-activity toggle counts
    if w_lm and r_lm:  # threshold only counts when the weight is on in both
        changed += cell["d_missing_run_s"] != ref["d_missing_run_s"]  # run-length differs
    if w_la and r_la:  # both low-activity weights on
        changed += cell["d_activity_thr_ms"] != ref["d_activity_thr_ms"]  # threshold differs
    return int(changed), float(abs(w_lm - r_lm) + abs(w_la - r_la))  # changes, weight sum


def cells_of(tables: dict) -> list[dict]:
    idx = tables[KINDS[0]].index.droplevel("match_id").unique()  # cells, drop match level
    return [dict(zip(D_PARAMS, map(float, k))) for k in idx]  # each cell as a plain dict


def d1_reading(tables: dict, ref: dict, regimes: tuple[str, ...], veto_tables: list) -> dict:
    """Maximin candidate over `regimes` (aligned labels) + the pinned adoption conditions."""
    rows = []  # candidate rows
    for cell in cells_of(tables):
        if key(cell) == key(ref):  # never compare the reference to itself
            continue  # skip the reference
        deltas = all_deltas(tables, cell, ref)  # all tables vs reference
        g = min(deltas[(r, "aligned")][0] for r in regimes)  # worst-regime gain
        rows.append({"cell": cell, "g": g, "deltas": deltas,
                     "tie": n_changed(cell, ref)})  # tie-break info
    best = max(r["g"] for r in rows)  # best maximin gain on the grid
    tied = [r for r in rows if r["g"] >= best - TIE]  # plateau of the best
    cand = min(tied, key=lambda r: (r["tie"][0], r["tie"][1], -r["g"]))  # fewest changes first
    wins_ok = all(cand["deltas"][(r, "aligned")][1] >= MIN_WINS for r in regimes)  # ≥ 5/7
    harm_ok = all(cand["deltas"][t][0] >= HARM for t in veto_tables)  # no-harm on vetoes
    return {"candidate": cand, "best_g": best, "n_tied": len(tied),
            "wins_ok": wins_ok, "harm_ok": harm_ok,
            "adopt": bool(cand["g"] > BAR and wins_ok and harm_ok),  # all three hold
            "ranked": sorted(rows, key=lambda r: -r["g"])}


def r1(tables: dict, ref: dict) -> dict:
    veto = [("phase_a2", "aligned")] + [(k, "orig") for k in KINDS]  # primary + all orig
    return d1_reading(tables, ref, R1_REGIMES, veto)  # run R1 reading


def r2(tables: dict, ref: dict) -> dict:
    return d1_reading(tables, ref, R2_REGIMES, [(k, "orig") for k in R2_REGIMES])  # orig vetoes


def cell_str(cell: dict) -> str:
    lm = (f"long-missing {cell['d_w_long_missing']:g} / {cell['d_missing_run_s']:g} s"  # in words
          if cell["d_w_long_missing"] else "long-missing off")
    la = (f"low-activity {cell['d_w_low_activity']:g} @ {cell['d_activity_thr_ms']:g} m/s"
          if cell["d_w_low_activity"] else "low-activity off")  # in words or off
    return f"{lm} · {la}"  # compact cell label


def _delta_row(name: str, deltas: dict) -> str:
    parts = [f"{deltas[(k, lab)][0]:+.4f} ({deltas[(k, lab)][1]}/7)"  # Δ (wins/7) per table
             for k in KINDS for lab in ("orig", "aligned")]
    return f"| {name} | " + " | ".join(parts) + " |"  # markdown row


def _header() -> list[str]:
    cols = [f"{k} × {lab}" for k in KINDS for lab in ("orig", "aligned")]  # one per table
    return ["| cell | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]  # header + separator


def stage4_drift(tables: dict, res: Path = RES) -> list[str]:
    """OFF on the corrected bed vs the committed (km/h bed) stage-4 frozen-cell rows."""
    out = []  # drift lines
    for kind in ("uniform", "phase_a2"):
        path = res / f"stage4_{kind}.csv"  # stage-4 file path
        if not path.exists():  # stage-4 file is optional
            continue  # nothing to compare
        old = pd.read_csv(path)  # committed stage-4 matrix
        for p, v in FROZEN_DECODER.items():  # reduce to the frozen decoder cell
            old = old[old[p] == v]  # keep frozen-cell rows
        old = old.set_index("match_id")  # index for reindexing
        for lab in ("orig", "aligned"):
            new = cell_scores(tables[kind], OFF, f"spell_f1_{lab}")  # OFF on corrected bed
            d = (new - old[f"spell_f1_{lab}"].reindex(new.index)).abs()  # per-match drift
            out.append(f"- {kind} × {lab}: mean |Δ| {d.mean():.4f}, max {d.max():.4f} "
                       f"(corrected {new.mean():.4f} vs committed {old[f'spell_f1_{lab}'].mean():.4f})")  # one drift line
    return out  # all drift lines


def build_report(res: Path = RES) -> tuple[str, dict]:
    tables = load_tables(res)  # load the matrices
    u1 = u1_verdict(tables)  # U1 runs first: it sets the reference
    ref = OFF if u1["adopt"] else AS_SHIPPED  # D1 reference follows the U1 verdict
    read1, read2 = r1(tables, ref), r2(tables, ref)  # both D1 readings
    base = pd.read_csv(res / "stage5_baselines.csv")  # organisers' baselines (B1)
    masks = pd.read_csv(res / "stage5_masks.csv")  # mask statistics

    L = ["# Stage 5 — P7 audit follow-up pass (U1, D1, B1)", "",  # report header
         "Pre-declared: `dossier.md` §7 P7 (2026-11-14). Corrected units (m/s), frozen decoder",
         "cell, four mask kinds × two label sets, 7 IDSSE matches. Δ = 7-match mean of paired",
         "spellF1 differences; (n/7) = matches where the difference is positive.", "",
         "## Mask statistics (7-match means)", "",
         "| kind | masked | P(missing given dead) | P(missing given alive) | P(dead given missing) |",
         "|---|---|---|---|---|"]
    for kind in KINDS:  # one mask-stats row per kind
        m = masks[masks["kind"] == kind].mean(numeric_only=True)  # means for this kind
        L.append(f"| {kind} | {m['masked_fraction']:.3f} | {m['miss_given_dead']:.3f} | "
                 f"{m['miss_given_alive']:.3f} | {m['dead_given_miss']:.3f} |")  # one stats row

    L += ["", "## U1 — low-activity weight (OFF − as shipped)", ""] + _header()  # U1 section
    L.append(_delta_row("OFF − ON", u1["deltas"]))  # all eight tables
    L += ["", f"- primary (phase_a2 × aligned): **{u1['primary']:+.4f}**, positive "
          f"{u1['wins']}/7 (needs > {BAR} and ≥ {MIN_WINS}/7)",  # pinned conditions
          f"- no-harm on the other seven tables: **{'yes' if u1['no_harm'] else 'no'}**",
          f"- **Verdict: {'ADOPT — d_w_low_activity = 0' if u1['adopt'] else 'KEEP the shipped value'}**",
          "", "Corrected bed with the weight off vs the committed stage-4 frozen-cell rows:"]
    L += stage4_drift(tables, res)  # OFF vs committed stage-4 rows

    L += ["", f"## D1 — dead-ball term (reference: {cell_str(ref)})", ""]  # D1 section
    for name, read, regimes in (("R1 — regime-agnostic (automatic)", read1, R1_REGIMES),
                                ("R2 — coupled-regime reading (operator option)", read2, R2_REGIMES)):
        c = read["candidate"]  # shorthand for candidate
        L += [f"### {name}", "",  # one subsection per reading
              f"- regimes: {', '.join(regimes)} (aligned labels); best maximin gain "
              f"{read['best_g']:+.4f}; {read['n_tied']} cell(s) within {TIE} of it",
              f"- candidate: **{cell_str(c['cell'])}** — maximin gain **{c['g']:+.4f}**",
              f"- gain > {BAR}: **{'yes' if c['g'] > BAR else 'no'}** · ≥ {MIN_WINS}/7 in every "
              f"regime: **{'yes' if read['wins_ok'] else 'no'}** · no-harm: "  # conditions
              f"**{'yes' if read['harm_ok'] else 'no'}**",
              f"- **Verdict: {'PASS' if read['adopt'] else 'REJECT'}**", ""] + _header()
        for r in read["ranked"][:8]:  # top-8 ranked cells
            L.append(_delta_row(cell_str(r["cell"]), r["deltas"]))  # one ranked row
        L.append("")  # blank separator line

    L += ["## Predicted dead share (7-match mean; true IDSSE dead share ≈ 0.45)", "",  # diagnostics
          "| cell | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    shown = [("as shipped", AS_SHIPPED), ("reference", ref),  # cells to display
             ("R1 candidate", read1["candidate"]["cell"]),
             ("R2 candidate", read2["candidate"]["cell"])]
    for name, cell in shown:  # one row per displayed cell
        vals = [cell_scores(tables[k], cell, "pred_d_share").mean() for k in KINDS]  # D shares
        L.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")  # one cell row

    L += ["", "## B1 — organisers' baselines on the same bed (spellF1 / macroF1, aligned labels)",  # B1
          "", "| system | " + " | ".join(KINDS) + " | published full match |",
          "|---|" + "---|" * (len(KINDS) + 1)]
    for system, (pub_s, pub_m) in PUBLISHED.items():  # one row per baseline system
        vals = []  # per-kind means
        for k in KINDS:  # one kind per entry
            b = base[(base["kind"] == k) & (base["system"] == system)]  # that system's rows
            vals.append(f"{b['spell_f1_aligned'].mean():.3f} / {b['macro_f1_aligned'].mean():.3f}")  # spellF1 / macroF1
        L.append(f"| {system} | " + " | ".join(vals) + f" | {pub_s:.3f} / {pub_m:.3f} |")  # baseline row
    for name, cell in shown[:2]:  # our two cells, same bed, for comparison
        vals = [f"{cell_scores(tables[k], cell, 'spell_f1_aligned').mean():.3f} / "
                f"{cell_scores(tables[k], cell, 'macro_f1_aligned').mean():.3f}" for k in KINDS]  # our means
        L.append(f"| ours, {name} | " + " | ".join(vals) + " | — |")  # our two rows
    L += ["", "The published column is the organisers' own figure for the provided match "
          "(EVALUATION.md); it is shown for orientation and selects nothing (dossier §7, Q5).",
          "", "## Reproduce", "", "```bash",  # orientation note + reproduce block
          ".venv/bin/python -m tuning.stage5          # full pass (~15 min, 6 workers)",
          ".venv/bin/python -m tuning.gate5           # this read + report", "```", ""]
    summary = {"u1": u1, "ref": ref, "r1": read1, "r2": read2}  # results for main()
    return "\n".join(L), summary  # report text + results


def main() -> int:
    text, summary = build_report()  # apply all pinned rules, read-only
    (RES / "report_stage5.md").write_text(text)  # write the report
    pd.DataFrame([  # one verdict row per gate
        {"gate": "U1", "verdict": "ADOPT" if summary["u1"]["adopt"] else "KEEP",
         "cell": cell_str(OFF), "gain": round(summary["u1"]["primary"], 4)},
        {"gate": "D1_R1", "verdict": "PASS" if summary["r1"]["adopt"] else "REJECT",
         "cell": cell_str(summary["r1"]["candidate"]["cell"]),
         "gain": round(summary["r1"]["candidate"]["g"], 4)},
        {"gate": "D1_R2", "verdict": "PASS" if summary["r2"]["adopt"] else "REJECT",
         "cell": cell_str(summary["r2"]["candidate"]["cell"]),
         "gain": round(summary["r2"]["candidate"]["g"], 4)},
    ]).to_csv(RES / "stage5_verdicts.csv", index=False)  # machine-readable verdicts
    print(text)  # echo the report
    return 0  # success


if __name__ == "__main__":
    sys.exit(main())  # python -m tuning.gate5 entry point
