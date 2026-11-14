"""Gate 5 — apply the dossier §7 P7 rules (U1, D1 R1/R2, B1) to the stage-5 matrices.

    python -m tuning.gate5        # prints the read, writes tuning/results/report_stage5.md

Read-only over `stage5_*.csv`. It never writes config: verdicts are recorded in the
dossier by hand, and only R1 can be adopted without a further operator decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.stage5 import AS_SHIPPED, D_PARAMS, FROZEN_DECODER, KINDS  # noqa: E402

RES = ROOT / "tuning" / "results"
OFF = {"d_w_long_missing": 0.5, "d_missing_run_s": 10.0,
       "d_w_low_activity": 0.0, "d_activity_thr_ms": 1.0}
PRIMARY_U1 = ("phase_a2", "aligned")
R1_REGIMES = ("uniform", "cover_q50", "cover_q90")
R2_REGIMES = ("cover_q50", "cover_q90")
BAR, MIN_WINS, HARM, TIE = 0.02, 5, -0.02, 0.005
PUBLISHED = {"baseline_raw": (0.0486, 0.6532), "baseline_smooth25": (0.1996, 0.6752),
             "baseline_smooth50": (0.2433, 0.6724), "baseline_smooth125": (0.2242, 0.6356)}


def key(cell: dict) -> tuple:
    return tuple(float(cell[p]) for p in D_PARAMS)


def load_tables(res: Path = RES) -> dict[str, pd.DataFrame]:
    tables = {}
    for kind in KINDS:
        df = pd.read_csv(res / f"stage5_{kind}.csv")
        n_cells = df.groupby(D_PARAMS).ngroups
        if len(df) != n_cells * 7:
            raise ValueError(f"{kind}: {len(df)} rows for {n_cells} cells x 7 matches")
        tables[kind] = df.set_index(D_PARAMS + ["match_id"]).sort_index()
    return tables


def cell_scores(df: pd.DataFrame, cell: dict, metric: str) -> pd.Series:
    """7 per-match values of `metric` for one cell, indexed by match_id."""
    return df.xs(key(cell), level=D_PARAMS)[metric]


def paired(df: pd.DataFrame, a: dict, b: dict, label: str) -> tuple[float, int]:
    d = (cell_scores(df, a, f"spell_f1_{label}") - cell_scores(df, b, f"spell_f1_{label}"))
    return float(d.mean()), int((d > 0).sum())


def all_deltas(tables: dict, a: dict, b: dict) -> dict[tuple[str, str], tuple[float, int]]:
    return {(kind, lab): paired(tables[kind], a, b, lab)
            for kind in KINDS for lab in ("orig", "aligned")}


def u1_verdict(tables: dict) -> dict:
    deltas = all_deltas(tables, OFF, AS_SHIPPED)
    d, wins = deltas[PRIMARY_U1]
    others_ok = all(v[0] >= HARM for k, v in deltas.items() if k != PRIMARY_U1)
    return {"deltas": deltas, "primary": d, "wins": wins, "no_harm": others_ok,
            "adopt": bool(d > BAR and wins >= MIN_WINS and others_ok)}


def n_changed(cell: dict, ref: dict) -> tuple[int, float]:
    """(effective parameters changed, summed weight change) — the pinned tie-break."""
    changed = 0
    w_lm, r_lm = cell["d_w_long_missing"], ref["d_w_long_missing"]
    w_la, r_la = cell["d_w_low_activity"], ref["d_w_low_activity"]
    changed += w_lm != r_lm
    changed += w_la != r_la
    if w_lm and r_lm:
        changed += cell["d_missing_run_s"] != ref["d_missing_run_s"]
    if w_la and r_la:
        changed += cell["d_activity_thr_ms"] != ref["d_activity_thr_ms"]
    return int(changed), float(abs(w_lm - r_lm) + abs(w_la - r_la))


def cells_of(tables: dict) -> list[dict]:
    idx = tables[KINDS[0]].index.droplevel("match_id").unique()
    return [dict(zip(D_PARAMS, map(float, k))) for k in idx]


def d1_reading(tables: dict, ref: dict, regimes: tuple[str, ...], veto_tables: list) -> dict:
    """Maximin candidate over `regimes` (aligned labels) + the pinned adoption conditions."""
    rows = []
    for cell in cells_of(tables):
        if key(cell) == key(ref):
            continue
        deltas = all_deltas(tables, cell, ref)
        g = min(deltas[(r, "aligned")][0] for r in regimes)
        rows.append({"cell": cell, "g": g, "deltas": deltas,
                     "tie": n_changed(cell, ref)})
    best = max(r["g"] for r in rows)
    tied = [r for r in rows if r["g"] >= best - TIE]
    cand = min(tied, key=lambda r: (r["tie"][0], r["tie"][1], -r["g"]))
    wins_ok = all(cand["deltas"][(r, "aligned")][1] >= MIN_WINS for r in regimes)
    harm_ok = all(cand["deltas"][t][0] >= HARM for t in veto_tables)
    return {"candidate": cand, "best_g": best, "n_tied": len(tied),
            "wins_ok": wins_ok, "harm_ok": harm_ok,
            "adopt": bool(cand["g"] > BAR and wins_ok and harm_ok),
            "ranked": sorted(rows, key=lambda r: -r["g"])}


def r1(tables: dict, ref: dict) -> dict:
    veto = [("phase_a2", "aligned")] + [(k, "orig") for k in KINDS]
    return d1_reading(tables, ref, R1_REGIMES, veto)


def r2(tables: dict, ref: dict) -> dict:
    return d1_reading(tables, ref, R2_REGIMES, [(k, "orig") for k in R2_REGIMES])


def cell_str(cell: dict) -> str:
    lm = (f"long-missing {cell['d_w_long_missing']:g} / {cell['d_missing_run_s']:g} s"
          if cell["d_w_long_missing"] else "long-missing off")
    la = (f"low-activity {cell['d_w_low_activity']:g} @ {cell['d_activity_thr_ms']:g} m/s"
          if cell["d_w_low_activity"] else "low-activity off")
    return f"{lm} · {la}"


def _delta_row(name: str, deltas: dict) -> str:
    parts = [f"{deltas[(k, lab)][0]:+.4f} ({deltas[(k, lab)][1]}/7)"
             for k in KINDS for lab in ("orig", "aligned")]
    return f"| {name} | " + " | ".join(parts) + " |"


def _header() -> list[str]:
    cols = [f"{k} × {lab}" for k in KINDS for lab in ("orig", "aligned")]
    return ["| cell | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]


def stage4_drift(tables: dict, res: Path = RES) -> list[str]:
    """OFF on the corrected bed vs the committed (km/h bed) stage-4 frozen-cell rows."""
    out = []
    for kind in ("uniform", "phase_a2"):
        path = res / f"stage4_{kind}.csv"
        if not path.exists():
            continue
        old = pd.read_csv(path)
        for p, v in FROZEN_DECODER.items():
            old = old[old[p] == v]
        old = old.set_index("match_id")
        for lab in ("orig", "aligned"):
            new = cell_scores(tables[kind], OFF, f"spell_f1_{lab}")
            d = (new - old[f"spell_f1_{lab}"].reindex(new.index)).abs()
            out.append(f"- {kind} × {lab}: mean |Δ| {d.mean():.4f}, max {d.max():.4f} "
                       f"(corrected {new.mean():.4f} vs committed {old[f'spell_f1_{lab}'].mean():.4f})")
    return out


def build_report(res: Path = RES) -> tuple[str, dict]:
    tables = load_tables(res)
    u1 = u1_verdict(tables)
    ref = OFF if u1["adopt"] else AS_SHIPPED
    read1, read2 = r1(tables, ref), r2(tables, ref)
    base = pd.read_csv(res / "stage5_baselines.csv")
    masks = pd.read_csv(res / "stage5_masks.csv")

    L = ["# Stage 5 — P7 audit follow-up pass (U1, D1, B1)", "",
         "Pre-declared: `dossier.md` §7 P7 (2026-11-14). Corrected units (m/s), frozen decoder",
         "cell, four mask kinds × two label sets, 7 IDSSE matches. Δ = 7-match mean of paired",
         "spellF1 differences; (n/7) = matches where the difference is positive.", "",
         "## Mask statistics (7-match means)", "",
         "| kind | masked | P(missing given dead) | P(missing given alive) | P(dead given missing) |",
         "|---|---|---|---|---|"]
    for kind in KINDS:
        m = masks[masks["kind"] == kind].mean(numeric_only=True)
        L.append(f"| {kind} | {m['masked_fraction']:.3f} | {m['miss_given_dead']:.3f} | "
                 f"{m['miss_given_alive']:.3f} | {m['dead_given_miss']:.3f} |")

    L += ["", "## U1 — low-activity weight (OFF − as shipped)", ""] + _header()
    L.append(_delta_row("OFF − ON", u1["deltas"]))
    L += ["", f"- primary (phase_a2 × aligned): **{u1['primary']:+.4f}**, positive "
          f"{u1['wins']}/7 (needs > {BAR} and ≥ {MIN_WINS}/7)",
          f"- no-harm on the other seven tables: **{'yes' if u1['no_harm'] else 'no'}**",
          f"- **Verdict: {'ADOPT — d_w_low_activity = 0' if u1['adopt'] else 'KEEP the shipped value'}**",
          "", "Corrected bed with the weight off vs the committed stage-4 frozen-cell rows:"]
    L += stage4_drift(tables, res)

    L += ["", f"## D1 — dead-ball term (reference: {cell_str(ref)})", ""]
    for name, read, regimes in (("R1 — regime-agnostic (automatic)", read1, R1_REGIMES),
                                ("R2 — coupled-regime reading (operator option)", read2, R2_REGIMES)):
        c = read["candidate"]
        L += [f"### {name}", "",
              f"- regimes: {', '.join(regimes)} (aligned labels); best maximin gain "
              f"{read['best_g']:+.4f}; {read['n_tied']} cell(s) within {TIE} of it",
              f"- candidate: **{cell_str(c['cell'])}** — maximin gain **{c['g']:+.4f}**",
              f"- gain > {BAR}: **{'yes' if c['g'] > BAR else 'no'}** · ≥ {MIN_WINS}/7 in every "
              f"regime: **{'yes' if read['wins_ok'] else 'no'}** · no-harm: "
              f"**{'yes' if read['harm_ok'] else 'no'}**",
              f"- **Verdict: {'PASS' if read['adopt'] else 'REJECT'}**", ""] + _header()
        for r in read["ranked"][:8]:
            L.append(_delta_row(cell_str(r["cell"]), r["deltas"]))
        L.append("")

    L += ["## Predicted dead share (7-match mean; true IDSSE dead share ≈ 0.45)", "",
          "| cell | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    shown = [("as shipped", AS_SHIPPED), ("reference", ref),
             ("R1 candidate", read1["candidate"]["cell"]),
             ("R2 candidate", read2["candidate"]["cell"])]
    for name, cell in shown:
        vals = [cell_scores(tables[k], cell, "pred_d_share").mean() for k in KINDS]
        L.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    L += ["", "## B1 — organisers' baselines on the same bed (spellF1 / macroF1, aligned labels)",
          "", "| system | " + " | ".join(KINDS) + " | published full match |",
          "|---|" + "---|" * (len(KINDS) + 1)]
    for system, (pub_s, pub_m) in PUBLISHED.items():
        vals = []
        for k in KINDS:
            b = base[(base["kind"] == k) & (base["system"] == system)]
            vals.append(f"{b['spell_f1_aligned'].mean():.3f} / {b['macro_f1_aligned'].mean():.3f}")
        L.append(f"| {system} | " + " | ".join(vals) + f" | {pub_s:.3f} / {pub_m:.3f} |")
    for name, cell in shown[:2]:
        vals = [f"{cell_scores(tables[k], cell, 'spell_f1_aligned').mean():.3f} / "
                f"{cell_scores(tables[k], cell, 'macro_f1_aligned').mean():.3f}" for k in KINDS]
        L.append(f"| ours, {name} | " + " | ".join(vals) + " | — |")
    L += ["", "The published column is the organisers' own figure for the provided match "
          "(EVALUATION.md); it is shown for orientation and selects nothing (dossier §7, Q5).",
          "", "## Reproduce", "", "```bash",
          ".venv/bin/python -m tuning.stage5          # full pass (~15 min, 6 workers)",
          ".venv/bin/python -m tuning.gate5           # this read + report", "```", ""]
    summary = {"u1": u1, "ref": ref, "r1": read1, "r2": read2}
    return "\n".join(L), summary


def main() -> int:
    text, summary = build_report()
    (RES / "report_stage5.md").write_text(text)
    pd.DataFrame([
        {"gate": "U1", "verdict": "ADOPT" if summary["u1"]["adopt"] else "KEEP",
         "cell": cell_str(OFF), "gain": round(summary["u1"]["primary"], 4)},
        {"gate": "D1_R1", "verdict": "PASS" if summary["r1"]["adopt"] else "REJECT",
         "cell": cell_str(summary["r1"]["candidate"]["cell"]),
         "gain": round(summary["r1"]["candidate"]["g"], 4)},
        {"gate": "D1_R2", "verdict": "PASS" if summary["r2"]["adopt"] else "REJECT",
         "cell": cell_str(summary["r2"]["candidate"]["cell"]),
         "gain": round(summary["r2"]["candidate"]["g"], 4)},
    ]).to_csv(RES / "stage5_verdicts.csv", index=False)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
