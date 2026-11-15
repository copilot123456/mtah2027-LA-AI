"""Gate 10 — apply the dossier §7 P10 rule to stage 10.

    python -m tuning.gate10     # prints the read, writes report_stage10.md

Read-only over `stage10_*.csv`. It never writes config: an adoption is recorded in the
dossier and in `config.py` by hand. The harness pin compares the 0.3 cell against the
stage-9 bundle row; a mismatch voids the pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.gate9 import cell_col, load_tables as load_stage9  # noqa: E402
from tuning.stage5 import KINDS  # noqa: E402
from tuning.stage10 import K_WEIGHTS, P_PARAMS, SHIPPED  # noqa: E402

RES = ROOT / "tuning" / "results"
PRIMARY = ("phase_a2", "aligned")
TABLES = [(k, lab) for k in KINDS for lab in ("orig", "aligned")]
BAR, MIN_WINS, HARM, PIN_TOL = 0.02, 5, -0.02, 1e-9
SCORE_COLS = ("spell_f1_orig", "spell_f1_aligned", "macro_f1_orig", "macro_f1_aligned")


def load_tables(res: Path = RES) -> dict[str, pd.DataFrame]:
    tables = {}
    for kind in KINDS:
        df = pd.read_csv(res / f"stage10_{kind}.csv")
        n_cells = df.groupby(P_PARAMS).ngroups
        if len(df) != n_cells * 7:
            raise ValueError(f"{kind}: {len(df)} rows for {n_cells} cells x 7 matches")
        tables[kind] = df.set_index(P_PARAMS + ["match_id"]).sort_index()
    return tables


def harness_pin(tables: dict, res: Path = RES) -> float:
    """Largest |stage-10 0.3 cell − stage-9 bundle row| over the score columns."""
    s9 = load_stage9(res)
    worst = 0.0
    for kind in KINDS:
        for col in SCORE_COLS:
            new = tables[kind].xs(float(SHIPPED["kin_w_dead"]), level="kin_w_dead")[col]
            old = cell_col(s9[kind], SHIPPED, col)
            worst = max(worst, float((new - old.reindex(new.index)).abs().max()))
    return worst


def deltas(tables: dict, w: float, ref: float) -> dict:
    """{(kind, label set): (mean paired spellF1 difference w − ref, matches positive)}."""
    out = {}
    for kind, lab in TABLES:
        d = (tables[kind].xs(w, level="kin_w_dead")[f"spell_f1_{lab}"]
             - tables[kind].xs(ref, level="kin_w_dead")[f"spell_f1_{lab}"])
        out[(kind, lab)] = (float(d.mean()), int((d > 0).sum()))
    return out


def apply_rule(tables: dict) -> tuple[list[dict], float | None]:
    ref = float(SHIPPED["kin_w_dead"])
    rows, chosen = [], None
    for w in K_WEIGHTS:
        if w == ref:
            continue
        d = deltas(tables, w, ref)
        prim, prim_n = d[PRIMARY]
        others = {t: v for t, v in d.items() if t != PRIMARY}
        worst = min(v[0] for v in others.values())
        keep = prim > BAR and prim_n >= MIN_WINS and worst >= HARM
        rows.append({"kin_w_dead": w, "primary_delta": prim, "primary_wins": prim_n,
                     "worst_other_delta": worst, "verdict": "ADOPT" if keep else "reject"})
        if keep and chosen is None:
            chosen = w
    return rows, chosen


def fmt_d(v: float, n: int) -> str:
    return f"{v:+.4f} ({n}/7)"


def write_report(tables: dict, verdicts: pd.DataFrame, chosen: float | None,
                 pin: float, record: pd.DataFrame) -> Path:
    ref = float(SHIPPED["kin_w_dead"])
    lines = [
        "# Stage 10 — P10 pass (kin_w_dead past the grid edge)",
        "",
        "Pre-declared: `dossier.md` §7 P10 (2026-11-15). Shipped bundle cell as the "
        "reference (coalesce on · K 0.3 · T 0.03 · tier kin · delay 0.64 s · slope 0), "
        "four mask kinds × two label sets, 7 IDSSE matches; kinematic models fitted "
        "leave-one-match-out, kinematic features of the test match from "
        "dropout-degraded player tracks. Δ = 7-match mean of paired spellF1 differences "
        "against the 0.3 cell; (n/7) = matches where the difference is positive.",
        "",
        f"Harness pin: the 0.3 cell reproduces the stage-9 bundle row, max |Δ| = {pin:.1e}.",
        "",
        "## K weight (against: the shipped bundle, K 0.3)",
        "",
        "| cell | uniform × orig | uniform × aligned | phase_a2 × orig | phase_a2 × "
        "aligned | cover_q50 × orig | cover_q50 × aligned | cover_q90 × orig | "
        "cover_q90 × aligned |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, v in verdicts.iterrows():
        w = float(v["kin_w_dead"])
        d = deltas(tables, w, ref)
        cells = " | ".join(fmt_d(*d[t]) for t in TABLES)
        lines.append(f"| K {w:g} | {cells} |")
    lines += ["", "### Pinned rule (smallest w: primary > +0.02, ≥ 5/7, no-harm ≥ −0.02)", ""]
    for _, v in verdicts.iterrows():
        w = float(v["kin_w_dead"])
        lines.append(f"- K {w:g}: primary {fmt_d(v['primary_delta'], int(v['primary_wins']))}, "
                     f"worst other table {v['worst_other_delta']:+.4f} — **{str(v['verdict']).upper()}**")
    lines += ["", f"- **Step verdict: "
              + (f"ADOPT `kin_w_dead = {chosen:g}`" if chosen is not None
                 else f"REJECT — `kin_w_dead = {ref:g}` ships unchanged")
              + "**", ""]

    lines += ["## Levels, predicted dead share and changed share (7-match means)", "",
              "| cell | uniform aligned | phase_a2 aligned | cover_q50 aligned | "
              "cover_q90 aligned | uniform D | phase_a2 D | cover_q50 D | cover_q90 D | "
              "uniform changed | phase_a2 changed | cover_q50 changed | cover_q90 changed |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in K_WEIGHTS:
        cells = []
        for lab_col in ("spell_f1_aligned", "pred_d_share", "changed_share"):
            for kind in KINDS:
                cells.append(f"{tables[kind].xs(w, level='kin_w_dead')[lab_col].mean():.4f}")
        lines.append(f"| K {w:g} | " + " | ".join(cells) + " |")

    lines += ["", "## External firing — share of in-match frames changed vs the 0.3 cell, "
              "smallest … largest per-match value over the four mask kinds", ""]
    for w in K_WEIGHTS:
        if w == ref:
            continue
        vals = [tables[k].xs(w, level="kin_w_dead")["changed_share"] for k in KINDS]
        lo = min(float(s.min()) for s in vals)
        hi = max(float(s.max()) for s in vals)
        lines.append(f"- K {w:g}: {lo:.1%} … {hi:.1%}")

    lines += ["", "## Record-only rows (never adopted)", "",
              "| cell | uniform aligned | phase_a2 aligned | cover_q50 aligned | "
              "cover_q90 aligned | uniform D | phase_a2 D | cover_q50 D | cover_q90 D |",
              "|---|---|---|---|---|---|---|---|"]
    for cell in sorted(record["cell"].unique()):
        r = record[record["cell"] == cell]
        cells = []
        for lab_col in ("spell_f1_aligned", "pred_d_share"):
            for kind in KINDS:
                cells.append(f"{r[r['kind'] == kind][lab_col].mean():.4f}")
        lines.append(f"| {cell} | " + " | ".join(cells) + " |")
    lines += ["",
              "`R_K05_all` puts the dead term on every frame (ball-visible included) and "
              "`R_K10_T0` reads the K×T interaction at K 1.0 with T 0 — both record-only; "
              "the bed cannot judge dead-ball features on ball-visible frames.", ""]

    path = RES / "report_stage10.md"
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    tables = load_tables()
    pin = harness_pin(tables)
    print(f"harness pin vs stage-9 bundle row: max |Δ| = {pin:.1e}")
    if pin > PIN_TOL:
        print("PIN FAILED — the pass is void; do not read the matrices.")
        return 1
    verdicts, chosen = apply_rule(tables)
    record = pd.read_csv(RES / "stage10_record.csv")
    path = write_report(tables, pd.DataFrame(verdicts), chosen, pin, record)
    pd.DataFrame(verdicts).to_csv(RES / "stage10_verdicts.csv", index=False)
    for v in verdicts:
        print(f"K {v['kin_w_dead']:g}: primary {fmt_d(v['primary_delta'], v['primary_wins'])}, "
              f"worst other {v['worst_other_delta']:+.4f} — {v['verdict'].upper()}")
    print("ADOPT kin_w_dead = %g" % chosen if chosen is not None
          else "REJECT — kin_w_dead = %g ships unchanged" % float(SHIPPED["kin_w_dead"]))
    print(f"wrote {path.name} and stage10_verdicts.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
