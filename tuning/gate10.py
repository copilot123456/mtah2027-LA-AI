"""Gate 10 — apply the dossier §7 P10 rule to stage 10.

    python -m tuning.gate10     # prints the read, writes report_stage10.md

Read-only over `stage10_*.csv`. It never writes config: an adoption is recorded in the
dossier and in `config.py` by hand. The harness pin compares the 0.3 cell against the
stage-9 bundle row; a mismatch voids the pass. The smallest K weight passing the pinned
test is the one chosen.
"""

from __future__ import annotations  # postponed annotations

import sys  # sys.path bootstrapping
from pathlib import Path  # path constants

import pandas as pd  # CSV reads and report tables

ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # make the repo importable from any cwd

# pinned constants, imported after the path bootstrap
# stage-9 helpers (cell_col, load_tables)
from tuning.gate9 import cell_col, load_tables as load_stage9  # noqa: E402
# the four mask kinds
from tuning.stage5 import KINDS  # noqa: E402
# P10 grid + shipped cell
from tuning.stage10 import K_WEIGHTS, P_PARAMS, SHIPPED  # noqa: E402

RES = ROOT / "tuning" / "results"  # matrices read here; report written here
PRIMARY = ("phase_a2", "aligned")  # deciding table for the keep-test
TABLES = [(k, lab) for k in KINDS for lab in ("orig", "aligned")]  # report columns
# thresholds: keep bar, min wins, harm floor, pin tolerance
BAR, MIN_WINS, HARM, PIN_TOL = 0.02, 5, -0.02, 1e-9
SCORE_COLS = ("spell_f1_orig", "spell_f1_aligned", "macro_f1_orig", "macro_f1_aligned")  # pin cols


def load_tables(res: Path = RES) -> dict[str, pd.DataFrame]:
    tables = {}  # one entry per mask kind
    for kind in KINDS:
        df = pd.read_csv(res / f"stage10_{kind}.csv")  # one matrix per mask kind
        n_cells = df.groupby(P_PARAMS).ngroups  # distinct K weights
        if len(df) != n_cells * 7:  # every cell must cover all 7 matches
            raise ValueError(f"{kind}: {len(df)} rows for {n_cells} cells x 7 matches")  # reject a malformed matrix
        tables[kind] = df.set_index(P_PARAMS + ["match_id"]).sort_index()  # key = cell + match
    return tables  # return the four matrices


def harness_pin(tables: dict, res: Path = RES) -> float:
    """Largest |stage-10 0.3 cell − stage-9 bundle row| over the score columns."""
    s9 = load_stage9(res)  # stage-9 tables to compare against
    worst = 0.0  # running worst drift
    for kind in KINDS:
        for col in SCORE_COLS:
            new = tables[kind].xs(float(SHIPPED["kin_w_dead"]), level="kin_w_dead")[col]  # 0.3
            old = cell_col(s9[kind], SHIPPED, col)  # stage-9 bundle row
            worst = max(worst, float((new - old.reindex(new.index)).abs().max()))  # keep the largest drift
    return worst  # 0.0 means the harness still pins


def deltas(tables: dict, w: float, ref: float) -> dict:
    """{(kind, label set): (mean paired spellF1 difference w − ref, matches positive)}."""
    out = {}  # deltas keyed by table
    for kind, lab in TABLES:
        d = (tables[kind].xs(w, level="kin_w_dead")[f"spell_f1_{lab}"]  # w − ref per match
             - tables[kind].xs(ref, level="kin_w_dead")[f"spell_f1_{lab}"])
        out[(kind, lab)] = (float(d.mean()), int((d > 0).sum()))  # mean Δ, wins/7
    return out  # return means and wins


def apply_rule(tables: dict) -> tuple[list[dict], float | None]:
    ref = float(SHIPPED["kin_w_dead"])  # shipped K weight (0.3)
    rows, chosen = [], None  # verdict list and winner
    for w in K_WEIGHTS:  # ascending grid; first passer wins
        if w == ref:
            continue  # the reference itself is not a candidate
        d = deltas(tables, w, ref)  # this weight vs reference
        prim, prim_n = d[PRIMARY]  # deciding table's Δ and wins
        others = {t: v for t, v in d.items() if t != PRIMARY}  # tables besides the primary
        worst = min(v[0] for v in others.values())  # worst other table
        keep = prim > BAR and prim_n >= MIN_WINS and worst >= HARM  # pinned keep-test
        rows.append({"kin_w_dead": w, "primary_delta": prim, "primary_wins": prim_n,  # store this weight's verdict
                     "worst_other_delta": worst, "verdict": "ADOPT" if keep else "reject"})
        if keep and chosen is None:
            chosen = w  # smallest passing weight (pinned order)
    return rows, chosen  # return all rows and winner


def fmt_d(v: float, n: int) -> str:
    return f"{v:+.4f} ({n}/7)"  # mean Δ (wins/7) cell format


def write_report(tables: dict, verdicts: pd.DataFrame, chosen: float | None,
                 pin: float, record: pd.DataFrame) -> Path:
    ref = float(SHIPPED["kin_w_dead"])  # shipped K weight (0.3)
    # report preamble: pre-declared context, pin line, table header
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
    for _, v in verdicts.iterrows():  # one table row per K weight
        w = float(v["kin_w_dead"])  # weight for this row
        d = deltas(tables, w, ref)  # deltas against the 0.3 cell
        cells = " | ".join(fmt_d(*d[t]) for t in TABLES)  # Δ per kind × label set
        lines.append(f"| K {w:g} | {cells} |")  # append the table row
    # pinned rule recap: thresholds, then one line per weight
    lines += ["", "### Pinned rule (smallest w: primary > +0.02, ≥ 5/7, no-harm ≥ −0.02)", ""]
    for _, v in verdicts.iterrows():  # one recap line per weight
        w = float(v["kin_w_dead"])  # weight for this recap
        lines.append(f"- K {w:g}: primary {fmt_d(v['primary_delta'], int(v['primary_wins']))}, "  # append the verdict recap
                     f"worst other table {v['worst_other_delta']:+.4f} — **{str(v['verdict']).upper()}**")
    # overall step verdict: ADOPT the chosen weight, else REJECT (ship unchanged)
    lines += ["", f"- **Step verdict: "
              + (f"ADOPT `kin_w_dead = {chosen:g}`" if chosen is not None
                 else f"REJECT — `kin_w_dead = {ref:g}` ships unchanged")
              + "**", ""]

    # 7-match means: aligned spellF1, predicted dead share, changed share
    lines += ["## Levels, predicted dead share and changed share (7-match means)", "",
              "| cell | uniform aligned | phase_a2 aligned | cover_q50 aligned | "
              "cover_q90 aligned | uniform D | phase_a2 D | cover_q50 D | cover_q90 D | "
              "uniform changed | phase_a2 changed | cover_q50 changed | cover_q90 changed |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in K_WEIGHTS:  # one row per K weight
        cells = []  # means for this weight
        for lab_col in ("spell_f1_aligned", "pred_d_share", "changed_share"):  # each measure in turn
            for kind in KINDS:  # one column per mask kind
                cells.append(f"{tables[kind].xs(w, level='kin_w_dead')[lab_col].mean():.4f}")  # append a seven-match mean
        lines.append(f"| K {w:g} | " + " | ".join(cells) + " |")  # append the table row

    # external firing: per-match changed share vs the 0.3 cell
    lines += ["", "## External firing — share of in-match frames changed vs the 0.3 cell, "
              "smallest … largest per-match value over the four mask kinds", ""]
    for w in K_WEIGHTS:
        if w == ref:
            continue  # the reference never fires against itself
        vals = [tables[k].xs(w, level="kin_w_dead")["changed_share"] for k in KINDS]  # changed share per kind
        lo = min(float(s.min()) for s in vals)  # smallest per-match value
        hi = max(float(s.max()) for s in vals)  # largest per-match value
        lines.append(f"- K {w:g}: {lo:.1%} … {hi:.1%}")  # append the range line

    # record-only rows: never adopted, with the reason below the table
    lines += ["", "## Record-only rows (never adopted)", "",
              "| cell | uniform aligned | phase_a2 aligned | cover_q50 aligned | "
              "cover_q90 aligned | uniform D | phase_a2 D | cover_q50 D | cover_q90 D |",
              "|---|---|---|---|---|---|---|---|"]
    for cell in sorted(record["cell"].unique()):  # one row per record-only cell
        r = record[record["cell"] == cell]  # rows for this cell
        cells = []  # means for this cell
        for lab_col in ("spell_f1_aligned", "pred_d_share"):  # two measures in turn
            for kind in KINDS:  # one column per mask kind
                cells.append(f"{r[r['kind'] == kind][lab_col].mean():.4f}")  # append a seven-match mean
        lines.append(f"| {cell} | " + " | ".join(cells) + " |")  # append the record row
    lines += ["",  # closing note: what the record rows mean
              "`R_K05_all` puts the dead term on every frame (ball-visible included) and "
              "`R_K10_T0` reads the K×T interaction at K 1.0 with T 0 — both record-only; "
              "the bed cannot judge dead-ball features on ball-visible frames.", ""]

    path = RES / "report_stage10.md"  # report file location
    path.write_text("\n".join(lines))  # write the report
    return path  # where the report went


def main() -> int:
    tables = load_tables()  # load stage-10 matrices
    pin = harness_pin(tables)  # 0.3 cell vs stage-9 bundle row
    print(f"harness pin vs stage-9 bundle row: max |Δ| = {pin:.1e}")  # show the pin value
    if pin > PIN_TOL:
        print("PIN FAILED — the pass is void; do not read the matrices.")  # warn the operator
        return 1  # matrices untrustworthy → stop before the rule
    verdicts, chosen = apply_rule(tables)  # run the pinned P10 rule
    record = pd.read_csv(RES / "stage10_record.csv")  # record-only rows
    path = write_report(tables, pd.DataFrame(verdicts), chosen, pin, record)  # write the report file
    pd.DataFrame(verdicts).to_csv(RES / "stage10_verdicts.csv", index=False)  # machine-readable
    for v in verdicts:  # echo one line per weight
        print(f"K {v['kin_w_dead']:g}: primary {fmt_d(v['primary_delta'], v['primary_wins'])}, "  # one line per weight
              f"worst other {v['worst_other_delta']:+.4f} — {v['verdict'].upper()}")
    print("ADOPT kin_w_dead = %g" % chosen if chosen is not None  # final verdict
          else "REJECT — kin_w_dead = %g ships unchanged" % float(SHIPPED["kin_w_dead"]))
    print(f"wrote {path.name} and stage10_verdicts.csv")  # confirm written files
    return 0  # success


if __name__ == "__main__":  # run as script
    sys.exit(main())  # exit with status
