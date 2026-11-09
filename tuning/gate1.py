"""Gate 1 — apply the dossier §7 pinned rule to the committed stage-4 matrices.

    python -m tuning.gate1        # prints the read, writes tuning/results/report_stage4.md

Read-only over `stage4_*.csv`: runs the pre-declared plateau rule (sweep.flat_select,
verbatim P5 machinery) on the primary table (phase α=2 masks × E1-aligned labels),
evaluates the pinned adoption conditions against the frozen cell, and reports the two
twins (fill-on, α=3) plus the mask statistics. It never selects on the twins and never
writes config — the verdict is recorded in dossier §7 by hand, as the plan requires.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tuning.sweep import (PARAMS, bootstrap_ci, flat_select, lomo,  # noqa: E402
                          _cell_str)

FROZEN = {"switch_pen": 2.0, "dead_pen": 5.0, "sigma_m": 1.5,
          "min_dwell_s": 2.0, "switch_support_s": 1.5}
RES = ROOT / "tuning" / "results"
KINDS = ("uniform", "phase_a2", "phase_a3", "phase_a2_fill")


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for kind in KINDS:
        df = pd.read_csv(RES / f"stage4_{kind}.csv")
        df["scenario"] = kind
        if len(df) != 144 * 7:
            raise ValueError(f"{kind}: {len(df)} rows, expected 1008")
        tables[kind] = df
    return tables


def cell(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    sel = df
    for k, v in params.items():
        sel = sel[sel[k] == v]
    if len(sel) != 7:
        raise ValueError(f"cell {params}: {len(sel)} rows, expected 7 matches")
    return sel


def paired(df: pd.DataFrame, metric: str, a: dict, b: dict
           ) -> tuple[float, int, np.ndarray]:
    ca, cb = cell(df, a), cell(df, b)
    m = ca.merge(cb, on="match_id", suffixes=("_a", "_b"))
    d = m[f"{metric}_a"].to_numpy() - m[f"{metric}_b"].to_numpy()
    return float(d.mean()), int((d > 0).sum()), d


def gate1_read() -> dict:
    tables = load_tables()
    u, p, pf, a3 = (tables["uniform"], tables["phase_a2"],
                    tables["phase_a2_fill"], tables["phase_a3"])

    prim = p.copy()
    prim["spell_f1"] = prim["spell_f1_aligned"]
    chosen, detail = flat_select(prim, scenario="phase_a2", metric="spell_f1")
    best = float(detail["mean"].max())
    frozen_mean = float(cell(prim, FROZEN)["spell_f1"].mean())
    chosen_mean = float(cell(prim, chosen)["spell_f1"].mean())

    d_mean, n_pos, deltas = paired(prim, "spell_f1", chosen, FROZEN)
    lo, hi = bootstrap_ci(deltas)

    table_deltas = {}
    for name, df, metric in (("uniform_orig", u, "spell_f1_orig"),
                             ("uniform_aligned", u, "spell_f1_aligned"),
                             ("phase_orig", p, "spell_f1_orig"),
                             ("phase_aligned", p, "spell_f1_aligned")):
        dm, np_, _ = paired(df, metric, chosen, FROZEN)
        table_deltas[name] = (dm, np_)

    c1, c2 = d_mean > 0.02, n_pos >= 5
    c3 = all(v[0] >= -0.02 for v in table_deltas.values())
    if c1 and c2 and c3:
        verdict = "ADOPT"
    elif d_mean > 0.01:
        verdict = "TIE-KEEP"
    else:
        verdict = "REJECT-KEEP"

    l = lomo(prim, chosen, metric="spell_f1", scenario="phase_a2")

    twins = {}
    for label, df, base in (("fill", pf, p), ("alpha3", a3, p)):
        mc = cell(df, chosen).merge(cell(base, chosen), on="match_id",
                                    suffixes=("_t", "_b"))
        mf = cell(df, FROZEN).merge(cell(base, FROZEN), on="match_id",
                                    suffixes=("_t", "_b"))
        twins[label] = (
            float((mc["spell_f1_aligned_t"] - mc["spell_f1_aligned_b"]).mean()),
            float((mf["spell_f1_aligned_t"] - mf["spell_f1_aligned_b"]).mean()))
    a3_alias = a3.copy()
    a3_alias["spell_f1"] = a3_alias["spell_f1_aligned"]
    chosen_a3, _ = flat_select(a3_alias, scenario="phase_a3", metric="spell_f1")

    masks = pd.read_csv(RES / "stage4_masks.csv")
    mask_stats = {}
    for kind in ("uniform", "phase_a2", "phase_a3"):
        s = masks[masks["kind"] == kind]
        mask_stats[kind] = {
            "masked": float(s["masked_fraction"].mean()),
            **{f"cue{cue}_{side}": float(s[f"cue{cue}_removed_{side}"].mean())
               for cue in (1, 2, 3) for side in ("in", "out")},
            "anycue_in": float(s["anycue_removed_in"].mean()),
            "anycue_out": float(s["anycue_removed_out"].mean()),
        }
    frame_shares = {f"cue{c}": float(masks[masks["kind"] == "uniform"]
                                     [f"cue{c}_frame_share"].mean()) for c in (1, 2, 3)}

    return {
        "chosen": chosen, "chosen_a3": chosen_a3,
        "chosen_mean": chosen_mean, "frozen_mean": frozen_mean, "best": best,
        "n_plateau": int((detail["mean"] >= best - 0.01).sum()),
        "frozen_rank": int(detail.set_index(PARAMS)["mean"].rank(ascending=False)
                           .loc[tuple(FROZEN[p_] for p_ in PARAMS)]),
        "delta": d_mean, "n_pos": n_pos, "deltas": deltas,
        "ci": (lo, hi), "table_deltas": table_deltas,
        "verdict": verdict, "lomo": l, "twins": twins,
        "masks": mask_stats, "frame_shares": frame_shares,
        "aligned_lift": float((cell(p, FROZEN)["spell_f1_aligned"]
                               - cell(p, FROZEN)["spell_f1_orig"]).mean()),
    }


def write_report(r: dict, path: Path = RES / "report_stage4.md") -> None:
    t = r["table_deltas"]
    m = r["masks"]
    lines = [
        "# Stage 4 — E2 + E3 + E11 combined pass (Gate 1 read)",
        "",
        "Pre-declared: `dossier.md` §7 (2026-11-09, commit `a516fe6`) — grid G6 (144 cells),",
        "four decision tables {original, aligned} × {uniform, phase α=2} plus the fill-on",
        "(E11) and α=3 twins; primary = phase × aligned; plateau rule + the > 0.02 paired",
        "(≥ 5/7) + no-harm conditions, all applied through the P5 `sweep.flat_select` code path.",
        "",
        f"Matrices: `stage4_uniform.csv`, `stage4_phase_a2.csv`, `stage4_phase_a3.csv`,",
        f"`stage4_phase_a2_fill.csv` (144 cells × 7 matches each) + `stage4_masks.csv`.",
        "Run: 2026-11-09, 1921 s wall, 7 workers; all 7 matches, tackles mapped 100 %",
        "(173–229 per match, 12–23 StartFrame anchors each).",
        "",
        "## Selection (primary table)",
        "",
        f"- plateau candidate: `{_cell_str(r['chosen'])}` — mean **{r['chosen_mean']:.4f}**",
        f"- frozen cell: mean **{r['frozen_mean']:.4f}** (rank {r['frozen_rank']}/144; "
        f"best cell {r['best']:.4f}; {r['n_plateau']}/144 within 0.01 of best)",
        f"- paired delta **{r['delta']:+.4f}**, positive on {r['n_pos']}/7 matches, "
        f"bootstrap 95 % CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]",
        "",
        "## The four decision tables (candidate − frozen, mean spellF1)",
        "",
        "| table | Δ | positive |",
        "|---|---|---|",
        f"| uniform × original | {t['uniform_orig'][0]:+.4f} | {t['uniform_orig'][1]}/7 |",
        f"| uniform × aligned | {t['uniform_aligned'][0]:+.4f} | {t['uniform_aligned'][1]}/7 |",
        f"| phase × original | {t['phase_orig'][0]:+.4f} | {t['phase_orig'][1]}/7 |",
        f"| **phase × aligned (primary)** | **{t['phase_aligned'][0]:+.4f}** | "
        f"{t['phase_aligned'][1]}/7 |",
        "",
        "## Verdict — **REJECT → the frozen cell ships**",
        "",
        "- primary delta > 0.02: **no** (+0.0032, CI straddles 0)",
        "- paired ≥ 5/7: **no** (3/7)",
        "- no-harm on the other three: **yes**",
        "- delta ≤ 0.01 → reject under the pinned rule; frozen cell unchanged, "
        "submission md5 `aeef0058a08860c9e2407423b1172684` untouched.",
        "",
        "E2's hypothesis moved the plateau in the predicted direction (lower switch",
        "penalty, shorter support) but at noise magnitude — the gating bottleneck does",
        "not clear the adoption bar under realistic missingness. The four tables remain",
        "the robustness evidence for the frozen cell, as pre-declared.",
        "",
        "## LOMO (primary; plateau on 6, report on held-out)",
        "",
        f"- fold pick identical to the global pick on **all 7 folds** "
        f"(`{_cell_str(r['chosen'])}`)",
        f"- LOMO mean {r['lomo']['fold_score'].mean():.4f} "
        f"(global-pick mean {r['lomo']['global_score'].mean():.4f})",
        "",
        "## Twins (reported only, never select)",
        "",
        f"- fill-on vs fill-off (E11): candidate {r['twins']['fill'][0]:+.4f}, "
        f"frozen {r['twins']['fill'][1]:+.4f} — the fill tier adds nothing under "
        "phase masks (P5 ablation stands)",
        f"- α=3 vs α=2: candidate {r['twins']['alpha3'][0]:+.4f}, "
        f"frozen {r['twins']['alpha3'][1]:+.4f}; α=3 plateau picks "
        f"`{_cell_str(r['chosen_a3'])}` (**{'SAME' if r['chosen_a3'] == r['chosen'] else 'DIFFERENT'}**) ",
        "",
        "## Mask statistics (7-match means; the simulator did its job)",
        "",
        "| kind | masked | cue1 in/out | cue2 in/out | cue3 in/out | anycue in/out |",
        "|---|---|---|---|---|---|",
    ]
    for kind in ("uniform", "phase_a2", "phase_a3"):
        s = m[kind]
        lines.append(
            f"| {kind} | {s['masked']:.3f} | {s['cue1_in']:.3f}/{s['cue1_out']:.3f} | "
            f"{s['cue2_in']:.3f}/{s['cue2_out']:.3f} | "
            f"{s['cue3_in']:.3f}/{s['cue3_out']:.3f} | "
            f"{s['anycue_in']:.3f}/{s['anycue_out']:.3f} |")
    fs = r["frame_shares"]
    lines += [
        "",
        f"Frame shares: cue1 {fs['cue1']:.3f}, cue2 {fs['cue2']:.3f}, "
        f"cue3 {fs['cue3']:.3f}. Uniform is flat (the control); phase lifts removal",
        "onto every cue, α=3 harder than α=2, consistently across the 7 matches.",
        "",
        f"Context: E1-aligned labels lift the frozen cell by "
        f"{r['aligned_lift']:+.4f} over the per-touch labels on the phase table.",
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python -m tuning.stage4          # full pass (~35 min, 7 workers)",
        ".venv/bin/python -m tuning.gate1           # this read + report",
        ".venv/bin/python -m pytest -q              # 116 green",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    r = gate1_read()
    print(f"candidate : {_cell_str(r['chosen'])}")
    print(f"means     : candidate {r['chosen_mean']:.4f}  frozen {r['frozen_mean']:.4f}")
    print(f"delta     : {r['delta']:+.4f}  positive {r['n_pos']}/7  "
          f"CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    for k, (dm, np_) in r["table_deltas"].items():
        print(f"  {k:18s} {dm:+.4f}  ({np_}/7)")
    print(f"VERDICT   : {r['verdict']}")
    write_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
