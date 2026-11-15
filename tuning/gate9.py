"""Gate 9 — apply the dossier §7 P9 rule (steps C, K, T, S, then the bundle) to stage 9.

    python -m tuning.gate9                  # prints the read, writes report_stage9.md
    python -m tuning.gate9 --hold K,T       # re-read with held components OFF (parity holds)

Read-only over `stage9_*.csv`. It never writes config: an adoption is recorded in the
dossier and in `config.py` by hand. `--hold` takes any of K, T, slope — the components a
failed target parity check (tuning.kin_parity) holds for the operator; the same matrices
are re-read with them at their OFF value, no new compute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.stage5 import KINDS  # noqa: E402
from tuning.stage9 import DELAYS, P_PARAMS, RECORD, REFERENCE, SLOPES, WEIGHTS  # noqa: E402

RES = ROOT / "tuning" / "results"
PRIMARY = ("phase_a2", "aligned")
TABLES = [(k, lab) for k in KINDS for lab in ("orig", "aligned")]
STEP_BAR, BUNDLE_BAR, MIN_WINS, HARM, TIE, PIN_TOL = 0.005, 0.02, 5, -0.02, 0.005, 1e-9
S8_SHIPPED = {"gate_mode": "move", "gap_switch_frac": 0.0, "switch_delay_s": 0.64,
              "d_min_dwell_s": 0.0}
STEPS = ("C", "K", "T", "S")
NAMES = {"C": "C — coalescing min-dwell", "K": "K — kinematic dead-ball term (weight × gate tier)",
         "T": "T — kinematic team term (weight × gate tier)", "S": "S — team-switch delay"}


def canon(cell: dict) -> dict:
    """The gate tier is immaterial at zero weights; the matrices store it as False."""
    out = dict(cell)
    if float(out["kin_w_dead"]) == 0.0 and float(out["kin_w_team"]) == 0.0:
        out["gate_kin_support"] = False
    return out


def key(cell: dict) -> tuple:
    c = canon(cell)
    return (bool(c["dwell_coalesce"]), float(c["kin_w_dead"]), float(c["kin_w_team"]),
            bool(c["gate_kin_support"]), float(c["switch_delay_s"]),
            float(c["switch_delay_slope"]))


def load_tables(res: Path = RES) -> dict[str, pd.DataFrame]:
    tables = {}
    for kind in KINDS:
        df = pd.read_csv(res / f"stage9_{kind}.csv")
        n_cells = df.groupby(P_PARAMS).ngroups
        if len(df) != n_cells * 7:
            raise ValueError(f"{kind}: {len(df)} rows for {n_cells} cells x 7 matches")
        tables[kind] = df.set_index(P_PARAMS + ["match_id"]).sort_index()
    return tables


def cell_col(df: pd.DataFrame, cell: dict, col: str) -> pd.Series:
    return df.xs(key(cell), level=P_PARAMS)[col]


def deltas(tables: dict, a: dict, b: dict) -> dict:
    """{(kind, label set): (mean paired spellF1 difference a − b, matches positive)}."""
    out = {}
    for kind, lab in TABLES:
        d = (cell_col(tables[kind], a, f"spell_f1_{lab}")
             - cell_col(tables[kind], b, f"spell_f1_{lab}"))
        out[(kind, lab)] = (float(d.mean()), int((d > 0).sum()))
    return out


def harness_pin(tables: dict, res: Path = RES) -> float:
    """Largest |reference − stage-8 shipped cell| over both spellF1 columns and all rows."""
    worst = 0.0
    for kind in KINDS:
        old = pd.read_csv(res / f"stage8_{kind}.csv")
        for p, v in S8_SHIPPED.items():
            old = old[old[p] == v]
        old = old.set_index("match_id")
        if len(old) != 7:
            raise ValueError(f"{kind}: stage-8 shipped cell has {len(old)} rows")
        for lab in ("orig", "aligned"):
            new = cell_col(tables[kind], REFERENCE, f"spell_f1_{lab}")
            worst = max(worst, float((new - old[f"spell_f1_{lab}"].reindex(new.index)).abs().max()))
    return worst


def candidates(step: str, start: dict, hold: frozenset) -> list[dict]:
    """A step's cells in 'smallest change first' order (the pinned tie-break)."""
    if step == "C":
        return [{**start, "dwell_coalesce": True}]
    if step in ("K", "T"):
        if step in hold:
            return []
        param = "kin_w_dead" if step == "K" else "kin_w_team"
        return [{**start, param: w, "gate_kin_support": tier}
                for tier in (False, True) for w in WEIGHTS[1:]]
    slopes = [0.0] if "slope" in hold else SLOPES
    cells = sorted(((slope > 0, delay, slope) for delay in DELAYS for slope in slopes))
    return [{**start, "switch_delay_s": d, "switch_delay_slope": s} for _, d, s in cells
            if (d, s) != (start["switch_delay_s"], start["switch_delay_slope"])]


def run_step(tables: dict, start: dict, cands: list[dict]) -> dict:
    """One pinned step: maximin candidate (ties -> list order = smallest change), keep-test."""
    if not cands:
        return {"rows": [], "candidate": None, "keep": False, "cell": start, "held": True}
    rows = []
    for cand in cands:
        d = deltas(tables, cand, start)
        rows.append({"cell": cand, "deltas": d,
                     "g": min(d[(k, "aligned")][0] for k in KINDS)})
    best = max(r["g"] for r in rows)
    cand = next(r for r in rows if r["g"] >= best - TIE)
    primary, wins = cand["deltas"][PRIMARY]
    harm_ok = all(v[0] >= HARM for t, v in cand["deltas"].items() if t != PRIMARY)
    keep = bool(primary > STEP_BAR and wins >= MIN_WINS and harm_ok)
    return {"rows": rows, "candidate": cand, "best_g": best, "primary": primary,
            "wins": wins, "harm_ok": harm_ok, "keep": keep, "held": False,
            "cell": canon(cand["cell"]) if keep else start}


def apply_rule(tables: dict, hold: frozenset = frozenset()) -> dict:
    ref = dict(REFERENCE)
    out: dict = {}
    cell = ref
    for step in STEPS:
        out[step] = run_step(tables, cell, candidates(step, cell, hold))
        out[step]["start"] = cell
        cell = out[step]["cell"]
    out["bundle"] = cell
    if key(cell) == key(ref):
        out.update(bundle_deltas=None, adopt=False)
        return out
    d = deltas(tables, cell, ref)
    primary, wins = d[PRIMARY]
    harm_ok = all(v[0] >= HARM for t, v in d.items() if t != PRIMARY)
    out.update(bundle_deltas=d, bundle_primary=primary, bundle_wins=wins,
               bundle_harm_ok=harm_ok,
               adopt=bool(primary > BUNDLE_BAR and wins >= MIN_WINS and harm_ok))
    return out


_LABELS = {"dwell_coalesce": lambda v: "coalesce on" if v else "coalesce off",
           "kin_w_dead": lambda v: f"K {v:g}", "kin_w_team": lambda v: f"T {v:g}",
           "gate_kin_support": lambda v: "tier kin" if v else "tier ball",
           "switch_delay_s": lambda v: f"delay {v:g} s",
           "switch_delay_slope": lambda v: f"slope {v:g}"}


def cell_str(cell: dict, start: dict | None = None) -> str:
    """The cell in words; with `start`, only the parameters that differ from it."""
    parts = [_LABELS[p](cell[p]) for p in P_PARAMS if start is None or cell[p] != start[p]]
    return " · ".join(parts) if parts else "reference"


def step_alone(rule: dict, step: str) -> dict:
    """The reference with only this step's kept values (the parity reference of the hold)."""
    kept = rule[step]["cell"]
    params = {"C": ("dwell_coalesce",), "K": ("kin_w_dead", "gate_kin_support"),
              "T": ("kin_w_team", "gate_kin_support"),
              "S": ("switch_delay_s", "switch_delay_slope")}[step]
    cell = {**REFERENCE, **{p: kept[p] for p in params}}
    if step == "K" and not rule["K"]["keep"]:
        cell["gate_kin_support"] = False
    return canon(cell)


def _header(first: str = "cell") -> list[str]:
    cols = [f"{k} × {lab}" for k, lab in TABLES]
    return [f"| {first} | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]


def _row(name: str, d: dict) -> str:
    return f"| {name} | " + " | ".join(f"{d[t][0]:+.4f} ({d[t][1]}/7)" for t in TABLES) + " |"


def _mean(tables: dict, cell: dict, col: str) -> list[float]:
    return [float(cell_col(tables[k], cell, col).mean()) for k in KINDS]


def build_report(res: Path = RES, hold: frozenset = frozenset()) -> tuple[str, dict]:
    tables = load_tables(res)
    pin = harness_pin(tables, res)
    if pin > PIN_TOL:
        raise ValueError(f"harness pin failed: reference drifts {pin:.2e} from stage 8 — pass void")
    rule = apply_rule(tables, hold)
    ref = dict(REFERENCE)
    record = pd.read_csv(res / "stage9_record.csv")
    diag = pd.read_csv(res / "stage9_diag.csv")

    L = ["# Stage 9 — P9 pass (C coalescing min-dwell, K / T kinematic terms, S switch delay)", "",
         "Pre-declared: `dossier.md` §7 P9 (2026-11-14). Shipped configuration as reference, four",
         "mask kinds × two label sets, 7 IDSSE matches; kinematic models fitted leave-one-match-out,",
         "kinematic features of the test match from dropout-degraded player tracks. Δ = 7-match",
         "mean of paired spellF1 differences; (n/7) = matches where the difference is positive.", "",
         f"Harness pin: the reference cell reproduces the stage-8 shipped rows, max |Δ| = {pin:.1e}.",
         ""]
    if hold:
        L += [f"**Held components (target parity check failed, OFF in this read): "
              f"{', '.join(sorted(hold))}.**", ""]
    for step in STEPS:
        r = rule[step]
        start = r["start"]
        L += [f"## {NAMES[step]} (against: {cell_str(start)})", ""]
        if r["held"]:
            L += ["Held — the step keeps its current value.", ""]
            continue
        L += _header()
        for row in r["rows"]:
            L.append(_row(cell_str(row["cell"], start), row["deltas"]))
        c = r["candidate"]
        L += ["", f"- candidate (maximin over the four mask kinds, aligned labels): "
              f"**{cell_str(c['cell'], start)}** — g = {c['g']:+.4f} (best {r['best_g']:+.4f})",
              f"- primary (phase_a2 × aligned): **{r['primary']:+.4f}**, positive {r['wins']}/7 "
              f"(needs > {STEP_BAR} and ≥ {MIN_WINS}/7) · no-harm on the other seven tables: "
              f"**{'yes' if r['harm_ok'] else 'no'}**",
              f"- **Step verdict: {'KEEP' if r['keep'] else 'DROP — the step keeps its current value'}**",
              ""]

    L += ["## Bundle (the cell that would ship) against the reference", ""]
    if rule["bundle_deltas"] is None:
        L += ["No step was kept — nothing to adopt.", ""]
    else:
        L += _header() + [_row(cell_str(rule["bundle"]), rule["bundle_deltas"]), "",
              f"- primary: **{rule['bundle_primary']:+.4f}**, positive {rule['bundle_wins']}/7 "
              f"(needs > {BUNDLE_BAR} and ≥ {MIN_WINS}/7) · no-harm: "
              f"**{'yes' if rule['bundle_harm_ok'] else 'no'}**",
              f"- **Verdict: {'ADOPT' if rule['adopt'] else 'REJECT — nothing is adopted'}**", ""]

    L += ["## Levels, predicted dead share and external firing (7-match means)", "",
          "| cell | " + " | ".join(f"{k} spellF1 aligned" for k in KINDS) + " | "
          + " | ".join(f"{k} D share" for k in KINDS) + " | "
          + " | ".join(f"{k} changed" for k in KINDS) + " |",
          "|---|" + "---|" * (3 * len(KINDS))]
    shown = [("reference", ref)]
    for step in STEPS:
        if rule[step]["keep"]:
            shown.append((f"after {step}", rule[step]["cell"]))
    for name, cell in shown:
        L.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in _mean(tables, cell, "spell_f1_aligned"))
                 + " | " + " | ".join(f"{v:.3f}" for v in _mean(tables, cell, "pred_d_share"))
                 + " | " + " | ".join(f"{v:.1%}" for v in _mean(tables, cell, "changed_share")) + " |")
    L += ["", "Share of in-match frames changed by a kept step on its own, smallest … largest "
          "per-match value over the four mask kinds (the parity reference of the pinned hold):", ""]
    firing = {}
    for step in STEPS:
        if rule[step]["keep"]:
            cell = step_alone(rule, step)
            vals = [cell_col(tables[k], cell, "changed_share") for k in KINDS]
            firing[step] = max(float(v.max()) for v in vals)
            lo = min(float(v.min()) for v in vals)
            L.append(f"- {step} alone ({cell_str(cell)}): {lo:.1%} … {firing[step]:.1%}")
    if rule["bundle_deltas"] is not None:
        vals = [cell_col(tables[k], rule["bundle"], "changed_share") for k in KINDS]
        firing["bundle"] = max(float(v.max()) for v in vals)
        L.append(f"- bundle: {min(float(v.min()) for v in vals):.1%} … {firing['bundle']:.1%}")

    base = rule["T"]["cell"]
    L += ["", "## Team-switch latency (aligned labels, predicted − true, frames; C / K / T as kept)",
          "", "| delay · slope | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for slope in SLOPES:
        for d in DELAYS:
            cell = {**base, "switch_delay_s": d, "switch_delay_slope": slope}
            L.append(f"| {d:g} s · {slope:g} | "
                     + " | ".join(f"{v:+.1f}" for v in _mean(tables, cell, "lat_mean")) + " |")

    L += ["", "## Team-swap equivariance (share of in-match frames not mirrored; 0 = symmetric)", "",
          "| cell | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for col, name in (("equiv_mismatch_ref", "reference"),
                      (f"equiv_mismatch_T{WEIGHTS[-1]:g}_tier0", f"T {WEIGHTS[-1]:g} · tier ball"),
                      (f"equiv_mismatch_T{WEIGHTS[-1]:g}_tier1", f"T {WEIGHTS[-1]:g} · tier kin")):
        L.append(f"| {name} | "
                 + " | ".join(f"{diag[diag['kind'] == k][col].mean():.2%}" for k in KINDS) + " |")

    L += ["", "## Record-only cell (never adopted in P9)", "",
          "The IDSSE ball is frozen while dead, so on ball-visible frames the ball-at-rest feature "
          "already reads the dead-ball label; this row cannot judge a kinematic term there.", ""]
    L += _header()
    for name in RECORD:
        d = {}
        for kind, lab in TABLES:
            r = record[(record["kind"] == kind) & (record["cell"] == name)].set_index("match_id")
            diff = r[f"spell_f1_{lab}"] - cell_col(tables[kind], ref, f"spell_f1_{lab}")
            d[(kind, lab)] = (float(diff.mean()), int((diff > 0).sum()))
        L.append(_row(name, d))

    q = diag[diag["kind"] == KINDS[0]]
    L += ["", "## IDSSE side of the target parity checks (per-match range)", "",
          f"- Q1 median per-frame median player speed: {q['speed_med_p50'].min():.2f} … "
          f"{q['speed_med_p50'].max():.2f} m/s",
          f"- Q4 team-logit agreement with the ball-based decoder on ball-visible frames "
          f"(no mask, leave-one-out model, clean tracks): {q['q4_team_agreement'].min():.3f} … "
          f"{q['q4_team_agreement'].max():.3f}",
          f"- Q5 mean visible slots: {q['n_vis_mean'].min():.1f} … {q['n_vis_mean'].max():.1f} "
          f"(degraded for the kinematic features: {q['n_vis_mean_degraded'].min():.1f} … "
          f"{q['n_vis_mean_degraded'].max():.1f})",
          f"- Q6 median ball speed on alive frames: {q['q6_ball_speed_p50'].min():.2f} … "
          f"{q['q6_ball_speed_p50'].max():.2f} m/s",
          "", "## Reproduce", "", "```bash",
          ".venv/bin/python -m tuning.stage9          # full pass (~25 min, 7 workers)",
          ".venv/bin/python -m tuning.gate9           # this read + report",
          ".venv/bin/python -m tuning.kin_fit         # shipped model -> src/kin_model.json",
          ".venv/bin/python -m tuning.kin_parity      # target parity checks, PASS / FAIL only",
          "```", ""]
    rule["firing"] = firing
    rule["pin"] = pin
    return "\n".join(L), rule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hold", default="", help="comma list of K, T, slope")
    args = parser.parse_args()
    hold = frozenset(h.strip() for h in args.hold.split(",") if h.strip())
    if not hold <= {"K", "T", "slope"}:
        parser.error("--hold takes K, T and/or slope")
    text, rule = build_report(hold=hold)
    (RES / "report_stage9.md").write_text(text)
    rows = []
    for step in STEPS:
        r = rule[step]
        rows.append({"gate": step,
                     "verdict": "HELD" if r["held"] else ("KEEP" if r["keep"] else "DROP"),
                     "cell": "" if r["held"] else cell_str(r["candidate"]["cell"], r["start"]),
                     "gain": "" if r["held"] else round(r["primary"], 4)})
    rows.append({"gate": "bundle", "verdict": "ADOPT" if rule["adopt"] else "REJECT",
                 "cell": cell_str(rule["bundle"], dict(REFERENCE)),
                 "gain": round(rule.get("bundle_primary", 0.0), 4)})
    pd.DataFrame(rows).to_csv(RES / "stage9_verdicts.csv", index=False)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
