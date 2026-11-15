"""Gate 8 — apply the dossier §7 P8 rule (steps G, S, M, then the bundle) to stage 8.

    python -m tuning.gate8        # prints the read, writes tuning/results/report_stage8.md

Read-only over `stage8_*.csv`. It never writes config: an adoption is recorded in the
dossier and in `config.py` by hand.
"""

from __future__ import annotations  # postponed annotation evaluation

import sys  # sys.path bootstrapping below
from pathlib import Path  # filesystem paths

import pandas as pd  # committed matrix tables

ROOT = Path(__file__).resolve().parents[1]  # repo root (above tuning/)
if str(ROOT) not in sys.path:  # make the repo importable
    sys.path.insert(0, str(ROOT))  # prepend repo root

# stage-5 kinds and stage-8 grid constants, reused verbatim
from tuning.stage5 import KINDS  # noqa: E402
from tuning.stage8 import D_MINS, DELAYS, G_CELLS, P_PARAMS, RECORD, REFERENCE  # noqa: E402

RES = ROOT / "tuning" / "results"  # committed matrices live here (read-only)
PRIMARY = ("phase_a2", "aligned")  # primary table
TABLES = [(k, lab) for k in KINDS for lab in ("orig", "aligned")]  # all eight tables
STEP_BAR, BUNDLE_BAR, MIN_WINS, HARM, TIE, PIN_TOL = 0.005, 0.02, 5, -0.02, 0.005, 1e-9  # pinned thresholds
S5_REF = {"d_w_long_missing": 0.5, "d_missing_run_s": 10.0,  # stage-5 reference cell
          "d_w_low_activity": 0.0}  # (harness pin target)


def key(cell: dict) -> tuple:
    return (cell["gate_mode"], float(cell["gap_switch_frac"]),  # hashable cell identity,
            float(cell["switch_delay_s"]), float(cell["d_min_dwell_s"]))  # four gate params


def load_tables(res: Path = RES) -> dict[str, pd.DataFrame]:
    tables = {}  # collected tables
    for kind in KINDS:
        df = pd.read_csv(res / f"stage8_{kind}.csv")  # one committed matrix
        n_cells = df.groupby(P_PARAMS).ngroups  # distinct cells present
        if len(df) != n_cells * 7:  # every cell needs its 7 matches
            raise ValueError(f"{kind}: {len(df)} rows for {n_cells} cells x 7 matches")  # matrix truncated
        tables[kind] = df.set_index(P_PARAMS + ["match_id"]).sort_index()  # index for xs()
    return tables  # all four loaded


def cell_col(df: pd.DataFrame, cell: dict, col: str) -> pd.Series:
    return df.xs(key(cell), level=P_PARAMS)[col]  # one cell's column, by match


def deltas(tables: dict, a: dict, b: dict) -> dict:
    """{(kind, label set): (mean paired spellF1 difference a − b, matches positive)}."""
    out = {}  # deltas storage
    for kind, lab in TABLES:
        d = (cell_col(tables[kind], a, f"spell_f1_{lab}")
             - cell_col(tables[kind], b, f"spell_f1_{lab}"))  # per-match delta a − b
        out[(kind, lab)] = (float(d.mean()), int((d > 0).sum()))  # mean, wins
    return out  # all eight deltas


def harness_pin(tables: dict, res: Path = RES) -> float:
    """Largest |reference − stage-5 reference| over both spellF1 columns and all rows."""
    worst = 0.0  # running worst drift
    for kind in KINDS:
        old = pd.read_csv(res / f"stage5_{kind}.csv")  # committed stage-5 matrix
        for p, v in S5_REF.items():  # reduce to the stage-5 reference rows
            old = old[old[p] == v]  # keep reference rows
        old = old.set_index("match_id")  # index for reindexing
        for lab in ("orig", "aligned"):
            new = cell_col(tables[kind], REFERENCE, f"spell_f1_{lab}")  # same cell, new bed
            worst = max(worst, float((new - old[f"spell_f1_{lab}"].reindex(new.index)).abs().max()))  # largest so far
    return worst  # largest drift seen


def run_step(tables: dict, start: dict, candidates: list[dict]) -> dict:
    """One pinned step: maximin candidate (ties -> list order = smallest change), keep-test."""
    rows = []  # candidate rows
    for cand in candidates:
        d = deltas(tables, cand, start)  # candidate vs current start cell
        rows.append({"cell": cand, "deltas": d,
                     "g": min(d[(k, "aligned")][0] for k in KINDS)})  # maximin over kinds
    best = max(r["g"] for r in rows)  # best maximin gain
    cand = next(r for r in rows if r["g"] >= best - TIE)      # candidates are in change order
    primary, wins = cand["deltas"][PRIMARY]  # primary-table gain
    harm_ok = all(v[0] >= HARM for t, v in cand["deltas"].items() if t != PRIMARY)  # no-harm
    keep = bool(primary > STEP_BAR and wins >= MIN_WINS and harm_ok)  # pinned keep-test
    return {"rows": rows, "candidate": cand, "best_g": best, "primary": primary,
            "wins": wins, "harm_ok": harm_ok, "keep": keep,
            "cell": cand["cell"] if keep else start}  # kept cell, else unchanged


def apply_rule(tables: dict) -> dict:
    ref = dict(REFERENCE)  # start from the stage-5 reference
    g = run_step(tables, ref, [{**ref, "gate_mode": m, "gap_switch_frac": f}  # step G:
                               for m, f in G_CELLS[1:]])  # gate mode + gap placement
    s = run_step(tables, g["cell"], [{**g["cell"], "switch_delay_s": d} for d in DELAYS[1:]])  # step S
    m = run_step(tables, s["cell"], [{**s["cell"], "d_min_dwell_s": d} for d in D_MINS[1:]])  # step M
    bundle = m["cell"]  # the cell that would ship after all three steps
    out = {"G": g, "S": s, "M": m, "bundle": bundle}  # collect step results
    if key(bundle) == key(ref):  # nothing kept → nothing to adopt
        out.update(bundle_deltas=None, adopt=False)  # nothing to evaluate
        return out  # early exit
    d = deltas(tables, bundle, ref)  # bundle vs reference, all tables
    primary, wins = d[PRIMARY]  # primary-table values
    harm_ok = all(v[0] >= HARM for t, v in d.items() if t != PRIMARY)  # no-harm
    out.update(bundle_deltas=d, bundle_primary=primary, bundle_wins=wins,
               bundle_harm_ok=harm_ok,
               adopt=bool(primary > BUNDLE_BAR and wins >= MIN_WINS and harm_ok))  # stricter bar
    return out  # rule results


_LABELS = {"gate_mode": "gate {}", "gap_switch_frac": "gap placement {:g}",  # report labels
           "switch_delay_s": "delay {:g} s", "d_min_dwell_s": "D minimum {:g} s"}


def cell_str(cell: dict, start: dict | None = None) -> str:
    """The cell in words; with `start`, only the parameters that differ from it."""
    parts = [_LABELS[p].format(cell[p]) for p in P_PARAMS
             if start is None or cell[p] != start[p]]  # only params that differ
    return " · ".join(parts) if parts else "reference"  # "reference" if identical


def _header(first: str = "cell") -> list[str]:
    cols = [f"{k} × {lab}" for k, lab in TABLES]  # one column per table
    return [f"| {first} | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]  # header + separator


def _row(name: str, d: dict) -> str:
    return f"| {name} | " + " | ".join(f"{d[t][0]:+.4f} ({d[t][1]}/7)" for t in TABLES) + " |"  # Δ row


def _mean(tables: dict, cell: dict, col: str) -> list[float]:
    return [float(cell_col(tables[k], cell, col).mean()) for k in KINDS]  # 7-match mean each


def build_report(res: Path = RES) -> tuple[str, dict]:
    tables = load_tables(res)  # load the matrices
    pin = harness_pin(tables, res)  # reference must reproduce stage 5
    if pin > PIN_TOL:  # drift voids the whole pass
        raise ValueError(f"harness pin failed: reference drifts {pin:.2e} from stage 5 — pass void")  # pin failed
    rule = apply_rule(tables)  # pinned G → S → M → bundle
    ref = dict(REFERENCE)  # local reference copy
    record = pd.read_csv(res / "stage8_record.csv")  # record-only cells (I2)
    diag = pd.read_csv(res / "stage8_diag.csv")  # equivariance + I4 diagnostics

    L = ["# Stage 8 — P8 pass (G switch gate, S team-switch delay, M dead-ball minimum)", "",  # header
         "Pre-declared: `dossier.md` §7 P8 (2026-11-14). Stage-5 reference, four mask kinds ×",
         "two label sets, 7 IDSSE matches. Δ = 7-match mean of paired spellF1 differences;",
         "(n/7) = matches where the difference is positive.", "",
         f"Harness pin: the reference cell reproduces the stage-5 reference rows, max |Δ| = {pin:.1e}.",
         ""]
    names = {"G": "G — switch gate", "S": "S — team-switch delay", "M": "M — dead-ball minimum"}  # step titles
    start = ref  # first step compares to reference
    for step in ("G", "S", "M"):  # one report section per pinned step
        r = rule[step]  # this step's result
        c = r["candidate"]  # shorthand for candidate
        L += [f"## {names[step]} (against: {cell_str(start)})", ""] + _header()
        for row in r["rows"]:  # every candidate cell tried at this step
            L.append(_row(cell_str(row["cell"], start), row["deltas"]))  # one candidate row
        L += ["", f"- candidate (maximin over the four mask kinds, aligned labels): "
              f"**{cell_str(c['cell'], start)}** — g = {c['g']:+.4f} (best {r['best_g']:+.4f})",
              f"- primary (phase_a2 × aligned): **{r['primary']:+.4f}**, positive {r['wins']}/7 "
              f"(needs > {STEP_BAR} and ≥ {MIN_WINS}/7) · no-harm on the other seven tables: "
              f"**{'yes' if r['harm_ok'] else 'no'}**",  # pinned conditions
              f"- **Step verdict: {'KEEP' if r['keep'] else 'DROP — the step keeps its current value'}**",
              ""]
        start = r["cell"]  # next step starts where this one ended

    L += ["## Bundle (the cell that would ship) against the reference", ""]  # bundle section
    if rule["bundle_deltas"] is None:  # no step kept
        L += ["No step was kept — nothing to adopt.", ""]  # nothing to evaluate
    else:
        L += _header() + [_row(cell_str(rule["bundle"]), rule["bundle_deltas"]), "",  # bundle row
              f"- primary: **{rule['bundle_primary']:+.4f}**, positive {rule['bundle_wins']}/7 "
              f"(needs > {BUNDLE_BAR} and ≥ {MIN_WINS}/7) · no-harm: "
              f"**{'yes' if rule['bundle_harm_ok'] else 'no'}**",
              f"- **Verdict: {'ADOPT' if rule['adopt'] else 'REJECT — nothing is adopted'}**", ""]

    L += ["## Levels and external firing (7-match means)", "",  # levels section
          "| cell | " + " | ".join(f"{k} spellF1 aligned" for k in KINDS) + " | "
          + " | ".join(f"{k} changed" for k in KINDS) + " |",
          "|---|" + "---|" * (2 * len(KINDS))]
    shown = [("reference", ref)]  # start with reference
    for step in ("G", "S", "M"):  # add the cell after each kept step
        if rule[step]["keep"]:  # only kept steps
            shown.append((f"after {step}", rule[step]["cell"]))  # cell after this step
    for name, cell in shown:  # one row: aligned spellF1 + changed share per kind
        L.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in _mean(tables, cell, "spell_f1_aligned"))
                 + " | " + " | ".join(f"{v:.1%}" for v in _mean(tables, cell, "changed_share")) + " |")  # one levels row
    L += ["", "Largest per-match share of in-match frames changed by a kept step on its own "
          "(the parity reference of the pinned hold):", ""]  # firing intro line
    step_cells = {"G": {**ref, **{p: rule["G"]["cell"][p] for p in ("gate_mode", "gap_switch_frac")}},
                  "S": {**ref, "switch_delay_s": rule["S"]["cell"]["switch_delay_s"]},
                  "M": {**ref, "d_min_dwell_s": rule["M"]["cell"]["d_min_dwell_s"]}}  # step alone
    firing = {}  # firing shares
    for step in ("G", "S", "M"):
        if rule[step]["keep"]:  # only kept steps fire
            firing[step] = max(float(cell_col(tables[k], step_cells[step], "changed_share").max())
                               for k in KINDS)  # largest per-match changed share
            lo = min(float(cell_col(tables[k], step_cells[step], "changed_share").min())
                     for k in KINDS)  # smallest changed share
            L.append(f"- {step} alone ({cell_str(step_cells[step])}): {lo:.1%} … {firing[step]:.1%}")  # firing line

    L += ["", "## Team-switch latency (aligned labels, predicted − true, frames; G as kept, D minimum off)",  # latency section
          "", "| delay | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for d in DELAYS:  # one row per delay, G as kept, M off
        cell = {**rule["G"]["cell"], "switch_delay_s": d, "d_min_dwell_s": 0.0}  # G kept, M off
        L.append(f"| {d:g} s | " + " | ".join(f"{v:+.1f}" for v in _mean(tables, cell, "lat_mean")) + " |")  # one latency row

    L += ["", "## Team-swap equivariance (share of in-match frames not mirrored; 0 = symmetric)", "",  # equivariance section
          "| gate cell | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for mode, frac in G_CELLS:  # one row per gate cell
        col = f"equiv_mismatch_{mode}_{frac:g}"  # diag column for cell
        L.append(f"| {mode}, gap placement {frac:g} | "
                 + " | ".join(f"{diag[diag['kind'] == k][col].mean():.2%}" for k in KINDS) + " |")  # one equivariance row
    L += ["", "I4 — a missed stoppage inherits the team that had the ball: share of missed dead "  # I4 note
          "frames given to home vs home's share of live frames (reference, 7-match means): "
          + " · ".join(f"{k} {diag[diag['kind'] == k]['missed_dead_to_home'].mean():.3f} vs "
                       f"{diag[diag['kind'] == k]['home_share_of_live'].mean():.3f}" for k in KINDS),
          "", "## Record-only cells (operator idea I2 — never adopted in P8)", "",  # I2 section
          "The IDSSE ball is frozen while dead, so `stationary` reads the dead-ball label there; "
          "these rows show the size of that dependence, not the value of the feature on real tracking.",
          ""] + _header()
    for name in RECORD:  # record-only cell vs reference, all tables
        d = {}  # deltas storage
        for kind, lab in TABLES:
            r = record[(record["kind"] == kind) & (record["cell"] == name)].set_index("match_id")  # cell's rows
            diff = r[f"spell_f1_{lab}"] - cell_col(tables[kind], ref, f"spell_f1_{lab}")  # delta vs reference
            d[(kind, lab)] = (float(diff.mean()), int((diff > 0).sum()))  # mean, wins
        L.append(_row(name, d))  # one record-only row
    L += ["", "| cell | " + " | ".join(f"{k} predicted D share" for k in KINDS) + " |",  # D-share table
          "|---|" + "---|" * len(KINDS),
          "| reference | " + " | ".join(f"{v:.3f}" for v in _mean(tables, ref, "pred_d_share")) + " |"]
    for name in RECORD:  # one D-share row per record-only cell
        L.append(f"| {name} | " + " | ".join(
            f"{record[(record['kind'] == k) & (record['cell'] == name)]['pred_d_share'].mean():.3f}"
            for k in KINDS) + " |")  # one D-share row
    L += ["", "## Reproduce", "", "```bash",  # reproduce block
          ".venv/bin/python -m tuning.stage8          # full pass (~3 min, 6 workers)",
          ".venv/bin/python -m tuning.gate8           # this read + report", "```", ""]
    rule["firing"] = firing  # attach extras for the caller
    rule["pin"] = pin  # expose pin to caller
    return "\n".join(L), rule  # report text + results


def main() -> int:
    text, rule = build_report()  # apply the pinned rule, read-only
    (RES / "report_stage8.md").write_text(text)  # write the report
    rows = [{"gate": step, "verdict": "KEEP" if rule[step]["keep"] else "DROP",  # one row per step
             "cell": cell_str(rule[step]["candidate"]["cell"]),
             "gain": round(rule[step]["primary"], 4)} for step in ("G", "S", "M")]
    rows.append({"gate": "bundle", "verdict": "ADOPT" if rule["adopt"] else "REJECT",  # bundle row
                 "cell": cell_str(rule["bundle"]),
                 "gain": round(rule.get("bundle_primary", 0.0), 4)})
    pd.DataFrame(rows).to_csv(RES / "stage8_verdicts.csv", index=False)  # machine-readable verdicts
    print(text)  # echo the report
    return 0  # success


if __name__ == "__main__":
    sys.exit(main())  # python -m tuning.gate8 entry point
