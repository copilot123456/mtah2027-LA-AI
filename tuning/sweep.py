"""The P5 sweep: grid over decoder/emission parameters on IDSSE, scored externally.

    python -m tuning.sweep                 # full run (parses missing matches first)
    python -m tuning.sweep --smoke         # plumbing check: 1 match, 6 grid points
    python -m tuning.sweep --select-only   # re-run selection on the existing results

Matrix design: per (match, scenario) the fill/emission depend only on sigma_m, so
emission is computed once per sigma and the decoder once per cell. Stage 1 is
7 matches x 2 scenarios x 243 cells = 3,402 rows; stages 2 and 3 extend the grid on the
`none` scenario only (768 and 324 cells) and the final selection runs on their union
(930 unique cells). Selection and LOMO are derived from the same matrix (no extra compute).

Selection rule (documented, pre-declared): among configs within 0.01 mean spellF1 of
the best, pick the one with the highest WORST grid-neighbour score (plateau centre),
ties broken by mean. A grid-neighbour is any swept cell that differs in exactly one
parameter (every other swept value of that parameter, not only the adjacent ones).
Selection scenario: `kinematic` in stage 1 (the pipeline as first built); `none` in
stages 2-3, after the layer ablation dropped the fill tier.

Scoring uses the vendored official metric code (evaluation/possession_metrics.py) on
IDSSE-derived labels: GREEN external output.
"""

from __future__ import annotations  # postponed annotation evaluation

# stdlib: CLI flags, grid product, sys.path bootstrap, timing, paths
import argparse
import itertools
import sys
import time
from pathlib import Path

# numeric arrays and the results table
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root
# repo root and vendored scorer dir go on sys.path
for p in (str(ROOT), str(ROOT / "evaluation")):
    if p not in sys.path:  # skip if already present
        sys.path.insert(0, p)  # prepend for import priority

# official metric code, vendored for pinning
import possession_metrics as pm  # noqa: E402
# decoder default configuration
from config import CFG  # noqa: E402
# IDSSE: the 7-match tuning bed and its loader
from tuning.idsse import MATCH_IDS, load_match  # noqa: E402
# missing-run simulation: empirical lengths and ball masking
from tuning.simulate import empirical_run_lengths, mask_ball  # noqa: E402

# stage-1 grid: 3^5 = 243 cells
GRID = {
    "switch_pen": [4.0, 8.0, 12.0],
    "dead_pen": [3.0, 5.0, 8.0],
    "sigma_m": [3.0, 4.5, 6.0],
    "min_dwell_s": [0.5, 1.0, 2.0],
    "switch_support_s": [0.3, 0.6, 1.0],
}
# Stage 2 (pre-justified by stage-1 marginals: switch/sigma improved toward the low
# edge, dwell/support toward the high edge, dead flat): one extra notch per direction.
GRID_EXT = {
    "switch_pen": [2.0, 4.0, 8.0, 12.0],
    "dead_pen": [3.0, 5.0, 8.0],
    "sigma_m": [2.0, 3.0, 4.5, 6.0],
    "min_dwell_s": [0.5, 1.0, 2.0, 3.0],
    "switch_support_s": [0.3, 0.6, 1.0, 1.5],
}
# Stage 3 (final, pre-justified by stage-2 marginals: only sigma still climbed at its
# edge (+0.017/notch, extend to the physical floor of ~1 m); switch/dwell/support shown
# flat or interior; dead kept flat for neighbour continuity).
GRID_S3 = {
    "switch_pen": [2.0, 4.0, 8.0],
    "dead_pen": [3.0, 5.0, 8.0],
    "sigma_m": [1.0, 1.5, 2.0, 3.0],
    "min_dwell_s": [1.0, 2.0, 3.0],
    "switch_support_s": [0.6, 1.0, 1.5],
}
PARAMS = list(GRID)              # swept parameter names, fixed order
SCENARIOS = ["kinematic", "none"]  # fill tier on vs off
MASK_SEED = 20261107              # fixed seed: reproducible masks
# per-stage result matrices and report files
RESULTS = ROOT / "tuning" / "results" / "sweep_results.csv"  # stage-1 matrix
RESULTS_STG2 = ROOT / "tuning" / "results" / "sweep_results_stage2.csv"  # stage-2 matrix
RESULTS_STG3 = ROOT / "tuning" / "results" / "sweep_results_stage3.csv"  # stage-3 matrix
REPORT = ROOT / "tuning" / "results" / "report.md"  # stage-1 report
REPORT_STG2 = ROOT / "tuning" / "results" / "report_stage2.md"  # stage-2 report
REPORT_FINAL = ROOT / "tuning" / "results" / "report_final.md"  # stage-3 report

_LETTERS = np.array(["X", "H", "A", "D"], dtype=object)  # index by internal code + 1


def points_of(grid: dict) -> list[dict]:
    # every value combination as one config dict
    return [dict(zip(PARAMS, combo))
            for combo in itertools.product(*(grid[p] for p in PARAMS))]


def grid_points() -> list[dict]:
    # stage-1 grid shorthand
    return points_of(GRID)


def _score(gt_codes: np.ndarray, pred_codes: np.ndarray) -> dict:
    """spellF1 / macroF1 exactly as the official evaluate() computes them.

    The scorer forward-fills predicted X inside each graded run (fill_abstentions) before
    the temporal metric and charges the raw X only in the frame-level confusion. Until
    2026-11-07 this helper skipped the fill, so a variant that writes X on graded frames
    (the P5 X-policy ablation) scored `H X X H` as two possessions; the sweep itself never
    writes X on graded frames and was unaffected. tests/test_tuning.py pins the equality.
    """
    runs = pm.valid_runs(gt_codes)  # graded GT runs
    filled = pm.fill_abstentions(pred_codes, runs)  # forward-fill X inside runs
    spell = pm.spell_scores(gt_codes, filled, runs)["mean_f1"]  # official spell F1
    cm3 = pm.scored_confusion(pm.confusion(gt_codes, pred_codes))  # raw X confusion
    return {"spell_f1": float(spell), "macro_f1": float(pm.macro_f1(cm3))}  # both pinned metrics


def run_task(args) -> list[dict]:
    """All grid cells for one (match, scenario). Runs in a worker process."""
    match_id, scenario, grid, run_lengths = args  # unpack worker task
    t0 = time.time()  # runtime clock for the report
    case = load_match(match_id)  # cached parse of this match
    rng = np.random.default_rng(MASK_SEED + sum(ord(c) for c in match_id))  # per-match seed
    masked_ball, mask_stats = mask_ball(case.ball, run_lengths, rng)  # hide the ball

    base = dict(CFG)  # copy decoder defaults
    if scenario == "none":
        base["fill_long_gap_s"] = 0.0        # tier off: masked frames stay `none`
    elif scenario != "kinematic":
        raise ValueError(scenario)  # only two scenarios exist

    # pipeline pieces, imported lazily inside the worker
    from src.ball_fill import build_ball_track
    from src.decoder import decode
    from src.emission import build_emission

    track = build_ball_track(masked_ball, case.half, base)  # fill over masked ball
    gt_codes = pm.label_codes(case.letters)  # GT as internal codes
    gt_runs = pm.valid_runs(gt_codes)  # GT graded runs
    n_gt = len(pm.possession_spells(gt_codes, gt_runs))  # GT spell count

    rows = []  # output rows for this task
    sigmas = sorted({float(c["sigma_m"]) for c in grid})  # emission cached per sigma
    for sigma in sigmas:
        # emission depends only on sigma: build once, reuse for its cells
        em = build_emission(track, case.players, case.players_s, case.slot_team,
                            {**base, "sigma_m": sigma})
        for cell in grid:
            if float(cell["sigma_m"]) != sigma:
                continue  # other sigmas handled in their own pass
            cfg = {**base, **cell}  # cell overrides on top of defaults
            decoded = decode(em, track.source, case.half, cfg)  # run the decoder
            pred = _LETTERS[np.asarray(decoded.labels, dtype=np.int8) + 1]  # codes -> letters
            pred_codes = pm.label_codes(pred)  # letters as internal codes
            scores = _score(gt_codes, pred_codes)  # official spell/macro F1
            rows.append({
                "match_id": match_id, "scenario": scenario, **cell, **scores,
                "n_gt_spells": n_gt,
                "n_pred_spells": len(pm.possession_spells(pred_codes, gt_runs)),
                "runtime_s": round(time.time() - t0, 1),  # elapsed for whole task
                "masked_fraction": round(mask_stats["masked_fraction"], 4),  # frames hidden
            })
    return rows  # all cells for this (match, scenario)


# ---------------- selection ----------------

def _mean_table(df: pd.DataFrame, metric: str = "spell_f1") -> pd.DataFrame:
    # per-config aggregate of the metric across matches
    return df.groupby(PARAMS)[metric].agg(["mean", "std", "count"]).reset_index()


def _neighbour_means(means: pd.Series) -> pd.Series:
    """For each config index: min mean over grid-neighbours (differ in exactly one param).

    The candidate values of each parameter are the ones actually present in `means`
    (the swept grid), so the rule is the same whichever stage's matrix is passed in.
    Stage 1 used the literal GRID; for the extended stages that would have ignored the
    new notches and still looked at stage-1 notches that were never swept at the new
    values. Audit 2026-11-07: on the committed stage-2+3 union both definitions pick
    the same cell with the same worst-neighbour score (0.2081), see AUDIT.md.
    """
    keys = list(means.index)  # every config in the table
    values = {param: sorted(means.index.get_level_values(param).unique())
              for param in PARAMS}  # swept values actually present
    out = {}  # worst-neighbour mean per config
    for key in keys:
        scores = []  # neighbour means gathered for this config
        for param in PARAMS:
            k = list(key) if isinstance(key, tuple) else [key]  # unpack config
            idx = PARAMS.index(param)  # slot of this parameter
            for value in values[param]:
                if value == k[idx]:
                    continue  # skip the config itself
                other = list(k)  # swap in the neighbour value
                other[idx] = value  # neighbour value written in
                other_key = tuple(other) if len(PARAMS) > 1 else other[0]  # index-key form
                if other_key in means.index:  # only swept neighbours count
                    scores.append(means.loc[other_key])  # record its mean
        out[key] = min(scores) if scores else np.nan  # worst neighbour wins
    return pd.Series(out, name="worst_neighbour")  # named result series


def flat_select(df: pd.DataFrame, scenario: str = "kinematic",
                metric: str = "spell_f1", tol: float = 0.01) -> tuple[dict, pd.DataFrame]:
    """Pre-declared plateau rule; returns (chosen_params, full_mean_table_with_details)."""
    sub = df[df["scenario"] == scenario]  # rows of the selection scenario
    means = _mean_table(sub, metric).set_index(PARAMS)[["mean", "std"]]  # indexed stats
    best = means["mean"].max()  # top mean here
    candidates = means[means["mean"] >= best - tol].copy()  # within 0.01 of best
    worst_nb = _neighbour_means(means["mean"])  # plateau-centre metric
    candidates["worst_neighbour"] = worst_nb.reindex(candidates.index)  # attach scores
    top = candidates.sort_values(["worst_neighbour", "mean"], ascending=False).iloc[0]  # best-ranked
    chosen = {p: float(v) for p, v in zip(PARAMS, top.name)}   # params live in the MultiIndex
    detail = means.copy()  # full table for the report
    detail["worst_neighbour"] = worst_nb  # scores into detail table
    detail["selected"] = False  # flag column, all False...
    detail.loc[[top.name], "selected"] = True  # ...then mark the winner
    return chosen, detail.reset_index()  # params plus flat detail


def lomo(df: pd.DataFrame, chosen: dict, metric: str = "spell_f1",
         scenario: str = "kinematic") -> pd.DataFrame:
    """Pick on 6 matches (plateau rule), report on the held-out one; vs global chosen."""
    rows = []  # one row per held-out match
    matches = sorted(df["match_id"].unique())  # every match, sorted
    for held in matches:  # leave one match out each fold
        train = df[df["match_id"] != held]  # the other 6 matches
        fold_chosen, _ = flat_select(train, metric=metric, scenario=scenario)  # pick on train
        test = df[df["match_id"] == held]  # held-out rows
        fold_pick = _cell_score(test, fold_chosen)  # fold choice scored on held-out
        global_pick = _cell_score(test, chosen)  # global choice on held-out
        rows.append({"held_out": held, "fold_pick": _cell_str(fold_chosen),  # fold result row
                     "fold_score": fold_pick, "global_score": global_pick})
    return pd.DataFrame(rows)  # fold results as a table


def _cell_score(df: pd.DataFrame, cell: dict) -> float:
    sel = df  # start with all rows
    for p, v in cell.items():
        sel = sel[sel[p] == v]  # keep rows matching this cell
    return float(sel["spell_f1"].mean())  # mean over the kept rows


def _cell_str(cell: dict) -> str:
    return "/".join(f"{k}={v:g}" for k, v in cell.items())  # compact display


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 7) -> tuple[float, float]:
    """Match-level bootstrap CI of the mean (matches are the sample unit)."""
    rng = np.random.default_rng(seed)  # reproducible resampling
    values = np.asarray(values, dtype=float)  # ensure float array
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]  # n resamples
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))  # 95% CI


def load_union() -> pd.DataFrame:
    """The stage-2 + stage-3 matrix the final selection ran on (stage 3 wins overlaps)."""
    prior = pd.read_csv(RESULTS_STG2)  # stage-2 matrix
    final = pd.read_csv(RESULTS_STG3)  # stage-3 matrix
    return (pd.concat([prior, final], ignore_index=True)
            .drop_duplicates(subset=["match_id", "scenario", *PARAMS], keep="last"))  # stage 3 wins


def smoke_grid() -> list[dict]:
    full = grid_points()  # full stage-1 grid
    return [full[0], full[40], full[80], full[120], full[160], full[-1]]  # 6 spread cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # docstring line one as help
    parser.add_argument("--smoke", action="store_true", help="1 match, 6 cells")  # quick check
    parser.add_argument("--select-only", action="store_true")  # reuse existing results
    parser.add_argument("--stage2", action="store_true",  # stage-2 flag
                        help="extended grid, none scenario only (fill tier dropped by ablation)")
    parser.add_argument("--stage3", action="store_true",  # stage-3 flag
                        help="final focused grid (sigma floor); selection runs on stage2+3 union")
    parser.add_argument("--workers", type=int, default=7)  # pool size
    args = parser.parse_args()  # parsed flags
    if args.stage2 and args.stage3:
        parser.error("--stage2 and --stage3 are mutually exclusive")  # pick one stage

    # later stages select on `none`; stage 1 keeps the original pipeline scenario
    select_scenario = "none" if (args.stage2 or args.stage3) else "kinematic"

    if args.select_only:
        # reload the stage's saved matrix instead of re-running the grid
        df = pd.read_csv(RESULTS_STG3 if args.stage3 else
                         (RESULTS_STG2 if args.stage2 else RESULTS))
    else:
        from src import data as D  # array loader for mask statistics

        arrays = D.load_arrays()  # arrays for mask stats
        missing_graded = np.isnan(arrays.ball).any(axis=1) & arrays.graded  # frames lacking ball
        run_lengths = empirical_run_lengths(missing_graded)  # run lengths to simulate
        print(f"empirical missing runs: n={len(run_lengths)} "  # report run-length stats
              f"median={int(np.median(run_lengths))} max={int(run_lengths.max())}", flush=True)

        # pick matches, scenarios and grid for the requested mode
        if args.smoke:
            matches, scenarios, grid = [MATCH_IDS[0]], ["kinematic"], smoke_grid()  # smoke: 1 match, 6 cells
        elif args.stage3:  # stage-3 mode
            matches, scenarios, grid = MATCH_IDS, ["none"], points_of(GRID_S3)  # final grid
        elif args.stage2:  # stage-2 mode
            matches, scenarios, grid = MATCH_IDS, ["none"], points_of(GRID_EXT)  # extended grid
        else:
            matches, scenarios, grid = MATCH_IDS, SCENARIOS, grid_points()  # full stage-1 run
        for mid in matches:
            load_match(mid)   # ensure cache (parses if needed)

        # one worker task per (match, scenario); all share the same grid
        tasks = [(m, s, grid, run_lengths) for m in matches for s in scenarios]
        print(f"{len(tasks)} tasks ({len(matches)} matches x {len(scenarios)} scenarios), "  # task/cell counts
              f"{len(grid)} cells each", flush=True)

        from multiprocessing import Pool  # parallel workers

        import time as _time  # clock for progress lines
        t0 = _time.time()  # progress clock start
        all_rows: list[dict] = []  # every task's rows
        with Pool(processes=min(args.workers, len(tasks))) as pool:
            for i, rows in enumerate(pool.imap_unordered(run_task, tasks), 1):
                all_rows.extend(rows)  # collect as tasks finish
                print(f"[{i}/{len(tasks)}] {rows[0]['match_id']}/{rows[0]['scenario']} "  # per-task progress
                      f"({len(rows)} cells, elapsed {_time.time() - t0:.0f}s)", flush=True)

        df = pd.DataFrame(all_rows)  # rows -> results table
        out_path = (ROOT / "tuning" / "results" / "smoke.csv") if args.smoke else (
            RESULTS_STG3 if args.stage3 else
            (RESULTS_STG2 if args.stage2 else RESULTS))  # stage-specific file
        df.to_csv(out_path, index=False)  # save the matrix
        print(f"wrote {out_path} ({len(df)} rows)", flush=True)  # confirm matrix written
        if args.smoke:
            return 0  # smoke stops before selection

    integrity = None  # no cross-stage check for stage 1
    if args.stage2 or args.stage3:
        # overlapping cells must reproduce the prior stage's numbers exactly
        prior = pd.read_csv(RESULTS if args.stage2 else RESULTS_STG2)  # previous matrix
        prior = prior[prior["scenario"] == "none"]  # only the overlapping scenario
        merged = df.merge(prior, on=["match_id", "scenario", *PARAMS],
                          suffixes=("_new", "_old"))  # join shared cells
        if len(merged):
            diff = (merged["spell_f1_new"] - merged["spell_f1_old"]).abs().max()  # worst drift
            integrity = (f"{len(merged)} overlapping cells reproduce the prior stage "  # check message
                         f"(max |Δ| {diff:.2e})")
            assert diff < 1e-9, integrity  # fail loudly on drift
            print("integrity:", integrity, flush=True)  # show check result
    if args.stage3:
        # final selection runs on the stage2+stage3 union (stage3 wins overlaps)
        df = load_union()

    # ---- selection + report ----
    chosen, detail = flat_select(df, scenario=select_scenario)  # apply plateau rule
    chosen_rows = df[df["scenario"] == select_scenario]  # selection-scenario rows
    for p, v in chosen.items():
        chosen_rows = chosen_rows[chosen_rows[p] == v]  # narrow to the chosen cell
    chosen_scores = chosen_rows["spell_f1"].to_numpy()  # per-match scores at chosen
    chosen_mean = float(chosen_scores.mean())  # mean at chosen cell
    best_mean = float(df[df["scenario"] == select_scenario]
                      .groupby(PARAMS)["spell_f1"].mean().max())  # best cell, for the gap
    ci = bootstrap_ci(chosen_scores)  # match-level 95% CI

    lomo_df = lomo(df, chosen, scenario=select_scenario)  # held-out sanity check
    gap = float((lomo_df["fold_score"] - lomo_df["global_score"]).mean())  # mean LOMO gap

    robust = {}  # chosen cell's mean score per scenario
    for scenario in sorted(df["scenario"].unique()):
        cell = df[df["scenario"] == scenario]  # rows of this scenario
        for p, v in chosen.items():  # narrow per parameter
            cell = cell[cell[p] == v]  # keep matching value only
        robust[scenario] = float(cell["spell_f1"].mean()) if len(cell) else None  # None if absent

    n_cells = int(df[PARAMS].drop_duplicates().shape[0])  # unique configs
    # markdown report body: summary, LOMO detail, top-10, documented deviations
    lines = [
        f"# P5 {'stage-3 final' if args.stage3 else ('stage-2 extension' if args.stage2 else 'sweep')} "
        "report (IDSSE, CC BY 4.0 — external only)",
        "",
        f"- grid: {n_cells} unique cells x {sorted(df['scenario'].unique())} x "
        f"{df['match_id'].nunique()} matches",
        "- metric: spellF1 (vendored official metric code) on algorithmically derived labels",
        f"- selection scenario: {select_scenario}; rule: within 0.01 of best, "
        "maximize worst neighbour",
        *( [f"- integrity: {integrity}"] if integrity else [] ),
        "",
        f"**chosen**: {_cell_str(chosen)}",
        f"- mean spellF1 ({select_scenario}): {chosen_mean:.4f}  | best cell: {best_mean:.4f}  "
        f"| gap to best: {best_mean - chosen_mean:.4f}",
        f"- match-level bootstrap 95% CI of chosen mean (n={len(chosen_scores)} matches): "
        f"[{ci[0]:.4f}, {ci[1]:.4f}]",
        f"- LOMO: mean(fold pick - global pick) on held-out = {gap:+.4f}",
        *[f"- scenario at chosen: {k} = {('%.4f' % v) if v is not None else 'n/a'}"
          for k, v in robust.items()],
        "",
        "## LOMO detail",
        lomo_df.to_markdown(index=False),
        "",
        f"## Top 10 cells by mean ({select_scenario})",
        detail.sort_values("mean", ascending=False).head(10).to_markdown(index=False),
        "",
        "## Documented deviations",
        "- cache format pickle instead of parquet (no pyarrow; same-env cache only)",
        "- bootstrap resamples MATCHES only (spellF1 not block-decomposable — plan R-U3)",
        "- third fill scenario (simulated detector) dropped with the detector tier (stretch)",
        "- mask placement uniform (real missingness correlates with phases we may not quantify)",
    ]
    report_path = REPORT_FINAL if args.stage3 else (REPORT_STG2 if args.stage2 else REPORT)  # stage-specific file
    report_path.write_text("\n".join(lines) + "\n")  # write the stage report
    print(f"chosen: {_cell_str(chosen)}")  # echo chosen cell
    print(f"mean {chosen_mean:.4f} (best {best_mean:.4f}) CI {ci}")  # echo mean, best, CI
    print(f"wrote {report_path}")  # confirm report written
    return 0  # success


if __name__ == "__main__":  # CLI entry
    raise SystemExit(main())  # run CLI, exit with its status
