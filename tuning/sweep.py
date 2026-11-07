"""The P5 sweep: grid over decoder/emission parameters on IDSSE, scored externally.

    python -m tuning.sweep                 # full run (parses missing matches first)
    python -m tuning.sweep --smoke         # plumbing check: 1 match, 6 grid points
    python -m tuning.sweep --select-only   # re-run selection on the existing results

Matrix design: per (match, scenario) the fill/emission depend only on sigma_m, so
emission is computed 3x and the decoder 81x per emission => 243 cells per match-scenario.
All 7 matches x 2 scenarios = 3,402 rows, computed in parallel, then BOTH selection and
LOMO are derived from the same matrix (no extra compute).

Selection rule (documented, pre-declared): among configs within 0.01 mean spellF1 of
the best, pick the one with the highest WORST grid-neighbour score (plateau centre),
ties broken by mean. Scenario for selection: `kinematic` (the shipped pipeline);
`none` is the no-harm robustness check.

Scoring uses the vendored official metric code (evaluation/possession_metrics.py) on
IDSSE-derived labels: GREEN external output.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import possession_metrics as pm  # noqa: E402
from config import CFG  # noqa: E402
from tuning.idsse import MATCH_IDS, load_match  # noqa: E402
from tuning.simulate import empirical_run_lengths, mask_ball  # noqa: E402

GRID = {
    "switch_pen": [4.0, 8.0, 12.0],
    "dead_pen": [3.0, 5.0, 8.0],
    "sigma_m": [3.0, 4.5, 6.0],
    "min_dwell_s": [0.5, 1.0, 2.0],
    "switch_support_s": [0.3, 0.6, 1.0],
}
PARAMS = list(GRID)
SCENARIOS = ["kinematic", "none"]
MASK_SEED = 20261107
RESULTS = ROOT / "tuning" / "results" / "sweep_results.csv"
REPORT = ROOT / "tuning" / "results" / "report.md"

_LETTERS = np.array(["X", "H", "A", "D"], dtype=object)  # index by internal code + 1


def grid_points() -> list[dict]:
    return [dict(zip(PARAMS, combo)) for combo in itertools.product(*(GRID[p] for p in PARAMS))]


def _score(gt_codes: np.ndarray, pred_codes: np.ndarray) -> dict:
    runs = pm.valid_runs(gt_codes)
    spell = pm.spell_scores(gt_codes, pred_codes, runs)["mean_f1"]
    cm3 = pm.scored_confusion(pm.confusion(gt_codes, pred_codes))
    return {"spell_f1": float(spell), "macro_f1": float(pm.macro_f1(cm3))}


def run_task(args) -> list[dict]:
    """All grid cells for one (match, scenario). Runs in a worker process."""
    match_id, scenario, grid, run_lengths = args
    t0 = time.time()
    case = load_match(match_id)
    rng = np.random.default_rng(MASK_SEED + sum(ord(c) for c in match_id))
    masked_ball, mask_stats = mask_ball(case.ball, run_lengths, rng)

    base = dict(CFG)
    if scenario == "none":
        base["fill_long_gap_s"] = 0.0        # tier off: masked frames stay `none`
    elif scenario != "kinematic":
        raise ValueError(scenario)

    from src.ball_fill import build_ball_track
    from src.decoder import decode
    from src.emission import build_emission

    track = build_ball_track(masked_ball, case.half, base)
    gt_codes = pm.label_codes(case.letters)
    gt_runs = pm.valid_runs(gt_codes)
    n_gt = len(pm.possession_spells(gt_codes, gt_runs))

    rows = []
    for sigma in GRID["sigma_m"]:
        em = build_emission(track, case.players, case.players_s, case.slot_team,
                            {**base, "sigma_m": sigma})
        for cell in grid:
            if cell["sigma_m"] != sigma:
                continue
            cfg = {**base, **cell}
            decoded = decode(em, track.source, case.half, cfg)
            pred = _LETTERS[np.asarray(decoded.labels, dtype=np.int8) + 1]
            pred_codes = pm.label_codes(pred)
            scores = _score(gt_codes, pred_codes)
            rows.append({
                "match_id": match_id, "scenario": scenario, **cell, **scores,
                "n_gt_spells": n_gt,
                "n_pred_spells": len(pm.possession_spells(pred_codes, gt_runs)),
                "runtime_s": round(time.time() - t0, 1),
                "masked_fraction": round(mask_stats["masked_fraction"], 4),
            })
    return rows


# ---------------- selection ----------------

def _mean_table(df: pd.DataFrame, metric: str = "spell_f1") -> pd.DataFrame:
    return df.groupby(PARAMS)[metric].agg(["mean", "std", "count"]).reset_index()


def _neighbour_means(means: pd.Series) -> pd.Series:
    """For each config index: min mean over grid-neighbours (differ in exactly one param)."""
    keys = list(means.index)
    out = {}
    for key in keys:
        scores = []
        for param in PARAMS:
            k = list(key) if isinstance(key, tuple) else [key]
            idx = PARAMS.index(param)
            for value in GRID[param]:
                if value == k[idx]:
                    continue
                other = list(k)
                other[idx] = value
                other_key = tuple(other) if len(PARAMS) > 1 else other[0]
                if other_key in means.index:
                    scores.append(means.loc[other_key])
        out[key] = min(scores) if scores else np.nan
    return pd.Series(out, name="worst_neighbour")


def flat_select(df: pd.DataFrame, scenario: str = "kinematic",
                metric: str = "spell_f1", tol: float = 0.01) -> tuple[dict, pd.DataFrame]:
    """Pre-declared plateau rule; returns (chosen_params, full_mean_table_with_details)."""
    sub = df[df["scenario"] == scenario]
    means = _mean_table(sub, metric).set_index(PARAMS)[["mean", "std"]]
    best = means["mean"].max()
    candidates = means[means["mean"] >= best - tol].copy()
    worst_nb = _neighbour_means(means["mean"])
    candidates["worst_neighbour"] = worst_nb.reindex(candidates.index)
    top = candidates.sort_values(["worst_neighbour", "mean"], ascending=False).iloc[0]
    chosen = {p: float(v) for p, v in zip(PARAMS, top.name)}   # params live in the MultiIndex
    detail = means.copy()
    detail["worst_neighbour"] = worst_nb
    detail["selected"] = False
    detail.loc[[top.name], "selected"] = True
    return chosen, detail.reset_index()


def lomo(df: pd.DataFrame, chosen: dict, metric: str = "spell_f1") -> pd.DataFrame:
    """Pick on 6 matches (plateau rule), report on the held-out one; vs global chosen."""
    rows = []
    matches = sorted(df["match_id"].unique())
    for held in matches:
        train = df[df["match_id"] != held]
        fold_chosen, _ = flat_select(train, metric=metric)
        test = df[df["match_id"] == held]
        fold_pick = _cell_score(test, fold_chosen)
        global_pick = _cell_score(test, chosen)
        rows.append({"held_out": held, "fold_pick": _cell_str(fold_chosen),
                     "fold_score": fold_pick, "global_score": global_pick})
    return pd.DataFrame(rows)


def _cell_score(df: pd.DataFrame, cell: dict) -> float:
    sel = df
    for p, v in cell.items():
        sel = sel[sel[p] == v]
    return float(sel["spell_f1"].mean())


def _cell_str(cell: dict) -> str:
    return "/".join(f"{k}={v:g}" for k, v in cell.items())


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 7) -> tuple[float, float]:
    """Match-level bootstrap CI of the mean (matches are the sample unit)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def smoke_grid() -> list[dict]:
    full = grid_points()
    return [full[0], full[40], full[80], full[120], full[160], full[-1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="1 match, 6 cells")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    if args.select_only:
        df = pd.read_csv(RESULTS)
    else:
        from src import data as D

        arrays = D.load_arrays()
        missing_graded = np.isnan(arrays.ball).any(axis=1) & arrays.graded
        run_lengths = empirical_run_lengths(missing_graded)
        print(f"empirical missing runs: n={len(run_lengths)} "
              f"median={int(np.median(run_lengths))} max={int(run_lengths.max())}", flush=True)

        matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS
        scenarios = ["kinematic"] if args.smoke else SCENARIOS
        grid = smoke_grid() if args.smoke else grid_points()
        for mid in matches:
            load_match(mid)   # ensure cache (parses if needed)

        tasks = [(m, s, grid, run_lengths) for m in matches for s in scenarios]
        print(f"{len(tasks)} tasks ({len(matches)} matches x {len(scenarios)} scenarios), "
              f"{len(grid)} cells each", flush=True)

        from multiprocessing import Pool

        all_rows: list[dict] = []
        with Pool(processes=min(args.workers, len(tasks))) as pool:
            for i, rows in enumerate(pool.imap_unordered(run_task, tasks), 1):
                all_rows.extend(rows)
                print(f"[{i}/{len(tasks)}] {rows[0]['match_id']}/{rows[0]['scenario']} "
                      f"done in {rows[0]['runtime_s']}s ({len(rows)} cells)", flush=True)

        df = pd.DataFrame(all_rows)
        df.to_csv(RESULTS, index=False)
        print(f"wrote {RESULTS} ({len(df)} rows)", flush=True)
        if args.smoke:
            return 0

    # ---- selection + report ----
    chosen, detail = flat_select(df)
    means_kin = df[df["scenario"] == "kinematic"].groupby(PARAMS)["spell_f1"].mean()
    chosen_mean = means_kin[
        (means_kin.index.get_level_values("switch_pen") == chosen["switch_pen"])
        & (means_kin.index.get_level_values("dead_pen") == chosen["dead_pen"])
        & (means_kin.index.get_level_values("sigma_m") == chosen["sigma_m"])
        & (means_kin.index.get_level_values("min_dwell_s") == chosen["min_dwell_s"])
        & (means_kin.index.get_level_values("switch_support_s") == chosen["switch_support_s"])
    ].iloc[0]
    best_mean = means_kin.max()
    ci = bootstrap_ci(means_kin.to_numpy())

    lomo_df = lomo(df, chosen)
    gap = float((lomo_df["fold_score"] - lomo_df["global_score"]).mean())

    robust = {}
    for scenario in SCENARIOS:
        sub = df[df["scenario"] == scenario]
        cell = sub
        for p, v in chosen.items():
            cell = cell[cell[p] == v]
        robust[scenario] = float(cell["spell_f1"].mean())
    best_none = float(df[df["scenario"] == "none"].groupby(PARAMS)["spell_f1"].mean().max())

    lines = [
        "# P5 sweep report (IDSSE, CC BY 4.0 — external only)",
        "",
        f"- grid: {len(grid_points())} cells x {len(SCENARIOS)} scenarios x {len(MATCH_IDS)} matches",
        f"- metric: spellF1 (vendored official metric code) on algorithmically derived labels",
        f"- selection scenario: kinematic; rule: within 0.01 of best, maximize worst neighbour",
        "",
        f"**chosen**: {_cell_str(chosen)}",
        f"- mean spellF1 (kinematic): {chosen_mean:.4f}  | best cell: {best_mean:.4f}  "
        f"| gap to best: {best_mean - chosen_mean:.4f}",
        f"- match-level bootstrap 95% CI of chosen mean: [{ci[0]:.4f}, {ci[1]:.4f}]",
        f"- LOMO: mean(fold pick - global pick) on held-out = {gap:+.4f} "
        f"(optimism; positive means LOMO picks beat global on held-out)",
        f"- no-harm on `none` scenario: chosen {robust['none']:.4f} vs best-none {best_none:.4f}",
        "",
        "## LOMO detail",
        lomo_df.to_markdown(index=False),
        "",
        "## Top 10 cells by mean (kinematic)",
        detail.sort_values("mean", ascending=False).head(10).to_markdown(index=False),
        "",
        "## Documented deviations",
        "- cache format pickle instead of parquet (no pyarrow; same-env cache only)",
        "- bootstrap resamples MATCHES only (no within-match blocks: spellF1 is not "
        "block-decomposable without edge artifacts — plan R-U3 caveat)",
        "- third fill scenario (simulated detector) dropped with the detector tier (stretch)",
        "- mask placement uniform (real missingness correlates with phases we may not quantify)",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"chosen: {_cell_str(chosen)}")
    print(f"mean {chosen_mean:.4f} (best {best_mean:.4f}) CI {ci}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
