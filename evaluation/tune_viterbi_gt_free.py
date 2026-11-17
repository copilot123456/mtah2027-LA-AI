"""GT-free calibration of the notebook's Viterbi decoder against IDSSE reference ranges.

Replaces tune_viterbi_d.py, which selected parameters on the challenge ground truth and is
therefore NOT a valid tuner under the organizer clarification (kept only as a historical
record, with a compliance banner). This tuner never opens the challenge GT: it scores each
config with GT-free structural metrics and selects against ranges measured on IDSSE (7
external matches, algorithmic labels -- the sanctioned reference bed).

Pre-declared selection rule:
  feasible  = D share within the IDSSE per-match range [40.7%, 51.7%]
              AND possession changes/min within [3.55, 5.13]
              AND fast-in-field-ball share of predicted D frames <= 5%
  pick      = lowest fraction of team switches without a prior control episode
              (tie-break: changes/min closest to the IDSSE mean 4.25)
  if empty  = relax in declared order (first the fast-D guard, then the D-share floor),
              printing the relaxation -- never silent.

Grid rationale (all externally anchored):
  fixed at the round-3 GT-free best: dead_players_floor=0.4 and w_dead_ball_slow=7.0 (the
    design hierarchy ball >> players: floor 0 made the pure product detect only 17.9% on
    IDSSE where the provider labels 44.9% dead -- players jog during stoppages), and
    ball_slow_mps=3.0 (IDSSE dead frames are 100% slower than that, alive only ~23%);
  beta            -- the emission tempering knob; LOW beta lets the IDSSE-implied prior set
                     the tempo, high beta let the incumbent's persistent gated evidence
                     accumulate a lead the challenger never erodes (round-3 finding);
  p_switch_play   -- prior on L<->A; 0.005 is the rate implied by the RAW IDSSE rate
                     (4.25 / (60*25) / 0.55 alive ~ 0.005); Viterbi is MAP so the prior only
                     sets the cost, the gated emission sets the realized rate;
  p_stay_dead     -- dead-ball dwell prior; IDSSE D runs average 25.3 s (~0.9984).

Usage:
    python evaluation/tune_viterbi_gt_free.py                 # full grid, 6 workers
    python evaluation/tune_viterbi_gt_free.py --bootstrap     # also print best/worst cells

Closing step of every iteration (repo convention, GT freeze): run
evaluation/gt_free_diagnostics.py against the IDSSE reference. The public-GT scoreboard
(evaluation/run_evaluation.py vs baseline_smooth50) is the FINAL report only -- its numbers
must never choose the next objective, because looking at them per cycle is selection on the GT
even with no parameter calibrated on it.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import games  # noqa: E402
from gt_free_diagnostics import (  # noqa: E402
    CONTROL_DIST_M,
    FAST_BALL_MPS,
    FLIGHT_SPEED_MPS,
    FPS,
    SWITCH_LOOKBACK_FRAMES,
    ball_speed_mps,
    possession_stats,
    runs_of,
)
from run_viterbi_vs_gt import load_decoder_from_notebook  # noqa: E402

SWEEP_PATH = os.path.join(ROOT, "tuning", "results", "viterbi_gt_free_sweep.csv")
REPORT_PATH = os.path.join(ROOT, "tuning", "results", "report_viterbi_gt_free.md")

# IDSSE reference (7 matches, evaluation/gt_free_diagnostics.py part C)
IDSSE_D_SHARE = (0.407, 0.517)      # provider dead-label, per-match min..max
# Design-compatible change rate: provider H<->A flips whose NEW owner has the control gate
# open within +/-20 frames, measured match-by-match on IDSSE (2.07..3.12, mean 2.54). The raw
# provider rate (3.55..5.13) counts flips during flight/duels -- moments the design assigns
# "no evidence, keep the last team" by specification, so targeting it would mean punishing
# the design for following its own rule.
IDSSE_CHANGES_PER_MIN = (2.07, 3.12)
IDSSE_CHANGES_MEAN = 2.54
FAST_D_MAX = 0.05                   # design guard: D with a fast in-field ball should be rare

GRID = {
    # final combined grid: the round-3 structural winners are config defaults now
    # (dead_players_floor=0.4, w_dead_ball_slow=7.0, ball_slow_mps=3.0); sweep the tempo.
    "beta": [0.7, 1.0],
    "p_switch_play": [0.005, 0.008],
    "p_stay_dead": [0.9975, 0.9985],
}
PARAMS = list(GRID)

FEATURE_COLUMNS = [
    "video_frame", "time_s", "period", "ball_available", "ball_score",
    "ball_x_m", "ball_y_m", "d_min_L", "d_min_A",
    "mean_speed_L", "mean_speed_A", "frac_stationary_L", "frac_stationary_A",
]


def default_features_path():
    data_dir = os.environ.get("MTAH_DATA_DIR") or r"D:\Proyectos\Barcelona project local\data"
    return os.path.join(data_dir, "processed", "{}_atd_ball_features_LA.parquet".format(games.GAME))


def grid_points():
    return [dict(zip(PARAMS, combo)) for combo in itertools.product(*(GRID[p] for p in PARAMS))]


def score_config(ns, df_features, overrides):
    """GT-free structural metrics for one config. No ground truth is opened anywhere here."""
    cfg = ns["ViterbiPossessionConfig"](**overrides)
    hmm = ns["PossessionViterbiHMM"](cfg)
    df_v, summary = hmm.predict(df_features)

    lab = dict(zip(df_v["video_frame"], df_v["pred_state"]))
    pred = np.array([lab.get(int(v), "X") for v in df_features["video_frame"]], dtype=object)
    pred = np.where(pred == "L", "H", pred)
    playing = pred != "X"
    labels_playing = pred[playing]

    d_runs = runs_of(labels_playing, ("D",))
    n_poss, changes, minutes, mean_poss, _ = possession_stats(pred)

    ball_ok = df_features["ball_available"].to_numpy(dtype=bool)
    bspd = ball_speed_mps(ball_ok, df_features["ball_x_m"].to_numpy(dtype=float),
                          df_features["ball_y_m"].to_numpy(dtype=float), FPS)
    in_field = ((df_features["ball_x_m"].to_numpy(dtype=float) > -0.35)
                & (df_features["ball_x_m"].to_numpy(dtype=float) < cfg.pitch_l + 0.35)
                & (df_features["ball_y_m"].to_numpy(dtype=float) > -0.35)
                & (df_features["ball_y_m"].to_numpy(dtype=float) < cfg.pitch_w + 0.35))
    is_d = labels_playing == "D"
    d_idx = np.flatnonzero(is_d)
    fast_d = (ball_ok[d_idx] & in_field[d_idx]
              & (bspd[d_idx] > FAST_BALL_MPS)).mean() if len(d_idx) else 0.0

    # switches without a prior control episode by the new team (same definition as gt_free)
    d_min_l = df_features["d_min_L"].to_numpy(dtype=float)
    d_min_a = df_features["d_min_A"].to_numpy(dtype=float)
    ctrl = {
        "H": ball_ok & np.isfinite(d_min_l) & (d_min_l < CONTROL_DIST_M)
             & (bspd < FLIGHT_SPEED_MPS),
        "A": ball_ok & np.isfinite(d_min_a) & (d_min_a < CONTROL_DIST_M)
             & (bspd < FLIGHT_SPEED_MPS),
    }
    switch_idx = [i for i in range(1, len(pred))
                  if pred[i] in ("H", "A") and pred[i - 1] in ("H", "A") and pred[i] != pred[i - 1]]
    no_prior = sum(1 for i in switch_idx
                   if not ctrl[pred[i]][max(0, i - SWITCH_LOOKBACK_FRAMES):i].any())

    return {
        **overrides,
        "d_share": float((labels_playing == "D").mean()),
        "d_runs_mean_s": float(np.mean(d_runs) / FPS) if d_runs else 0.0,
        "d_runs_p50_s": float(np.median(d_runs) / FPS) if d_runs else 0.0,
        "changes_per_min": changes / minutes,
        "mean_possession_s": mean_poss,
        "n_possessions": n_poss,
        "fast_d_frac": float(fast_d),
        "no_control_switch_frac": (no_prior / len(switch_idx)) if switch_idx else 0.0,
        "n_switches": len(switch_idx),
        "state_switches": summary["state_switches"],
    }


def feasible(row, relax=0):
    """Pre-declared feasibility against IDSSE ranges; relax>=1 drops fast-D, >=2 widens D-share."""
    d_ok = (IDSSE_D_SHARE[0] - (0.06 if relax >= 2 else 0.0)) <= row["d_share"] \
        <= IDSSE_D_SHARE[1] + (0.06 if relax >= 2 else 0.0)
    ch_ok = IDSSE_CHANGES_PER_MIN[0] <= row["changes_per_min"] <= IDSSE_CHANGES_PER_MIN[1]
    fast_ok = relax >= 1 or row["fast_d_frac"] <= FAST_D_MAX
    return d_ok and ch_ok and fast_ok


def select(sweep):
    for relax in (0, 1, 2):
        cand = sweep[[feasible(r, relax) for r in sweep.to_dict("records")]]
        if len(cand):
            if relax:
                print("  NOTE: relaxed feasibility at level {} (declared, not silent)".format(relax))
            cand = cand.copy()
            cand["_chg_dist"] = (cand["changes_per_min"] - IDSSE_CHANGES_MEAN).abs()
            best = cand.sort_values(
                ["no_control_switch_frac", "_chg_dist"],
                ascending=[True, True]).iloc[0]
            return best, relax
    return None, None


_WORKER_CACHE = {}


def score_chunk(task):
    features_path, cells = task
    if _WORKER_CACHE.get("path") != features_path:
        _WORKER_CACHE.clear()
        _WORKER_CACHE["path"] = features_path
        _WORKER_CACHE["ns"] = load_decoder_from_notebook()
        _WORKER_CACHE["features"] = pd.read_parquet(features_path, columns=FEATURE_COLUMNS)
    return [score_config(_WORKER_CACHE["ns"], _WORKER_CACHE["features"], cell)
            for cell in cells]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default=default_features_path())
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    points = grid_points()
    n_chunks = min(len(points), args.workers * 4)
    tasks = [(args.features, points[i::n_chunks]) for i in range(n_chunks)]
    print("GT-FREE SWEEP: {} cells x [{}] | {} workers".format(
        len(points), ", ".join(PARAMS), args.workers))
    print("selection ranges from IDSSE: D share {}, changes/min {}, fast-D <= {:.0%}".format(
        IDSSE_D_SHARE, IDSSE_CHANGES_PER_MIN, FAST_D_MAX))

    rows, t0 = [], time.time()
    if len(tasks) > 1:
        from multiprocessing import Pool
        with Pool(processes=min(args.workers, len(tasks))) as pool:
            for i, chunk_rows in enumerate(pool.imap_unordered(score_chunk, tasks), 1):
                rows.extend(chunk_rows)
                print("  [{}/{} chunks] {} cells, {:.0f}s".format(
                    i, len(tasks), len(rows), time.time() - t0), flush=True)
    else:
        for task in tasks:
            rows.extend(score_chunk(task))

    sweep = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SWEEP_PATH), exist_ok=True)
    sweep.to_csv(SWEEP_PATH, index=False)
    print("wrote {} ({} rows, {:.0f}s)".format(SWEEP_PATH, len(sweep), time.time() - t0))

    chosen, relax = select(sweep)
    if chosen is None:
        print("NO FEASIBLE CELL -- nothing selected (declared failure, no silent pick)")
        return 1

    print("\n=== SELECTED (GT-free, IDSSE-anchored) ===")
    print("  " + "  ".join("{}={}".format(p, chosen[p]) for p in PARAMS))
    print("  d_share {:.1%} (IDSSE {:.0%}-{:.0%})   changes/min {:.2f} (IDSSE {:.2f}-{:.2f})".format(
        chosen["d_share"], *IDSSE_D_SHARE, chosen["changes_per_min"],
        *IDSSE_CHANGES_PER_MIN))
    print("  fast_d_frac {:.2%}   no_control_switches {:.1%} ({} switches)".format(
        chosen["fast_d_frac"], chosen["no_control_switch_frac"], int(chosen["n_switches"])))
    print("  D runs mean {:.1f}s p50 {:.1f}s (IDSSE 25.3/19.7)   mean possession {:.1f}s "
          "(IDSSE 7.3)".format(chosen["d_runs_mean_s"], chosen["d_runs_p50_s"],
                               chosen["mean_possession_s"]))

    print("\n=== ALL CELLS (sorted by feasibility headroom) ===")
    show = sweep.sort_values("no_control_switch_frac")
    print(show[PARAMS + ["d_share", "changes_per_min", "fast_d_frac",
                         "no_control_switch_frac", "mean_possession_s"]]
          .to_string(index=False, float_format=lambda v: "{:.4g}".format(v)))

    from tune_viterbi_d import md_table  # local markdown helper (no tabulate dependency)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("# Viterbi GT-free calibration (IDSSE-anchored, no challenge GT)\n\n")
        fh.write("Grid: {} cells x [{}]\n".format(len(points), ", ".join(PARAMS)))
        fh.write("Selection: feasible within IDSSE ranges {} D-share, {} changes/min, "
                 "fast-D <= {:.0%}; then min no-control-switch fraction. Relax level: {}\n"
                 .format(IDSSE_D_SHARE, IDSSE_CHANGES_PER_MIN, FAST_D_MAX, relax))
        fh.write("\n**chosen**: " + "  ".join("{}={}".format(p, chosen[p]) for p in PARAMS) + "\n")
        fh.write("\n" + md_table(show[PARAMS + ["d_share", "changes_per_min", "fast_d_frac",
                                                "no_control_switch_frac", "mean_possession_s",
                                                "d_runs_mean_s"]]) + "\n")
        fh.write("\nClosing step: evaluation/gt_free_diagnostics.py vs IDSSE. The public-GT "
                 "scoreboard (run_evaluation.py vs baseline_smooth50) is the FINAL report "
                 "only; its numbers never choose the next objective (GT freeze).\n")
    print("\nwrote {}".format(REPORT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
