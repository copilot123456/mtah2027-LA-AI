"""Tune the D (dead-ball) channel of the notebook's Viterbi decoder on the public GT window.

The decoder under test is the notebook's own: like run_viterbi_vs_gt.py, this script execs the
cell that defines ViterbiPossessionConfig / PossessionViterbiHMM and overrides config fields per
grid cell, so the sweep always scores the real logic as it lives in notebooks/viterbi.ipynb --
the selected values are then written back into that cell, never copied out.

Why only the first half: the public ground truth grades frames 11185..18684, all inside the
first half, and predict() runs one independent Viterbi pass per period. The second half's path
cannot change the first half's, so decoding period=="first_half" reproduces the full decode for
every graded frame at about half the cost (verified empirically by --diag).

WHY SELECTION IS ON macro_f1, NOT spell_f1. The window has 7500 frames but only 17
possessions, so spell_f1 there is extremely noisy -- the baseline's own segment bootstrap on
this window spans 0.19..0.39, wider than any plausible tuning gain (possession_metrics.
SPELL_NOISE_BAND = 0.04). macro_f1 is frame-level with a ~0.01 band and is the leaderboard's
first tie-break anyway. spell_f1 is still reported for every cell, and the chosen cell must not
lose more than the spell noise band against the baseline, otherwise the D fix just trades
frame accuracy for possession structure.

SELECTION RULE (pre-declared): among cells within 0.01 of the best macro_f1 AND whose spell_f1
is within SPELL_NOISE_BAND of the baseline's, take the one with the highest worst
grid-neighbour (a neighbour differs in exactly one parameter), ties broken by macro_f1 then
spell_f1. The spell guard is part of the rule, not an afterthought: the smoke run showed the
macro-optimal cell paying for its D recall with split possessions (spell_f1 0.28 -> 0.22), and
a D fix that shreds possession structure is not a fix. If no cell passes both constraints the
guard relaxes with a warning -- reported, never silent.

IN-SAMPLE CAVEAT: the window that selects is the window that reports. The grid is kept small
and physically motivated; the two-half split and the paired bootstrap printed at the end are the
overfit checks. A gain smaller than the bootstrap noise is not a gain.

Usage:
    python evaluation/tune_viterbi_d.py --diag        # diagnosis + equivalence check, no sweep
    python evaluation/tune_viterbi_d.py               # diagnosis + sweep + selection + report
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
import possession_metrics as pm  # noqa: E402
from run_viterbi_vs_gt import (  # noqa: E402
    build_submission,
    load_decoder_from_notebook,
)

SWEEP_PATH = os.path.join(ROOT, "tuning", "results", "viterbi_d_sweep.csv")
REPORT_PATH = os.path.join(ROOT, "tuning", "results", "report_viterbi_d.md")

# Focused grid over the D channel only, designed from the --diag evidence:
#   * frac_stat during GT dead ball centres on ~0.23 (median), never near the notebook's 0.44,
#     so the stationary threshold has to come DOWN into the data's actual range or the
#     highest-weight D term (w_dead_stationary=3.60) never fires;
#   * mean speed is the real discriminator (GT-D 1.13-1.38 vs alive 2.16/2.52 m/s): raise the
#     speed threshold, and -- because the diagnosis shows the model is confident about
#     possession during missed dead ball (pL/pA ~0.67 vs pD ~0.17, a ~1.4 nat/frame gap that a
#     one-off ~8-nat transition cost cannot beat) -- scale the low-speed EVIDENCE itself;
#   * w_dead_active_penalty stays in the grid because it is what protects slow-but-alive
#     frames from the aggressive thresholds;
#   * min_dead_frames goes UP, not down: every GT dead run in the window is >=138 frames, so a
#     higher floor cannot hurt recall and absorbs false D on slow alive phases.
# p_play_to_dead is deliberately NOT swept: the emission gap over a 138+ frame run dwarfs the
# one-off entry cost, so it cannot be the blocker (verified by probe after the sweep).
# H/A/X emission weights are untouched -- the baseline already scores 0.735/0.740 F1 on H/A.
GRID = {
    "stat_threshold_dead": [0.16, 0.20, 0.26, 0.33, 0.44],
    "speed_threshold_dead": [1.65, 1.95, 2.25, 2.55],
    "w_dead_low_speed": [2.9, 4.5, 6.0, 8.0],
    "w_dead_active_penalty": [1.0, 1.9, 2.8],
    "min_dead_frames": [16, 25, 40],
}
PARAMS = list(GRID)

# The notebook's current defaults -- every one of them is a member of GRID, so the baseline
# cell is scored by the sweep like any other and the selection is apples-to-apples.
BASELINE = {
    "stat_threshold_dead": 0.44,
    "speed_threshold_dead": 1.65,
    "w_dead_low_speed": 2.9,
    "w_dead_active_penalty": 2.8,
    "min_dead_frames": 25,
}
assert all(BASELINE[p] in GRID[p] for p in PARAMS)

MACRO_TOL = 0.01          # plateau tolerance on macro_f1 (possession_metrics noise floor)
SPELL_NOISE_BAND = pm.SPELL_NOISE_BAND  # guard: chosen spell_f1 may not lose more than this


# -- decoding -------------------------------------------------------------------------------

def decode_first_half(ns, df_features, overrides):
    """Run the notebook decoder with config overrides; returns (df_viterbi, summary, seconds)."""
    cfg_cls = ns["ViterbiPossessionConfig"]
    hmm_cls = ns["PossessionViterbiHMM"]
    df_half = (df_features[df_features["period"] == "first_half"]
               .sort_values("video_frame").reset_index(drop=True))
    t0 = time.time()
    hmm = hmm_cls(cfg_cls(**overrides))
    df_viterbi, summary = hmm.predict(df_half)
    return df_viterbi, summary, time.time() - t0, hmm, df_half


def score_submission(df_viterbi, gt_codes, n_frames):
    """Official metrics for one decode against the full-length GT codes."""
    submission = build_submission(df_viterbi, n_frames)
    pred_codes = pm.label_codes(submission["label"])
    return pm.evaluate(gt_codes, pred_codes, bootstrap_n=0), pred_codes


def half_scores(gt_codes, pred_codes, split_frame):
    """macro_f1 on each half of the graded window (overfit check)."""
    out = {}
    frames = np.arange(len(gt_codes))
    graded = gt_codes != pm.label_codes(["X"])[0]
    for name, mask in [
        ("win_first", graded & (frames < split_frame)),
        ("win_second", graded & (frames >= split_frame)),
    ]:
        cm3 = pm.scored_confusion(pm.confusion(gt_codes[mask], pred_codes[mask]))
        out[name] = pm.macro_f1(cm3)
    return out


# Columns the decoder actually reads. The parquet carries ~305 (46 tracked slots x 6 + team
# features); loading only these keeps one worker at ~15 MB instead of ~400 MB, which is what
# six parallel decodes need (a full-column worker OOM'd on the deep copy inside predict()).
# The emission's two by-name optionals are added when the file has them, so behaviour still
# matches the notebook reading the whole parquet.
SCORE_COLUMNS = [
    "video_frame", "time_s", "period", "ball_available",
    "ball_x_m", "ball_y_m",
    "d_min_L", "d_min_A",
    "closing_speed_L", "closing_speed_A",
    "count_3m_L", "count_5m_L", "count_3m_A", "count_5m_A",
    "centroid_x_L", "centroid_x_A",
    "mean_speed_L", "mean_speed_A",
    "frac_stationary_L", "frac_stationary_A",
    "n_visible_L", "n_visible_A",
]
OPTIONAL_COLUMNS = ["ball_speed_mps", "low_confidence_flag"]


def load_features(path):
    """Read only the columns the notebook decoder touches (see SCORE_COLUMNS)."""
    cols = list(SCORE_COLUMNS)
    try:
        import pyarrow.parquet as pq
        names = set(pq.ParquetFile(path).schema.names)
        cols += [c for c in OPTIONAL_COLUMNS if c in names]
    except Exception:
        pass  # no pyarrow metadata path: current parquet has neither optional column
    return pd.read_parquet(path, columns=cols)


def load_gt(path):
    """GT table + full-length codes + window context shared by main and the sweep workers."""
    gt = pd.read_csv(path).rename(columns={"label": "gt"}).sort_values("frame")
    gt["gt"] = gt["gt"].astype(str).str.strip().str.upper()
    gt_codes_all = pm.label_codes(gt["gt"].to_numpy())
    window_mask = np.isin(gt["gt"], list(pm.SCORED))
    window_frames = gt.loc[window_mask, "frame"].to_numpy(dtype=np.int64)
    return {"gt": gt, "gt_codes": gt_codes_all, "n_frames": len(gt_codes_all),
            "window_mask": window_mask, "window_frames": window_frames,
            "split_frame": int(np.median(window_frames)) + 1}


def md_table(df):
    """DataFrame -> markdown table; local because pandas.to_markdown needs tabulate, which is
    not a project dependency."""
    def fmt(v):
        return "{:.4f}".format(v) if isinstance(v, float) else str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def score_cell(ns, df_features, cell, g):
    """Decode one grid cell and score it; the unit the sweep parallelises over."""
    df_dec, _, _, _, _ = decode_first_half(ns, df_features, cell)
    res, pred = score_submission(df_dec, g["gt_codes"], g["n_frames"])
    halves = half_scores(g["gt_codes"], pred, g["split_frame"])
    return {
        **cell,
        "macro_f1": res["macro_f1"],
        "spell_f1": res["spell_f1_mean"],
        "D_precision": res["per_class"]["D"]["precision"],
        "D_recall": res["per_class"]["D"]["recall"],
        "D_f1": res["per_class"]["D"]["f1"],
        "alive_dead_f1": res["alive_dead_f1"],
        "team_acc_both_alive": res["team_acc_both_alive"],
        "n_abstain": res["n_abstain_on_graded"],
        "n_pred_spells": res["n_pred_spells"],
        "win_first": halves["win_first"],
        "win_second": halves["win_second"],
    }


# Worker-side cache: on Windows the Pool spawns fresh interpreters, so each process reads the
# parquet and execs the notebook cell exactly once, keyed by path.
_WORKER_CACHE = {}


def score_chunk(task):
    """All cells of one chunk, for multiprocessing.Pool.imap_unordered."""
    features_path, gt_path, cells = task
    key = (features_path, gt_path)
    if _WORKER_CACHE.get("key") != key:
        _WORKER_CACHE.clear()
        _WORKER_CACHE["key"] = key
        _WORKER_CACHE["ns"] = load_decoder_from_notebook()
        _WORKER_CACHE["features"] = load_features(features_path)
        _WORKER_CACHE["gt"] = load_gt(gt_path)
    return [score_cell(_WORKER_CACHE["ns"], _WORKER_CACHE["features"], cell,
                       _WORKER_CACHE["gt"])
            for cell in cells]


# -- diagnosis -------------------------------------------------------------------------------

def _ema(arr, alpha, ns):
    return ns["_bidirectional_ema"](np.asarray(arr, dtype=np.float64), alpha)


def print_diagnosis(ns, df_win, gt_labels, pred_labels, probs):
    """Why are GT dead frames called alive? Feature + emission evidence, GT-D split by verdict."""
    w = df_win.copy()
    w["gt"] = gt_labels
    w["pred"] = pred_labels
    w["pL"], w["pA"], w["pD"], w["pX"] = probs.T
    w["mean_spd"] = 0.5 * (w["mean_speed_L"] + w["mean_speed_A"])
    w["frac_stat"] = 0.5 * (w["frac_stationary_L"] + w["frac_stationary_A"])
    w["ball_ok"] = w["ball_available"].astype(bool)

    # ball speed the way the emission computes it (no ball_speed_mps column in the parquet)
    bx = np.nan_to_num(w["ball_x_m"].to_numpy(dtype=float), nan=0.0)
    by = np.nan_to_num(w["ball_y_m"].to_numpy(dtype=float), nan=0.0)
    w["ball_spd"] = np.hypot(np.gradient(bx), np.gradient(by)) * ns["ViterbiPossessionConfig"]().fps
    w["ball_idle"] = w["ball_ok"] & (w["ball_spd"] < 0.20)
    pitch_l = ns["ViterbiPossessionConfig"]().pitch_l
    pitch_w = ns["ViterbiPossessionConfig"]().pitch_w
    w["ball_oob"] = (w["ball_ok"]
                     & ((w["ball_x_m"] < -0.35) | (w["ball_x_m"] > pitch_l + 0.35)
                        | (w["ball_y_m"] < -0.35) | (w["ball_y_m"] > pitch_w + 0.35)))

    # the exact stoppage_evidence terms the emission uses
    cfg = ns["ViterbiPossessionConfig"]()
    frac_stat = _ema(w["frac_stat"], 0.25, ns)
    mean_spd = _ema(w["mean_spd"], 0.25, ns)
    excess = np.clip((frac_stat - cfg.stat_threshold_dead) / (1.0 - cfg.stat_threshold_dead), 0, 1)
    deficit = np.clip((cfg.speed_threshold_dead - mean_spd) / cfg.speed_threshold_dead, 0, 1)
    w["stop_evid"] = np.maximum(excess, deficit)

    stat = ["mean_spd", "frac_stat", "stop_evid"]
    print("\n=== DIAGNOSIS: graded window, n={} ===".format(len(w)))
    print("\n-- feature signature by GT label --")
    print(w.groupby("gt")[stat + ["ball_ok", "ball_idle", "ball_oob"]]
          .agg(["mean", "median"]).round(3).to_string())

    gt_d = w[w["gt"] == "D"]
    print("\n-- GT=D frames (n={}), split by what Viterbi said --".format(len(gt_d)))
    print(gt_d.groupby("pred")[stat + ["ball_ok", "ball_idle", "ball_oob", "pD", "pL", "pA", "pX"]]
          .agg(["mean", "median"]).round(3).to_string())

    missed = gt_d[gt_d["pred"] != "D"]
    caught = gt_d[gt_d["pred"] == "D"]
    for name, sub in [("missed", missed), ("caught", caught)]:
        if not len(sub):
            continue
        print("  GT-D {:6s}: frac_stat>={:.2f} in {:5.1f}% | mean_spd<{:.2f} in {:5.1f}% | "
              "either in {:5.1f}% | both in {:5.1f}%".format(
                  name, cfg.stat_threshold_dead,
                  100.0 * (sub["frac_stat"] >= cfg.stat_threshold_dead).mean(),
                  cfg.speed_threshold_dead,
                  100.0 * (sub["mean_spd"] < cfg.speed_threshold_dead).mean(),
                  100.0 * ((sub["frac_stat"] >= cfg.stat_threshold_dead)
                           | (sub["mean_spd"] < cfg.speed_threshold_dead)).mean(),
                  100.0 * ((sub["frac_stat"] >= cfg.stat_threshold_dead)
                           & (sub["mean_spd"] < cfg.speed_threshold_dead)).mean()))

    alive_wrong_d = w[(w["gt"].isin(["H", "A"])) & (w["pred"] == "D")]
    print("\n-- alive frames wrongly called D (n={}): --".format(len(alive_wrong_d)))
    if len(alive_wrong_d):
        print(alive_wrong_d[stat + ["ball_ok"]].agg(["mean", "median"]).round(3).to_string())

    def runs_of(labels, value):
        out, cur = [], 0
        for lab in labels:
            if lab == value:
                cur += 1
            elif cur:
                out.append(cur)
                cur = 0
        if cur:
            out.append(cur)
        return out

    gt_d_runs = runs_of(w["gt"].tolist(), "D")
    pred_d_runs = runs_of(w["pred"].tolist(), "D")
    print("\n-- dead runs in window (frames @25fps) --")
    print("  GT   D runs: n={} lengths={}".format(len(gt_d_runs), gt_d_runs))
    print("  pred D runs: n={} lengths={}".format(len(pred_d_runs), pred_d_runs))
    # per GT dead run, how much did the model detect?
    idx = 0
    for length in gt_d_runs:
        sub = w.iloc[idx:idx + length]
        print("    GT run frames {:>5}..{:<5}: pred D {:5.1f}%  (mean pD={:.3f})".format(
            int(sub["video_frame"].iloc[0]), int(sub["video_frame"].iloc[-1]),
            100.0 * (sub["pred"] == "D").mean(), sub["pD"].mean()))
        idx += length
    return w


# -- sweep -----------------------------------------------------------------------------------

def grid_points():
    return [dict(zip(PARAMS, combo)) for combo in itertools.product(*(GRID[p] for p in PARAMS))]


def worst_neighbour(means, key):
    """min macro_f1 over cells differing from `key` in exactly one swept parameter."""
    values = {p: sorted(set(means.index.get_level_values(p))) for p in PARAMS}
    worst = np.inf
    for pos, param in enumerate(PARAMS):
        for value in values[param]:
            if value == key[pos]:
                continue
            other = list(key)
            other[pos] = value
            other = tuple(other)
            if other in means.index:
                worst = min(worst, float(means.loc[other]))
    return worst


def select_cell(df, baseline_spell):
    """Pre-declared rule: within MACRO_TOL of best macro_f1 AND within SPELL_NOISE_BAND of the
    baseline spell_f1, maximise worst grid-neighbour; relaxes the spell guard with a warning if
    no cell passes both (never silent)."""
    means = df.groupby(PARAMS)["macro_f1"].mean()
    best = float(means.max())
    rows = []
    for key in means.index:
        key = key if isinstance(key, tuple) else (key,)
        if float(means.loc[key]) >= best - MACRO_TOL:
            row = df
            for p, v in zip(PARAMS, key):
                row = row[row[p] == v]
            rows.append({"key": key, "macro_f1": float(means.loc[key]),
                         "spell_f1": float(row["spell_f1"].iloc[0]),
                         "worst_neighbour": worst_neighbour(means, key)})
    feasible = [r for r in rows if r["spell_f1"] >= baseline_spell - SPELL_NOISE_BAND]
    if not feasible:
        print("  WARNING: no cell within {:.4f} of baseline spell_f1 {:.4f} also reaches the "
              "macro plateau -- relaxing the spell guard (declaring it, not hiding it)".format(
                  SPELL_NOISE_BAND, baseline_spell))
        feasible = rows
    pick = sorted(feasible, key=lambda r: (-r["worst_neighbour"], -r["macro_f1"],
                                           -r["spell_f1"]))[0]
    return dict(zip(PARAMS, pick["key"])), pick, best


# -- main ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default=os.path.join(
        os.environ.get("MTAH_DATA_DIR") or r"D:\Proyectos\Barcelona project local\data",
        "processed", "{}_atd_ball_features_LA.parquet".format(games.GAME)))
    ap.add_argument("--gt", default=os.path.join(HERE, "gt", "{}_possession_gt_public.csv".format(games.GAME)))
    ap.add_argument("--diag", action="store_true", help="diagnosis + equivalence check only")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    ns = load_decoder_from_notebook()
    df_features = load_features(args.features)

    g = load_gt(args.gt)
    gt = g["gt"]
    gt_codes_all = g["gt_codes"]
    n_frames = g["n_frames"]
    window_mask = g["window_mask"]
    window_frames = g["window_frames"]
    split_frame = g["split_frame"]
    print("graded window: frames {}..{} (n={}), split for half-check at {}".format(
        window_frames.min(), window_frames.max(), len(window_frames), split_frame))

    # --- baseline decode (first half only) ---------------------------------------------------
    df_base, summary, secs, hmm, df_half = decode_first_half(ns, df_features, {})
    print("\n=== BASELINE DECODE SUMMARY (first half only, {:.1f}s) ===".format(secs))
    for k, v in summary.items():
        print("  {:<24}: {}".format(k, v))

    base_result, base_pred = score_submission(df_base, gt_codes_all, n_frames)

    def brief(res, name):
        print("  {:<10} macro_f1 {:.4f} | spell_f1 {:.4f} | D P {:.3f} R {:.3f} F1 {:.3f} | "
              "alive/dead F1 {:.4f} | abstain {}".format(
                  name, res["macro_f1"], res["spell_f1_mean"],
                  res["per_class"]["D"]["precision"], res["per_class"]["D"]["recall"],
                  res["per_class"]["D"]["f1"], res["alive_dead_f1"],
                  res["n_abstain_on_graded"]))

    print("\n=== BASELINE ON GT WINDOW ===")
    brief(base_result, "baseline")
    base_halves = half_scores(gt_codes_all, base_pred, split_frame)
    print("  halves    first {:.4f}  second {:.4f}".format(
        base_halves["win_first"], base_halves["win_second"]))

    # --- diagnosis ----------------------------------------------------------------------------
    log_emit = hmm._compute_log_emissions(df_half)
    probs = np.exp(log_emit)
    frame_to_pred = dict(zip(df_base["video_frame"], df_base["pred_state"]))
    pred_labels = np.array([frame_to_pred.get(f, "X") for f in window_frames], dtype=object)
    gt_labels = gt.loc[window_mask, "gt"].to_numpy(dtype=object)
    win_mask = df_half["video_frame"].isin(window_frames).to_numpy()
    df_win = df_half[win_mask].reset_index(drop=True)
    assert np.array_equal(df_win["video_frame"].to_numpy(), window_frames), \
        "window frame alignment broken"
    win_probs = probs[win_mask]
    print_diagnosis(ns, df_win, gt_labels, pred_labels, win_probs)

    if args.diag:
        # equivalence check: full decode must agree on every graded frame
        full = (ns["PossessionViterbiHMM"](ns["ViterbiPossessionConfig"]())
                .predict(df_features)[0])
        full_pred = dict(zip(full["video_frame"], full["pred_state"]))
        full_labels = np.array([full_pred.get(f, "X") for f in window_frames], dtype=object)
        agree = float((full_labels == pred_labels).mean())
        print("\n=== EQUIVALENCE: first-half-only vs full decode on graded frames ===")
        print("  agreement {:.6f}  (mismatched {} of {})".format(
            agree, int((full_labels != pred_labels).sum()), len(window_frames)))
        return 0

    # --- sweep --------------------------------------------------------------------------------
    points = grid_points()
    n_workers = max(1, min(6, (os.cpu_count() or 2) - 1))
    n_chunks = min(len(points), n_workers * 4)
    tasks = [(args.features, args.gt, points[i::n_chunks]) for i in range(n_chunks)]
    print("\n=== SWEEP: {} cells x {} params | {} workers, {} chunks ===".format(
        len(points), ", ".join(PARAMS), n_workers, n_chunks))
    rows = []
    t0 = time.time()
    if len(tasks) > 1:
        from multiprocessing import Pool
        with Pool(processes=min(n_workers, len(tasks))) as pool:
            for i, chunk_rows in enumerate(pool.imap_unordered(score_chunk, tasks), 1):
                rows.extend(chunk_rows)
                print("  [{}/{} chunks] {} cells done, elapsed {:.0f}s".format(
                    i, len(tasks), len(rows), time.time() - t0), flush=True)
    else:
        for task in tasks:
            rows.extend(score_chunk(task))
    sweep = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SWEEP_PATH), exist_ok=True)
    sweep.to_csv(SWEEP_PATH, index=False)
    print("wrote {} ({} rows, {:.0f}s)".format(SWEEP_PATH, len(sweep), time.time() - t0))

    # --- selection ----------------------------------------------------------------------------
    chosen, pick, best_macro = select_cell(sweep, base_result["spell_f1_mean"])
    same_as_base = chosen == BASELINE

    print("\n=== SELECTION (plateau on macro_f1 tol={}, spell guard band={}) ===".format(
        MACRO_TOL, SPELL_NOISE_BAND))
    print("  best cell macro_f1 {:.4f}".format(best_macro))
    print("  chosen: " + "  ".join("{}={}".format(p, chosen[p]) for p in PARAMS))
    if same_as_base:
        print("  NOTE: selection matches the notebook defaults -- no change warranted")
    print("         macro_f1 {:.4f}  worst-neighbour {:.4f}  spell_f1 {:.4f}".format(
        pick["macro_f1"], pick["worst_neighbour"], pick["spell_f1"]))
    guard_ok = pick["spell_f1"] >= base_result["spell_f1_mean"] - SPELL_NOISE_BAND
    print("  spell guard: {} (chosen {:.4f} vs baseline {:.4f} - band {:.4f})".format(
        "PASS" if guard_ok else "RELAXED (see warning above)",
        pick["spell_f1"], base_result["spell_f1_mean"], SPELL_NOISE_BAND))

    chosen_row = sweep
    for p, v in chosen.items():
        chosen_row = chosen_row[chosen_row[p] == v]
    chosen_row = chosen_row.iloc[0]

    base_map = {
        "macro_f1": base_result["macro_f1"],
        "spell_f1": base_result["spell_f1_mean"],
        "D_recall": base_result["per_class"]["D"]["recall"],
        "D_precision": base_result["per_class"]["D"]["precision"],
        "D_f1": base_result["per_class"]["D"]["f1"],
        "alive_dead_f1": base_result["alive_dead_f1"],
        "team_acc_both_alive": base_result["team_acc_both_alive"],
        "n_abstain": base_result["n_abstain_on_graded"],
        "n_pred_spells": base_result["n_pred_spells"],
        "win_first": base_halves["win_first"],
        "win_second": base_halves["win_second"],
    }

    print("\n=== BASELINE vs CHOSEN ===")
    print("  {:<24} {:>10} {:>10}".format("metric", "baseline", "chosen"))
    for key, label in [("macro_f1", "macro_f1"), ("spell_f1", "spell_f1_mean"),
                       ("D_recall", "D recall"), ("D_precision", "D precision"),
                       ("D_f1", "D F1"), ("alive_dead_f1", "alive/dead F1"),
                       ("team_acc_both_alive", "team acc (both alive)"),
                       ("n_abstain", "abstains (X)"), ("n_pred_spells", "pred possessions"),
                       ("win_first", "macro_f1 window 1st half"),
                       ("win_second", "macro_f1 window 2nd half")]:
        base_val = base_map[key]
        print("  {:<24} {:>10} {:>10}".format(
            label,
            "{:.4f}".format(base_val) if isinstance(base_val, float) else str(base_val),
            "{:.4f}".format(chosen_row[key])))

    # --- probe: is the cost of entering D really irrelevant? ----------------------------------
    def window_codes(df_dec):
        sub = build_submission(df_dec, n_frames)
        return pm.label_codes(sub["label"].to_numpy(dtype=object))[window_mask]

    c_dec, _, _, _, _ = decode_first_half(ns, df_features, chosen)
    chosen_codes = window_codes(c_dec)
    p2d = ns["ViterbiPossessionConfig"]().p_play_to_dead
    print("\n=== PROBE: p_play_to_dead x10 / x0.1 at chosen ===")
    print("  (diagnosis says the one-off ~8-nat entry cost is dwarfed by the ~1.4 nat/frame")
    print("   emission gap over 138+ frame runs, so these should barely move)")
    for factor in (10.0, 0.1):
        d2, _, _, _, _ = decode_first_half(ns, df_features,
                                           {**chosen, "p_play_to_dead": p2d * factor})
        c2 = window_codes(d2)
        print("  p_play_to_dead={:.6f}: agreement {:.6f}  (changed {} of {} graded frames)".format(
            p2d * factor, float((c2 == chosen_codes).mean()),
            int((c2 != chosen_codes).sum()), len(chosen_codes)))

    # --- paired bootstrap (baseline vs chosen) ------------------------------------------------
    if args.bootstrap:
        print("\n=== BOOTSTRAP ({} draws, GT segments) ===".format(args.bootstrap))
        b_low, b_high = pm.bootstrap_macro_f1(gt_codes_all, base_pred,
                                              pm.valid_runs(gt_codes_all), args.bootstrap)
        _, c_pred = score_submission(c_dec, gt_codes_all, n_frames)
        c_low, c_high = pm.bootstrap_macro_f1(gt_codes_all, c_pred,
                                              pm.valid_runs(gt_codes_all), args.bootstrap)
        sb_low, sb_high = pm.bootstrap_spell_f1(gt_codes_all, base_pred,
                                                pm.valid_runs(gt_codes_all), args.bootstrap)
        sc_low, sc_high = pm.bootstrap_spell_f1(gt_codes_all, c_pred,
                                                pm.valid_runs(gt_codes_all), args.bootstrap)
        print("  macro_f1 baseline [{:.4f}, {:.4f}]  chosen [{:.4f}, {:.4f}]".format(
            b_low, b_high, c_low, c_high))
        print("  spell_f1 baseline [{:.4f}, {:.4f}]  chosen [{:.4f}, {:.4f}]".format(
            sb_low, sb_high, sc_low, sc_high))

    # --- top cells + report -------------------------------------------------------------------
    top = sweep.sort_values("macro_f1", ascending=False).head(10)
    print("\n=== TOP 10 CELLS BY macro_f1 ===")
    print(top[PARAMS + ["macro_f1", "spell_f1", "D_recall", "D_precision"]].to_string(index=False))

    lines = [
        "# Viterbi D-channel tuning report (public GT window, in-sample)",
        "",
        "- grid: {} cells x [{}]; selection: macro_f1 plateau (tol {}) constrained by the "
        "spell guard (band {})".format(
            len(points), ", ".join(PARAMS), MACRO_TOL, SPELL_NOISE_BAND),
        "- window: frames {}..{} ({} graded frames, {} GT possessions)".format(
            window_frames.min(), window_frames.max(), len(window_frames),
            base_result["n_gt_spells"]),
        "- notebook defaults (baseline cell): "
        + "  ".join("{}={}".format(p, BASELINE[p]) for p in PARAMS),
        "- spell guard: {}".format("PASS" if guard_ok else "RELAXED"),
        "",
        "**chosen**: " + "  ".join("{}={}".format(p, chosen[p]) for p in PARAMS),
        "",
        "| metric | baseline | chosen |",
        "|---|---:|---:|",
    ]
    for key, label in [("macro_f1", "macro_f1"), ("spell_f1", "spell_f1_mean"),
                       ("D_recall", "D recall"), ("D_precision", "D precision"),
                       ("D_f1", "D F1"), ("alive_dead_f1", "alive/dead F1"),
                       ("win_first", "macro_f1 1st half"), ("win_second", "macro_f1 2nd half")]:
        lines.append("| {} | {:.4f} | {:.4f} |".format(label, base_map[key], chosen_row[key]))
    lines += ["", "## Top 10 cells by macro_f1", "",
              md_table(top[PARAMS + ["macro_f1", "spell_f1", "D_recall", "D_precision"]])]
    if args.bootstrap:
        lines += ["", "## Bootstrap ({} draws, GT segments)".format(args.bootstrap),
                  "- macro_f1 baseline [{:.4f}, {:.4f}] vs chosen [{:.4f}, {:.4f}]".format(
                      b_low, b_high, c_low, c_high),
                  "- spell_f1 baseline [{:.4f}, {:.4f}] vs chosen [{:.4f}, {:.4f}]".format(
                      sb_low, sb_high, sc_low, sc_high)]
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote {}".format(REPORT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
