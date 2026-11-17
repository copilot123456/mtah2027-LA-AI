"""Leave-one-match-out selection of the notebook's Viterbi config on IDSSE spellF1.

Replaces rate-range matching with the direct objective the board ranks on -- spellF1 over
possession spells -- computed on IDSSE's algorithmic possession labels (external, sanctioned).
The challenge GT is never opened: seven IDSSE matches give seven folds, so each config is
selected on six matches and reported on the held-out seventh (LOMO), which answers "does this
config generalize to a match it never saw" without touching the frozen public GT.

Swept (per the review's dynamics analysis: the morphological filter and w_control are the real
levers in the last-clear-control regime, beta/p_switch nearly flat):
    beta                [0.7, 1.0]
    p_switch_play       [0.005, 0.008]
    min_possession_frames [25, 40]   (40 absorbed A-claims 5.1x more than L ones -- part D1)
    w_control           [1.8, 2.6]
    w_duel_pulse        [0.0, 1.0]   (soft evidence at zero-evidence landings, design 6.3)

The current notebook config (1.0 / 0.008 / 40 / 2.6 / 0) is a grid member, so every fold also
reports how the incumbent does on the held-out match.

Usage:
    python evaluation/select_viterbi_idsse.py             # full LOMO, 6 workers
    python evaluation/select_viterbi_idsse.py --workers 4
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
from run_viterbi_vs_gt import load_decoder_from_notebook  # noqa: E402

FPS = games.GAMES[games.GAME]["fps"]
PITCH = np.array(games.GAMES[games.GAME]["pitch"], dtype=np.float64)

GRID = {
    "beta": [0.7, 1.0],
    "p_switch_play": [0.005, 0.008],
    "min_possession_frames": [25, 40],
    "w_control": [1.8, 2.6],
    "w_duel_pulse": [0.0, 1.0],
}
PARAMS = list(GRID)
CURRENT = {"beta": 1.0, "p_switch_play": 0.008, "min_possession_frames": 40,
           "w_control": 2.6, "w_duel_pulse": 0.0}   # the incumbent notebook config
OUT_CSV = os.path.join(ROOT, "tuning", "results", "viterbi_idsse_lomo.csv")
OUT_MD = os.path.join(ROOT, "tuning", "results", "report_viterbi_idsse_lomo.md")
OUT_MD_DEG = os.path.join(ROOT, "tuning", "results", "report_viterbi_idsse_lomo_degraded.md")
OUT_CSV_DEG = os.path.join(ROOT, "tuning", "results", "viterbi_idsse_lomo_degraded.csv")

GAP_SEED = 20261117  # recorded: C2 degraded-IDSSE gap placement


def default_features_path():
    data_dir = os.environ.get("MTAH_DATA_DIR") or r"D:\Proyectos\Barcelona project local\data"
    return os.path.join(data_dir, "processed", "{}_atd_ball_features_LA.parquet".format(games.GAME))


def atd_gap_structure():
    """Coarse ball-gap structure of the provided match: run lengths + missing share.

    Same precedent as dossier section 7 (seeded placement of run lengths sampled from the
    match's coarse structural distribution). Used to degrade IDSSE to ATD conditions so the
    LOMO measures the parameters under the availability they will actually run with.
    """
    path = default_features_path()
    ok = pd.read_parquet(path, columns=["ball_available"])["ball_available"].to_numpy(dtype=bool)
    lengths, cur = [], 0
    for v in ok:
        if not v:
            cur += 1
        elif cur:
            lengths.append(cur)
            cur = 0
    if cur:
        lengths.append(cur)
    return lengths, float(1.0 - ok.mean())


def degrade_to_atd(features: pd.DataFrame, lengths, missing_share, seed) -> pd.DataFrame:
    """Mask ball evidence in seeded gaps matching the match's gap-length mix and share."""
    rng = np.random.default_rng(seed)
    n = len(features)
    keep = np.ones(n, dtype=bool)
    target, placed, tries = int(missing_share * n), 0, 0
    while placed < target and tries < 20000:
        tries += 1
        ln = int(rng.choice(lengths))
        start = int(rng.integers(0, max(1, n - ln)))
        if keep[start:start + ln].all():
            keep[start:start + ln] = False
            placed += ln
    out = features.copy()
    out.loc[~keep, "ball_available"] = False
    out.loc[~keep, ["ball_x_m", "ball_y_m", "d_min_L", "d_min_A"]] = np.nan
    out.loc[~keep, "ball_score"] = 0.0
    return out


def grid_points():
    return [dict(zip(PARAMS, combo)) for combo in itertools.product(*(GRID[p] for p in PARAMS))]


def build_features(case) -> pd.DataFrame:
    """The notebook emission's input columns, from one IDSSE match (metres, m/s, 25 Hz).

    ball_score is 1.0: IDSSE's provider always has the ball (0% missing), so the confidence
    channel has nothing to discriminate there -- documented, not assumed for BAR-ATH.
    """
    ball = case.ball * PITCH
    ball_ok = np.isfinite(ball).all(axis=1)
    players = case.players * PITCH                      # (N, K, 2) metres
    d = np.linalg.norm(players - ball[:, None, :], axis=-1)
    d = np.where(np.isfinite(d) & ball_ok[:, None], d, np.nan)
    d_home = np.nanmin(np.where(case.slot_team[None, :] == 0, d, np.nan), axis=1)
    d_away = np.nanmin(np.where(case.slot_team[None, :] == 1, d, np.nan), axis=1)

    speed = case.players_s / 3.6                        # km/h -> m/s (verified)
    mean_home = np.nanmean(np.where(case.slot_team[None, :] == 0, speed, np.nan), axis=1)
    mean_away = np.nanmean(np.where(case.slot_team[None, :] == 1, speed, np.nan), axis=1)
    stat = speed < 0.65                                 # atdProcess.stationary_speed_mps
    frac_home = np.nanmean(np.where(case.slot_team[None, :] == 0, stat, np.nan), axis=1)
    frac_away = np.nanmean(np.where(case.slot_team[None, :] == 1, stat, np.nan), axis=1)

    in_match = case.half >= 1
    n = int(in_match.sum())
    return pd.DataFrame({
        "video_frame": np.arange(n, dtype=np.int64),
        "time_s": np.arange(n, dtype=np.float64) / FPS,
        "period": np.where(case.half[in_match] == 1, "first_half", "second_half"),
        "ball_available": ball_ok[in_match],
        "ball_score": np.ones(n, dtype=np.float64),
        "ball_x_m": ball[in_match, 0],
        "ball_y_m": ball[in_match, 1],
        "d_min_L": d_home[in_match],
        "d_min_A": d_away[in_match],
        "mean_speed_L": mean_home[in_match],
        "mean_speed_A": mean_away[in_match],
        "frac_stationary_L": frac_home[in_match],
        "frac_stationary_A": frac_away[in_match],
    })


# Thresholds are PER-SYSTEM (part E: ball speed differs 2.6x, frac_stationary 2.7x between
# IDSSE and the ATD tracking). The notebook carries the quantile-matched ATD-scale values;
# running IDSSE at ATD-scale thresholds shifts the operating point (measured: chosen config
# scores 0.4213 at ATD-scale vs 0.4488 at IDSSE-scale on the same matches). The LOMO/ablation
# select only the dynamics parameters -- thresholds stay at each system's own scale.
IDSSE_THRESHOLDS = {"ball_slow_mps": 3.0, "fast_ball_mps": 4.0,
                    "flight_speed_mps": 7.0, "stat_threshold_dead": 0.15}


def score_config(ns, features, gt_codes, cell):
    cfg = ns["ViterbiPossessionConfig"](**{**IDSSE_THRESHOLDS, **cell})
    hmm = ns["PossessionViterbiHMM"](cfg)
    df_v, _ = hmm.predict(features)
    pred = df_v["pred_state"].to_numpy(dtype=object)
    pred = np.where(pred == "L", "H", pred)             # notebook alphabet -> eval alphabet
    pred_codes = pm.label_codes(pred)
    runs = pm.valid_runs(gt_codes)
    spell = pm.spell_scores(gt_codes, pred_codes, runs)["mean_f1"]
    cm3 = pm.scored_confusion(pm.confusion(gt_codes, pred_codes))
    return {**cell, "spell_f1": float(spell), "macro_f1": pm.macro_f1(cm3)}


_CACHE = {}


def run_match(task):
    match_id, cells, degraded = task[0], task[1], task[2]
    gap_stats = task[3] if len(task) > 3 else None
    from tuning import idsse
    if _CACHE.get("match") != match_id or _CACHE.get("deg") != degraded:
        case = idsse.load_match(match_id)
        _CACHE.clear()
        _CACHE["match"] = match_id
        _CACHE["deg"] = degraded
        features = build_features(case)
        if degraded:
            lengths, share = gap_stats
            features = degrade_to_atd(features, lengths, share,
                                      GAP_SEED + sum(ord(c) for c in match_id))
        _CACHE["features"] = features
        _CACHE["gt"] = pm.label_codes(case.letters[case.half >= 1])
        _CACHE["ns"] = load_decoder_from_notebook()
    return [{**row, "match_id": match_id}
            for row in (score_config(_CACHE["ns"], _CACHE["features"], _CACHE["gt"], cell)
                        for cell in cells)]


def pick(df, metric="spell_f1"):
    """Best mean over matches; tie-break on macro_f1."""
    means = df.groupby(PARAMS)[metric].mean()
    best = means.max()
    tied = means[means >= best - 1e-12]
    detail = df[df.set_index(PARAMS).index.isin(tied.index)]
    macro = detail.groupby(PARAMS)["macro_f1"].mean()
    top = macro.sort_values(ascending=False).index[0]
    return dict(zip(PARAMS, top if isinstance(top, tuple) else (top,)))


def ablation_cells():
    """Frozen chosen config + one-piece-off ablations (each measured against it).

    Every cell carries ALL ablation-relevant fields explicitly so row filters match exactly
    (duel_gap/min_dead would otherwise leak into the baseline mean via NaN columns).
    """
    chosen = {"beta": 0.7, "p_switch_play": 0.005, "min_possession_frames": 25,
              "w_control": 2.6, "w_duel_pulse": 1.0,
              "duel_gap_m": 0.7, "min_dead_frames": 50}
    return chosen, [
        ("chosen (frozen)", chosen),
        ("pulse off (6.3)", {**chosen, "w_duel_pulse": 0.0}),
        ("duel_gap 0.7->0", {**chosen, "duel_gap_m": 0.0}),
        ("min_poss 25->40", {**chosen, "min_possession_frames": 40}),
        ("min_dead 50->25", {**chosen, "min_dead_frames": 25}),
        ("beta 0.7->1.0", {**chosen, "beta": 1.0}),
        ("beta 0.7->0.35", {**chosen, "beta": 0.35}),
    ]


def run_ablation(workers):
    from multiprocessing import Pool
    from tuning.idsse import MATCH_IDS
    chosen, variants = ablation_cells()
    cells = [c for _, c in variants]
    tasks = [(mid, cells[i::len(variants)], False, None) for mid in MATCH_IDS
             for i in range(len(variants))]
    rows, t0 = [], time.time()
    with Pool(processes=min(workers, len(tasks))) as pool:
        for chunk in pool.imap_unordered(run_match, tasks):
            rows.extend(chunk)
    df = pd.DataFrame(rows)
    print("\n=== ABLATIONS ON IDSSE (spellF1 mean over 7 matches, frozen chosen config) ===")
    print("  {:22s} {:>10s} {:>10s} {:>9s}".format("variant", "spellF1", "delta", "macroF1"))
    base = None
    for name, cell in variants:
        sel = df
        for p, v in cell.items():
            sel = sel[sel[p] == v]
        s = sel["spell_f1"].mean()
        m = sel["macro_f1"].mean()
        if base is None:
            base = s
        print("  {:22s} {:10.4f} {:+10.4f} {:9.4f}".format(name, s, s - base, m))
    print("  (a delta within ~+-0.01 on 7 matches is noise: see LOMO spread in part E)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ablate", action="store_true",
                    help="one-piece-off ablations of the frozen config on IDSSE (item 4)")
    ap.add_argument("--degraded", action="store_true",
                    help="C2: inject the match's ball-gap structure into IDSSE before the "
                         "LOMO, so selection runs under ATD availability conditions")
    args = ap.parse_args()
    if args.ablate:
        return run_ablation(args.workers)

    from tuning import idsse
    cells = grid_points()
    gap_stats = atd_gap_structure() if args.degraded else None
    # chunk cells per match so the pool load-balances (one task per match would leave a
    # single worker doing 32 decodes alone)
    tasks = [(mid, cells[i::4], args.degraded, gap_stats)
             for mid in idsse.MATCH_IDS for i in range(4)]
    print("IDSSE LOMO selection{}: {} cells x {} matches ({} tasks) | {} workers".format(
        " [DEGRADED: ATD ball gaps injected, seed {}]".format(GAP_SEED) if args.degraded else "",
        len(cells), len(idsse.MATCH_IDS), len(tasks), args.workers))
    if args.degraded:
        print("  injected gap structure: {} gaps, missing share {:.1%} (from the match)".format(
            len(gap_stats[0]), gap_stats[1]))
    print("grid: " + "  ".join("{}={}".format(p, GRID[p]) for p in PARAMS))

    rows, t0 = [], time.time()
    if len(tasks) > 1 and args.workers > 1:
        from multiprocessing import Pool
        with Pool(processes=min(args.workers, len(tasks))) as pool:
            for i, match_rows in enumerate(pool.imap_unordered(run_match, tasks), 1):
                rows.extend(match_rows)
                print("  [{}/{}] {} cells scored, {:.0f}s".format(
                    i, len(tasks), len(rows), time.time() - t0), flush=True)
    else:
        for task in tasks:
            rows.extend(run_match(task))

    df = pd.DataFrame(rows)
    out_csv = OUT_CSV_DEG if args.degraded else OUT_CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("wrote {} ({} rows, {:.0f}s)".format(out_csv, len(df), time.time() - t0))

    # --- LOMO ---------------------------------------------------------------------------------
    matches = sorted(df["match_id"].unique())
    global_pick = pick(df)
    print("\n=== GLOBAL PICK (all {} matches, mean spellF1) ===".format(len(matches)))
    print("  " + "  ".join("{}={}".format(p, global_pick[p]) for p in PARAMS))
    g_rows = df
    for p, v in global_pick.items():
        g_rows = g_rows[g_rows[p] == v]
    print("  mean spellF1 {:.4f}  macroF1 {:.4f}".format(
        g_rows["spell_f1"].mean(), g_rows["macro_f1"].mean()))

    cur_rows = df
    for p, v in CURRENT.items():
        cur_rows = cur_rows[cur_rows[p] == v]
    print("  incumbent config: mean spellF1 {:.4f}  macroF1 {:.4f}".format(
        cur_rows["spell_f1"].mean(), cur_rows["macro_f1"].mean()))

    print("\n=== LOMO (select on 6, report on the held-out match) ===")
    folds = []
    for held in matches:
        train = df[df["match_id"] != held]
        fold_pick = pick(train)
        test = df[df["match_id"] == held]
        fold_row = test
        for p, v in fold_pick.items():
            fold_row = fold_row[fold_row[p] == v]
        glob_row = test
        for p, v in global_pick.items():
            glob_row = glob_row[glob_row[p] == v]
        cur_row = test
        for p, v in CURRENT.items():
            cur_row = cur_row[cur_row[p] == v]
        row = {"held_out": held,
               "fold_pick": "/".join("{}={:g}".format(p, fold_pick[p]) for p in PARAMS),
               "fold_score": float(fold_row["spell_f1"].iloc[0]),
               "global_score": float(glob_row["spell_f1"].iloc[0]),
               "incumbent_score": float(cur_row["spell_f1"].iloc[0])}
        folds.append(row)
        print("  {}: fold {:.4f} | global {:.4f} | incumbent {:.4f}   [{}]"
              .format(held, row["fold_score"], row["global_score"],
                      row["incumbent_score"], row["fold_pick"]))

    folds_df = pd.DataFrame(folds)
    print("\n  held-out mean: fold-pick {:.4f} | global-pick {:.4f} | incumbent {:.4f}".format(
        folds_df["fold_score"].mean(), folds_df["global_score"].mean(),
        folds_df["incumbent_score"].mean()))

    # --- report ---------------------------------------------------------------------------------
    top = (df.groupby(PARAMS)[["spell_f1", "macro_f1"]].mean()
           .reset_index().sort_values("spell_f1", ascending=False).head(12))
    lines = [
        "# IDSSE LOMO selection{} (spellF1 on IDSSE labels, challenge GT untouched)".format(
            " -- DEGRADED: ATD ball gaps injected (C2, seed {})".format(GAP_SEED)
            if args.degraded else ""),
        "",
        "- grid: {} cells x 7 matches; metric: spellF1 mean over IoU 0.3..0.7".format(len(cells)),
        "- current notebook config held-out mean: {:.4f} spellF1".format(
            folds_df["incumbent_score"].mean()),
        "- global pick held-out mean: {:.4f} spellF1".format(folds_df["global_score"].mean()),
        "- fold-pick held-out mean: {:.4f} spellF1 (LOMO generalization)".format(
            folds_df["fold_score"].mean()),
        "",
        "**global pick**: " + "  ".join("{}={:g}".format(p, global_pick[p]) for p in PARAMS),
        "",
        "## LOMO detail",
        "",
        folds_df.to_markdown(index=False) if _has_tabulate() else _md(folds_df),
        "",
        "## Top 12 cells by mean spellF1",
        "",
        _md(top),
        "",
        "Closing step: gt_free_diagnostics.py vs IDSSE. Public-GT scoreboard = final report only.",
    ]
    out_md = OUT_MD_DEG if args.degraded else OUT_MD
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote {}".format(out_md))
    return 0


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _md(frame):
    def fmt(v):
        return "{:.4f}".format(v) if isinstance(v, float) else str(v)
    cols = list(frame.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in frame.iterrows():
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
