"""GT-free diagnostics: score-free health checks on our own predictions + IDSSE reference.

Why this exists: the possession ground truth must NOT be used to train, tune or calibrate
(organizer clarification, 2026-11). This module is the allowed alternative -- everything it
prints is either (a) computed from our own predictions and the unlabeled tracking of the match,
or (b) computed from IDSSE (external matches, algorithmic labels), which is the sanctioned
reference bed.

Part A (always) -- our predictions (evaluation/results/viterbi.csv) on the BAR-ATH tracking:
  1. ball coordinate ranges (origin check: corner 0..pitch vs centred) -- request (c);
  2. attack-direction check per half (territorial channel sign) -- request (d3);
  3. ball_state / ball_score provenance & confidence columns -- request (f);
  4. predicted D share + D-run duration distribution;
  5. possessions per minute + mean possession length (prediction structure);
  6. fraction of predicted-D frames with ball IN FIELD and FAST (should be low) -- diag (3);
  7. fraction of team switches with NO prior control episode by the new team -- diag (3).

Part B (IDSSE available) -- reference ranges from the external bed, same definitions:
  D share, D-run durations, possessions/min, mean possession length, and the raw material to
  recalibrate the D channel without the challenge GT (frac_stationary and team speed by label).

Definitions (kept identical across parts so the numbers are comparable):
  * playing frame  = a frame the features parquet covers (both halves, no halftime);
  * possession     = maximal run of one team (H or A); D splits possessions;
  * control episode= ball available AND d_min(team) < 2.0 m AND ball speed < 7.0 m/s
                     (ball not in flight) -- our operational definition of effective control;
  * ball fast      = ball speed > 4.0 m/s (above this, per the design notes, play is near-certain);
  * fps            = 25 (repo constant; idsse.py also assumes 25 when deriving speeds).

Usage:
    python evaluation/gt_free_diagnostics.py            # part A only if IDSSE cache missing
    python evaluation/gt_free_diagnostics.py --idsse    # force part B attempt (downloads)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import games  # noqa: E402

FPS = games.GAMES[games.GAME]["fps"]
PITCH_L, PITCH_W = games.GAMES[games.GAME]["pitch"]
SUB_PATH = os.path.join(HERE, "results", "viterbi.csv")
FEATURES_PATH = os.path.join(
    os.environ.get("MTAH_DATA_DIR") or r"D:\Proyectos\Barcelona project local\data",
    "processed", "{}_atd_ball_features_LA.parquet".format(games.GAME))

CONTROL_DIST_M = 2.0        # d_min below this = candidate control
FLIGHT_SPEED_MPS = 7.0      # above this the ball is in flight -> nobody controls it
FAST_BALL_MPS = 4.0         # ball this fast inside the field = play is near-certain
SWITCH_LOOKBACK_FRAMES = 50  # 2 s: how far back a control episode may precede a switch
MARGIN_M = 0.35             # same out-of-bounds margin the notebook emission uses


def ball_speed_mps(ball_ok, bx, by, fps):
    """Ball speed only between consecutive frames that both have a ball.

    nan->0 plus np.gradient fabricates spikes at the edges of tracking gaps (21% of our frames
    have no ball; IDSSE has none), and those spikes trip the 'fast' gate right where dead-ball
    detection matters. Unknown speed -> NaN, which is neither slow nor fast downstream.
    """
    ball_ok = np.asarray(ball_ok, dtype=bool)
    bx = np.nan_to_num(np.asarray(bx, dtype=float), nan=0.0)
    by = np.nan_to_num(np.asarray(by, dtype=float), nan=0.0)
    valid = ball_ok.copy()
    valid[1:] &= ball_ok[:-1]
    dx = np.zeros(len(bx))
    dy = np.zeros(len(by))
    dx[1:] = bx[1:] - bx[:-1]
    dy[1:] = by[1:] - by[:-1]
    return np.where(valid, np.hypot(dx, dy) * fps, np.nan)


def run_length_stats(durations_frames):
    if not len(durations_frames):
        return "n=0"
    d = np.asarray(durations_frames, dtype=float) / FPS
    return ("n={:d}  mean {:.1f}s  p50 {:.1f}s  p90 {:.1f}s  max {:.1f}s  <1s {:.0%}  <2s {:.0%}"
            .format(len(d), d.mean(), np.median(d), np.percentile(d, 90), d.max(),
                    (d < 1.0).mean(), (d < 2.0).mean()))


def runs_of(labels, values):
    """Durations (in frames) of maximal runs whose label is in `values`."""
    out, cur = [], 0
    for lab in labels:
        if lab in values:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def possession_stats(labels):
    """(n_possessions, possession_changes, playing_minutes, mean_possession_s, team_runs).

    A possession is a maximal run of a SINGLE team label -- ended by a dead-ball frame OR by a
    team change, the same segmentation as possession_metrics.possession_spells. (An earlier
    version only split on D, which merged H->A changes inside alive play and reported
    'possessions == D runs'; fixed after the IDSSE numbers refused to add up.)
    A possession change is a consecutive pair of possessions held by different teams, so a
    restart that resumes with the same team is not a change.
    X (abstention / not in match) is its own label and therefore splits segments.
    """
    labels = list(labels)
    segs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segs.append((start, i, labels[start]))
            start = i
    team = [(a, b, k) for a, b, k in segs if k in ("H", "A")]
    n_poss = len(team)
    changes = sum(1 for (_, _, a), (_, _, b) in zip(team, team[1:]) if a != b)
    minutes = len(labels) / FPS / 60.0
    lengths = [b - a for a, b, _ in team]
    mean_s = float(np.mean(lengths)) / FPS if lengths else 0.0
    return n_poss, changes, minutes, mean_s, [k for _, _, k in team]


def load_predictions(features_path, submission=None):
    """Frame->label map from the notebook decoder (default) or an existing submission CSV.

    Default path keeps the iteration loop GT-free: the notebook cell is exec'd, run on the
    features parquet, and the diagnostics score those predictions -- no ground truth anywhere.
    """
    if submission:
        sub = pd.read_csv(submission)
        return dict(zip(sub["frame"], sub["label"]))
    from run_viterbi_vs_gt import load_decoder_from_notebook
    ns = load_decoder_from_notebook()
    feat = pd.read_parquet(features_path)
    hmm = ns["PossessionViterbiHMM"](ns["ViterbiPossessionConfig"]())
    df_v, summary = hmm.predict(feat)
    print("decoder summary (from notebook): " + "  ".join(
        "{}={}".format(k, v) for k, v in summary.items()))
    return dict(zip(df_v["video_frame"], df_v["pred_state"]))


def part_a(sub_path=SUB_PATH, features_path=FEATURES_PATH, submission=None):
    print("=" * 78)
    print("PART A: our predictions, no GT involved")
    print("=" * 78)
    feat = pd.read_parquet(features_path)
    lab = load_predictions(features_path, submission)
    pred = np.array([lab.get(int(v), "X") for v in feat["video_frame"]], dtype=object)
    pred = np.where(pred == "L", "H", pred)  # notebook alphabet -> eval alphabet

    # --- (c) coordinate origin ---------------------------------------------------------------
    print("\n-- (c) ball coordinate ranges (ball_available only) --")
    for period, g in feat[feat["ball_available"]].groupby("period"):
        bx, by = g["ball_x_m"], g["ball_y_m"]
        print("  {}: x  min {:+8.2f}  p1 {:+8.2f}  p50 {:+8.2f}  p99 {:+8.2f}  max {:+8.2f}   "
              "outside [{:.2f},{:.2f}]: {:.3%}".format(
                  period, bx.min(), bx.quantile(0.01), bx.median(), bx.quantile(0.99), bx.max(),
                  -MARGIN_M, PITCH_L + MARGIN_M,
                  ((bx < -MARGIN_M) | (bx > PITCH_L + MARGIN_M)).mean()))
        print("      {} y  min {:+8.2f}  p1 {:+8.2f}  p50 {:+8.2f}  p99 {:+8.2f}  max {:+8.2f}   "
              "outside [{:.2f},{:.2f}]: {:.3%}".format(
                  " " * len(period), by.min(), by.quantile(0.01), by.median(),
                  by.quantile(0.99), by.max(), -MARGIN_M, PITCH_W + MARGIN_M,
                  ((by < -MARGIN_M) | (by > PITCH_W + MARGIN_M)).mean()))
    print("  -> corner-origin (0..pitch) keeps the notebook's OOB test valid; centred "
          "coordinates would flag everything.")

    # --- (d3) attack direction ----------------------------------------------------------------
    print("\n-- (d3) territorial sign: median centroid_x by team and half --")
    for period, g in feat.groupby("period"):
        cx_l, cx_a = g["centroid_x_L"].median(), g["centroid_x_A"].median()
        print("  {}: centroid_x_L {:+7.2f}  centroid_x_A {:+7.2f}  L-A {:+7.2f}  "
              "-> L attacks {}".format(period, cx_l, cx_a, cx_l - cx_a,
                                       "+x" if cx_l > cx_a else "-x"))
    print("  -> if the sign flips between halves, any territorial term must be re-oriented "
          "per half (the redesign removes the channel entirely).")

    # --- (f) ball provenance ------------------------------------------------------------------
    print("\n-- (f) ball_state / ball_score columns --")
    print("  ball_state x ball_available:")
    print(pd.crosstab(feat["ball_state"], feat["ball_available"]).to_string())
    print("  ball_score by state:")
    print(feat.groupby("ball_state")["ball_score"].describe(
        percentiles=[0.05, 0.5, 0.95]).to_string())

    # --- prediction structure -----------------------------------------------------------------
    playing = pred != "X"
    labels_playing = pred[playing]
    d_runs = runs_of(labels_playing, ("D",))
    n_poss, changes, minutes, mean_poss, _ = possession_stats(pred)  # X splits segments
    print("\n-- (4) predicted dead ball --")
    print("  D share of playing frames: {:.2%}".format((labels_playing == "D").mean()))
    print("  D runs: {}".format(run_length_stats(d_runs)))
    print("\n-- (5) predicted possession structure --")
    print("  possessions {:d}   possession changes {:d}   playing {:.1f} min".format(
        n_poss, changes, minutes))
    print("  changes/min {:.2f}   mean possession {:.1f}s (IDSSE reference below)".format(
        changes / minutes, mean_poss))

    # --- ball-speed sanity on predicted D ------------------------------------------------------
    ball_ok = feat["ball_available"].to_numpy(dtype=bool)
    bspd = ball_speed_mps(ball_ok, feat["ball_x_m"].to_numpy(dtype=float),
                          feat["ball_y_m"].to_numpy(dtype=float), FPS)
    in_field = ((feat["ball_x_m"].to_numpy(dtype=float) > -MARGIN_M)
                & (feat["ball_x_m"].to_numpy(dtype=float) < PITCH_L + MARGIN_M)
                & (feat["ball_y_m"].to_numpy(dtype=float) > -MARGIN_M)
                & (feat["ball_y_m"].to_numpy(dtype=float) < PITCH_W + MARGIN_M))
    is_d = labels_playing == "D"
    d_idx = np.flatnonzero(is_d)
    fast_d = ball_ok[d_idx] & in_field[d_idx] & (bspd[d_idx] > FAST_BALL_MPS)
    print("\n-- (6) predicted-D frames with ball in field and fast (> {:.1f} m/s) --".format(
        FAST_BALL_MPS))
    print("  {:.2%} of predicted D frames  ({:d} of {:d})".format(
        fast_d.mean(), int(fast_d.sum()), len(d_idx)))
    print("  [reference: should be low -- a fast in-field ball means the ball is in play]")

    # --- D-channel gate rates (fixed constants, identical formulas in part C on IDSSE) --------
    rm = lambda a, w: pd.Series(a).rolling(w, min_periods=1).mean().to_numpy()
    slow_sust = rm((ball_ok & np.isfinite(bspd) & (bspd < 2.5)).astype(float), 50) >= 0.6
    fast_sust = rm((in_field & np.isfinite(bspd) & (bspd > 4.0)).astype(float), 12) >= 0.5
    missing_sust = rm((~ball_ok).astype(float), 50) >= 0.7
    mean_spd = 0.5 * (feat["mean_speed_L"].to_numpy(dtype=float)
                      + feat["mean_speed_A"].to_numpy(dtype=float))
    roll = pd.Series(mean_spd).rolling(7500, center=True, min_periods=375).median().to_numpy()
    slow_rel = np.where(np.isfinite(mean_spd) & np.isfinite(roll),
                        np.clip((roll - mean_spd) / np.maximum(roll, 1.0), 0.0, 1.0), 0.0)
    frac = 0.5 * (feat["frac_stationary_L"].fillna(0).to_numpy()
                  + feat["frac_stationary_A"].fillna(0).to_numpy())
    players_slow = np.maximum(slow_rel, np.clip((frac - 0.15) / 0.85, 0.0, 1.0))
    stop = slow_sust * players_slow
    print("\n-- D-channel gate rates on our match (same formulas as part C on IDSSE) --")
    print("  slow_sust(2s,<2.5) {:6.2%}   players_slow>0.3 {:6.2%}   stop_evid>0.4 {:6.2%}".format(
        slow_sust.mean(), (players_slow > 0.3).mean(), (stop > 0.4).mean()))
    print("  fast_sust(0.5s,>4) {:6.2%}   missing_sust(2s) {:6.2%}".format(
        fast_sust.mean(), missing_sust.mean()))

    # --- switches without a prior control episode ----------------------------------------------
    d_min_l = feat["d_min_L"].to_numpy(dtype=float)
    d_min_a = feat["d_min_A"].to_numpy(dtype=float)
    ctrl_h = ball_ok & np.isfinite(d_min_l) & (d_min_l < CONTROL_DIST_M) & (bspd < FLIGHT_SPEED_MPS)
    ctrl_a = ball_ok & np.isfinite(d_min_a) & (d_min_a < CONTROL_DIST_M) & (bspd < FLIGHT_SPEED_MPS)
    ctrl = {"H": ctrl_h, "A": ctrl_a}

    switch_idx = [i for i in range(1, len(pred))
                  if pred[i] in ("H", "A") and pred[i - 1] in ("H", "A") and pred[i] != pred[i - 1]]
    no_prior = 0
    for i in switch_idx:
        new = pred[i]
        window = ctrl[new][max(0, i - SWITCH_LOOKBACK_FRAMES):i]
        if not window.any():
            no_prior += 1
    print("\n-- (7) team switches without a prior control episode by the new team --".format())
    print("  {} of {} switches ({:.1%}) had no control episode (d_min < {:.1f} m, ball not in "
          "flight) in the previous {} frames".format(
              no_prior, len(switch_idx),
              no_prior / len(switch_idx) if switch_idx else 0.0,
              CONTROL_DIST_M, SWITCH_LOOKBACK_FRAMES))
    print("  [reference: should be low -- possession should change on effective control]")
    return {"d_share": float((labels_playing == "D").mean()),
            "d_runs": d_runs, "possessions": n_poss, "possession_changes": changes,
            "minutes": minutes, "mean_possession_s": mean_poss,
            "fast_d_frac": float(fast_d.mean()), "no_control_switch_frac":
            no_prior / len(switch_idx) if switch_idx else 0.0}


def part_b(only_cached=True):
    print("\n" + "=" * 78)
    print("PART B: IDSSE external reference (algorithmic labels, no challenge GT)")
    print("=" * 78)
    from tuning import idsse

    cache_dir = os.path.join(ROOT, "data", ".cache")
    cached = {f[len("idsse_"):-len(".pkl")] for f in
              (os.listdir(cache_dir) if os.path.isdir(cache_dir) else [])
              if f.startswith("idsse_") and f.endswith(".pkl")}

    rows, d_runs_all, frac_by_label, spd_by_label = [], [], [], []
    for mid in idsse.MATCH_IDS:
        if mid not in cached and only_cached:
            print("  {}: cache not ready -- skipped (background: python -m tuning.idsse)"
                  .format(mid))
            continue
        try:
            case = idsse.load_match(mid)
        except Exception as exc:  # missing data / network -- report and continue
            print("  {}: unavailable ({})".format(mid, type(exc).__name__))
            continue
        letters = case.letters
        in_match = letters != "X"
        lab = letters[in_match]
        n_poss, switches, minutes, mean_poss, _ = possession_stats(lab)
        d_runs = runs_of(lab, ("D",))
        d_runs_all.extend(d_runs)
        d_share = float((lab == "D").mean())
        rows.append({"match": mid, "frames": int(in_match.sum()), "D_share": d_share,
                     "D_runs": len(d_runs), "possessions": n_poss,
                     "changes_per_min": switches / minutes, "mean_poss_s": mean_poss})

        # raw material for D-channel recalibration: player-speed features by label.
        # provider `players_s` is in km/h (verified: p50 5.05 -> 1.40 m/s, matching a
        # gradient-based m/s recomputation with median 1.46) -- convert before use.
        speed = case.players_s / 3.6
        with np.errstate(invalid="ignore"):
            stat = speed < 0.65  # same stationary threshold as atdProcess.stationary_speed_mps
        vis = np.isfinite(speed).sum(axis=1)
        frac_stat = np.where(vis > 0, stat.sum(axis=1) / np.maximum(vis, 1), np.nan)
        mean_spd = np.nanmean(np.where(np.isfinite(speed), speed, np.nan), axis=1)
        for name in ("D", "H", "A"):
            m = lab == name
            if m.sum():
                frac_by_label.append((name, float(np.nanmedian(frac_stat[in_match][m]))))
                spd_by_label.append((name, float(np.nanmedian(mean_spd[in_match][m]))))
        print("  {}: D {:.1%}  possessions {:d}  changes/min {:.2f}  mean poss {:.1f}s  "
              "D runs {}".format(mid, d_share, n_poss, switches / minutes, mean_poss,
                                 run_length_stats(d_runs)))

    if not rows:
        print("  no IDSSE matches available -- part B skipped (run: python -m tuning.idsse)")
        return None
    agg = pd.DataFrame(rows)
    print("\n  AGGREGATE over {} matches:".format(len(agg)))
    print("    D share            : mean {:.1%}  (per match: {})".format(
        agg["D_share"].mean(), ", ".join("{:.0%}".format(v) for v in agg["D_share"])))
    print("    D runs             : {}".format(run_length_stats(d_runs_all)))
    print("    possessions/min    : mean {:.2f}  (per match: {})".format(
        agg["changes_per_min"].mean(),
        ", ".join("{:.2f}".format(v) for v in agg["changes_per_min"])))
    print("    mean possession    : mean {:.1f}s".format(agg["mean_poss_s"].mean()))
    frac_df = pd.DataFrame(frac_by_label, columns=["label", "frac_stationary_median"])
    spd_df = pd.DataFrame(spd_by_label, columns=["label", "mean_speed_median"])
    print("\n  D-channel calibration inputs (per-label medians, pooled over matches):")
    print("    frac_stationary: " + "  ".join(
        "{} {:.3f}".format(k, v) for k, v in
        frac_df.groupby("label")["frac_stationary_median"].mean().items()))
    print("    team mean_speed: " + "  ".join(
        "{} {:.2f} m/s".format(k, v) for k, v in
        spd_df.groupby("label")["mean_speed_median"].mean().items()))
    return agg


def part_c():
    """Gate ceilings from IDSSE: how often the emission's gates actually open per label.

    This is the calibration view for thresholds: a gate that never opens on real dead ball can
    never detect D, no matter what the transition matrix does -- and a control gate that never
    opens during alive play can never produce a possession change. Everything here is computed
    from IDSSE external tracking + algorithmic labels; the challenge GT is not opened.
    """
    print("\n" + "=" * 78)
    print("PART C: emission gate ceilings on IDSSE (where the redesign's doors open)")
    print("=" * 78)
    from tuning import idsse

    cache_dir = os.path.join(ROOT, "data", ".cache")
    cached = {f[len("idsse_"):-len(".pkl")] for f in
              (os.listdir(cache_dir) if os.path.isdir(cache_dir) else [])
              if f.startswith("idsse_") and f.endswith(".pkl")}
    if not cached:
        print("  no IDSSE cache -- skipped")
        return

    pit = np.array([games.GAMES[games.GAME]["pitch"][0], games.GAMES[games.GAME]["pitch"][1]])
    agg, stop_rates = [], []
    for mid in sorted(cached):
        case = idsse.load_match(mid)
        lab = case.letters
        ball = case.ball * pit                       # normalised -> metres
        steps = np.diff(ball, axis=0, prepend=ball[:1]) * FPS
        bspd = np.hypot(steps[:, 0], steps[:, 1])
        ball_ok = np.isfinite(ball).all(axis=1)
        bspd = np.where(ball_ok, bspd, np.nan)

        # the notebook's own dead-ball detector, run on IDSSE: this -- not the provider's D
        # label -- is the apples-to-apples D-share target for a physics/AND design
        speed = case.players_s / 3.6                 # km/h -> m/s (verified in part B)
        team_l = np.nanmean(np.where(case.slot_team[None, :] == 0, speed, np.nan), axis=1)
        team_a = np.nanmean(np.where(case.slot_team[None, :] == 1, speed, np.nan), axis=1)
        mean_spd = 0.5 * (team_l + team_a)
        roll5 = (pd.Series(mean_spd).rolling(7500, center=True, min_periods=375)
                 .median().to_numpy())
        slow_rel = np.where(np.isfinite(mean_spd) & np.isfinite(roll5),
                            np.clip((roll5 - mean_spd) / np.maximum(roll5, 1.0), 0.0, 1.0), 0.0)
        stat_t = speed < 0.65
        frac_team = 0.5 * (np.nanmean(stat_t[:, case.slot_team == 0], axis=1)
                           + np.nanmean(stat_t[:, case.slot_team == 1], axis=1))
        players_slow = np.maximum(slow_rel, np.clip((frac_team - 0.15) / 0.85, 0.0, 1.0))

        players = case.players * pit                 # (N, K, 2) metres
        d = np.linalg.norm(players - ball[:, None, :], axis=-1)  # (N, K)
        d = np.where(np.isfinite(d) & ball_ok[:, None], d, np.nan)
        d_home = np.nanmin(np.where(case.slot_team[None, :] == 0, d, np.nan), axis=1)
        d_away = np.nanmin(np.where(case.slot_team[None, :] == 1, d, np.nan), axis=1)

        slow25 = ball_ok & (np.nan_to_num(bspd, nan=99.0) < 2.5)
        slow30 = ball_ok & (np.nan_to_num(bspd, nan=99.0) < 3.0)
        # the emission's 2s sustain: >=60% of the last 50 frames slow
        slow_sust25 = pd.Series(slow25.astype(float)).rolling(50, center=True,
                                                               min_periods=1).mean().to_numpy() >= 0.6
        fast4 = ball_ok & (np.nan_to_num(bspd, nan=0.0) > 4.0)
        missing = ~ball_ok
        # anchor with the CURRENT detector formula (ball-dominant floor from the notebook's
        # dead_players_floor, not the original pure product) so the anchor matches the model
        stop_rates.append((mid, float((slow_sust25 * (0.4 + 0.6 * players_slow) > 0.4).mean())))

        for name in ("D", "H", "A"):
            m = lab == name
            if not m.sum():
                continue
            agg.append({
                "label": name, "match": mid, "n": int(m.sum()),
                "ball_slow25": float(slow25[m].mean()),
                "ball_slow30": float(slow30[m].mean()),
                "slow_sust25_2s": float(slow_sust25[m].mean()),
                "ball_fast4": float(fast4[m].mean()),
                "ball_missing": float(missing[m].mean()),
            })

        # control-gate ceilings on alive frames (both labels pooled per match)
        alive = np.isin(lab, ("H", "A"))
        bfast = np.nan_to_num(bspd, nan=0.0)
        solo = alive & ball_ok & (bfast < 7.0) & (
            ((d_home < 2.0) & (d_away >= 2.0)) | ((d_away < 2.0) & (d_home >= 2.0)))
        duel = alive & ball_ok & (bfast < 7.0) & (d_home < 2.0) & (d_away < 2.0)
        loose = alive & ball_ok & (bfast < 7.0) & (d_home >= 2.0) & (d_away >= 2.0)
        agg.append({"label": "ALIVE_GATE", "match": mid, "n": int(alive.sum()),
                    "ball_slow25": float(solo.mean()),      # reuse cols: solo/duel/loose
                    "ball_slow30": float(duel.mean()),
                    "slow_sust25_2s": float(loose.mean()),
                    "ball_fast4": float(fast4[alive].mean()),
                    "ball_missing": float(missing[alive].mean())})

    df = pd.DataFrame(agg)
    print("\n-- ball gates by label (pooled mean over matches) --")
    by = df[df["label"] != "ALIVE_GATE"].groupby("label")[
        ["ball_slow25", "ball_slow30", "slow_sust25_2s", "ball_fast4", "ball_missing"]].mean()
    print(by.to_string(float_format=lambda v: "{:.3f}".format(v)))
    print("  (rows: D=dead, H/A=alive; slow_sust25_2s = the emission's 2s slow gate)")
    print("\n-- control gate on ALIVE frames (solo | duel | loose), pooled --")
    alive_rows = df[df["label"] == "ALIVE_GATE"]
    print("  solo(one team <2m, other >=2m, ball<7m/s): {:.1%}   duel(both <2m): {:.1%}   "
          "loose(both >=2m): {:.1%}   ball fast>4: {:.1%}   missing: {:.1%}".format(
              alive_rows["ball_slow25"].mean(), alive_rows["ball_slow30"].mean(),
              alive_rows["slow_sust25_2s"].mean(), alive_rows["ball_fast4"].mean(),
              alive_rows["ball_missing"].mean()))
    print("  -> solo is the switch-evidence ceiling: if it is rare, no beta can reach the")
    print("     IDSSE change rate with a control-gated emission; the gate itself must widen.")
    print("\n-- physics-dead rate: the notebook's own detector run on IDSSE --")
    for mid, rate in stop_rates:
        print("  {}: {:6.2%}".format(mid, rate))
    vals = [r for _, r in stop_rates]
    if vals:
        print("  RANGE {:.1%} .. {:.1%}   mean {:.1%}   <- D-share target for an AND design"
              .format(min(vals), max(vals), float(np.mean(vals))))

    # Design-compatible possession-change rate (PROVENANCE of the tuner's target range):
    # provider H<->A flips whose NEW owner has our control gate open within +/-20 frames.
    # Raw provider rate counts flips during flight/duels -- moments the design assigns
    # "no evidence, keep the last team" by specification. This is IDSSE-only; the challenge
    # GT's possession count has never been used to derive it.
    print("\n-- design-compatible possession-change rate (IDSSE-only provenance) --")
    compat_rows = []
    for mid in sorted(cached):
        case = idsse.load_match(mid)
        lab = case.letters
        ball = case.ball * pit
        steps = np.diff(ball, axis=0, prepend=ball[:1]) * FPS
        bspd = np.hypot(steps[:, 0], steps[:, 1])
        ball_ok = np.isfinite(ball).all(axis=1)
        players = case.players * pit
        d = np.linalg.norm(players - ball[:, None, :], axis=-1)
        d = np.where(np.isfinite(d) & ball_ok[:, None], d, np.nan)
        d_home = np.nanmin(np.where(case.slot_team[None, :] == 0, d, np.nan), axis=1)
        d_away = np.nanmin(np.where(case.slot_team[None, :] == 1, d, np.nan), axis=1)
        flight = ~(ball_ok & np.isfinite(bspd) & (bspd <= 7.0))
        clear_h = (d_away >= 2.0) | ((d_away < 2.0) & ((d_away - d_home) >= 0.7))
        clear_a = (d_home >= 2.0) | ((d_home < 2.0) & ((d_home - d_away) >= 0.7))
        solo_h = ball_ok & ~flight & (d_home < 2.0) & clear_h
        solo_a = ball_ok & ~flight & (d_away < 2.0) & clear_a
        gate = np.where(solo_h & ~solo_a, "H", np.where(solo_a & ~solo_h, "A", None))
        gate = pd.Series(gate).groupby(case.half).ffill().to_numpy()
        flips = [(i, lab[i]) for i in range(1, len(lab))
                 if lab[i] in ("H", "A") and lab[i - 1] in ("H", "A") and lab[i] != lab[i - 1]]
        compat = sum(1 for i, new in flips
                     if any(gate[j] == new for j in range(max(0, i - 20),
                                                          min(len(gate), i + 20))))
        minutes = len(lab) / FPS / 60.0
        compat_rows.append((mid, len(flips), compat / minutes))
        print("  {}: raw {:5.2f}/min   design-compatible {:5.2f}/min".format(
            mid, len(flips) / minutes, compat / minutes))
    cm = [r[2] for r in compat_rows]
    print("  RANGE {:.2f} .. {:.2f}  mean {:.2f}  <- tuner's changes/min feasibility range"
          .format(min(cm), max(cm), float(np.mean(cm))))


def part_d():
    """Self-consistency diagnostics for the H/A bias, on our own predictions (no GT).

    D1 absorption per team -- hypothesis: min_possession_frames=40 eats short A possessions
       (L-A-L sandwiches) more than L ones, because the recovering team's claims are brief.
    D2 zero-evidence share -- hypothesis: duel_gap=0.7 leaves large stretches with no evidence
       for either side, so the incumbent keeps the state by default.
    D3 symmetry test -- swap L/A features and decode: the output must be the exact mirror.
       Any mismatch is a hidden code asymmetry (np.argmax tie-breaking favours index 0).
    D4 tracking asymmetry -- if one team is seen less, its d_min is systematically larger and
       its controls are lost; that is data, not code, and shows up here.
    """
    from run_viterbi_vs_gt import load_decoder_from_notebook

    print("\n" + "=" * 78)
    print("PART D: H/A-bias self-consistency (no GT)")
    print("=" * 78)
    ns = load_decoder_from_notebook()
    cfg = ns["ViterbiPossessionConfig"]()
    hmm = ns["PossessionViterbiHMM"](cfg)
    cols = ["video_frame", "time_s", "period", "ball_available", "ball_score",
            "ball_x_m", "ball_y_m", "d_min_L", "d_min_A",
            "mean_speed_L", "mean_speed_A", "frac_stationary_L", "frac_stationary_A",
            "n_visible_L", "n_visible_A"]
    feat = pd.read_parquet(FEATURES_PATH, columns=cols)

    # --- decode raw (pre-absorb) per period, then both real and instrumented absorb ---------
    raw_paths, final_paths, period_frames = {}, {}, []
    absorbed = {"L": [0, 0], "A": [0, 0], "D": [0, 0], "DLD": [0, 0]}  # [runs, frames]
    for period, df_p in feat.groupby("period", sort=False):
        df_p = df_p.sort_values("video_frame").reset_index(drop=True)
        period_frames.append(df_p["video_frame"].to_numpy())
        log_emit = hmm._compute_log_emissions(df_p)
        path_idx, _ = hmm._viterbi_decode(log_emit)
        final = hmm._absorb_short_segments(path_idx)
        # instrumented copy of the filter: must reproduce `final` exactly (asserted below)
        inst = path_idx.copy()
        for _ in range(3):
            runs, start = [], 0
            for i in range(1, len(inst)):
                if inst[i] != inst[start]:
                    runs.append([start, i, int(inst[start])])
                    start = i
            runs.append([start, len(inst), int(inst[start])])
            changed = False
            for r_i in range(1, len(runs) - 1):
                s, e, st = runs[r_i]
                dur = e - s
                prev_st, next_st = runs[r_i - 1][2], runs[r_i + 1][2]
                name = {0: "L", 1: "A", 2: "D"}.get(st)
                if st == hmm.state_to_idx["D"] and dur < cfg.min_dead_frames:
                    absorbed["D"][0] += 1
                    absorbed["D"][1] += dur
                    if prev_st == next_st:
                        inst[s:e] = prev_st
                    else:
                        mid = s + dur // 2
                        inst[s:mid] = prev_st
                        inst[mid:e] = next_st
                    changed = True
                elif st in (hmm.state_to_idx["L"], hmm.state_to_idx["A"]):
                    if (dur < cfg.min_possession_frames
                            and prev_st == next_st
                            and prev_st in (hmm.state_to_idx["L"],
                                            hmm.state_to_idx["A"])):
                        absorbed[name][0] += 1
                        absorbed[name][1] += dur
                        inst[s:e] = prev_st
                        changed = True
                    elif (dur < cfg.min_d_poss_frames
                            and prev_st == hmm.state_to_idx["D"]
                            and next_st == hmm.state_to_idx["D"]):
                        # C3 rule, mirrored from the notebook cell (asserted equal below)
                        absorbed["DLD"][0] += 1
                        absorbed["DLD"][1] += dur
                        inst[s:e] = hmm.state_to_idx["D"]
                        changed = True
            if not changed:
                break
        assert np.array_equal(inst, final), "instrumented filter diverged from the real one"
        raw_paths[period] = path_idx
        final_paths[period] = final

    print("\n-- D1: absorbed by the morphological filter (whole match) --")
    for name in ("L", "A", "D", "DLD"):
        runs_n, frames_n = absorbed[name]
        print("  {:>3}: {:4d} runs absorbed, {:6d} frames ({:.1f} min of play)".format(
            name, runs_n, frames_n, frames_n / FPS / 60.0))
    l_frames, a_frames = absorbed["L"][1], absorbed["A"][1]
    if min(l_frames, a_frames) > 0:
        print("  A/L absorption ratio: {:.2f}  (>1 means A's short claims are eaten more)"
              .format(a_frames / max(1, l_frames)))

    # --- D2: evidence classification on in-play frames ---------------------------------------
    ball_ok = feat["ball_available"].to_numpy(dtype=bool)
    bspd = ball_speed_mps(ball_ok, feat["ball_x_m"].to_numpy(dtype=float),
                          feat["ball_y_m"].to_numpy(dtype=float), FPS)
    ema = ns["_bidirectional_ema"]
    d_L = _ema_valid(feat["d_min_L"].to_numpy(dtype=float), ball_ok, ema)
    d_A = _ema_valid(feat["d_min_A"].to_numpy(dtype=float), ball_ok, ema)
    flying = (~ball_ok) | ~np.isfinite(bspd) | (bspd > cfg.flight_speed_mps)
    in_play = ~flying
    clear_L = (d_A >= cfg.control_radius_m) | ((d_A < cfg.control_radius_m)
                                               & ((d_A - d_L) >= cfg.duel_gap_m))
    clear_A = (d_L >= cfg.control_radius_m) | ((d_L < cfg.control_radius_m)
                                               & ((d_L - d_A) >= cfg.duel_gap_m))
    solo_L = in_play & (d_L < cfg.control_radius_m) & clear_L
    solo_A = in_play & (d_A < cfg.control_radius_m) & clear_A
    duel = in_play & (d_L < cfg.control_radius_m) & (d_A < cfg.control_radius_m) & ~(clear_L | clear_A)
    loose = in_play & (d_L >= cfg.control_radius_m) & (d_A >= cfg.control_radius_m)

    n_play = int(in_play.sum())
    print("\n-- D2: evidence share of in-play frames (ball_ok, not flying) n={} --".format(n_play))
    for name, m in [("solo_L (clear control L)", solo_L), ("solo_A (clear control A)", solo_A),
                    ("duel zone (zero evidence)", duel), ("loose (zero evidence)", loose)]:
        print("  {:28s} {:6.2%}".format(name, m.sum() / n_play))
    zero = (duel | loose).sum() / n_play
    print("  ZERO-EVIDENCE share of in-play: {:.2%}  (incumbent keeps state there)".format(zero))
    print("  flying/missing share of all frames: {:.2%}".format(flying.mean()))

    # after-flight landings: where does the ball arrive when a flight ends?
    was_flying = np.roll(flying, 1)
    was_flying[0] = True
    landing = (~flying) & was_flying
    n_land = int(landing.sum())
    if n_land:
        print("  after-flight landings: {}  -> duel zone {:.1%} | loose {:.1%} | solo {:.1%}"
              .format(n_land, (landing & duel).sum() / n_land, (landing & loose).sum() / n_land,
                      (landing & (solo_L | solo_A)).sum() / n_land))

    # --- D3: L/A symmetry test ----------------------------------------------------------------
    swapped = feat.copy()
    for a, b in [("d_min_L", "d_min_A"), ("mean_speed_L", "mean_speed_A"),
                 ("frac_stationary_L", "frac_stationary_A")]:
        swapped[[a, b]] = swapped[[b, a]]
    df_sw, _ = hmm.predict(swapped)
    sw_state = df_sw["pred_state"].to_numpy()
    mirror = np.where(sw_state == "L", "A", np.where(sw_state == "A", "L", sw_state))
    orig = np.concatenate([final_paths[p] for p in final_paths])
    orig_letters = np.array([hmm.STATES[i] for i in orig], dtype=object)
    frames_loop = np.concatenate(period_frames)
    assert np.array_equal(df_sw["video_frame"].to_numpy(), frames_loop), "frame order mismatch"
    mismatch = int((mirror != orig_letters).sum())
    print("\n-- D3: L/A swap symmetry --")
    print("  mismatches vs exact mirror: {} of {} frames ({:.4%})".format(
        mismatch, len(orig_letters), mismatch / len(orig_letters)))
    print("  (non-zero => hidden code asymmetry, e.g. np.argmax tie-breaking favouring L)")

    # --- D4: tracking asymmetry ----------------------------------------------------------------
    print("\n-- D4: tracking asymmetry (data, not code) --")
    nv_l = feat["n_visible_L"].to_numpy(dtype=float)
    nv_a = feat["n_visible_A"].to_numpy(dtype=float)
    print("  n_visible  L mean {:.2f}  A mean {:.2f}   |  frames with L<A: {:.1%}, L>A: {:.1%}"
          .format(np.nanmean(nv_l), np.nanmean(nv_a),
                  float((nv_l < nv_a).mean()), float((nv_l > nv_a).mean())))
    play = ball_ok & np.isfinite(d_L) & np.isfinite(d_A)
    print("  d_min with ball in play: L mean {:.2f} m  A mean {:.2f} m  (diff {:+.3f})"
          .format(np.mean(d_L[play]), np.mean(d_A[play]),
                  np.mean(d_A[play]) - np.mean(d_L[play])))
    print("  solo rate: L {:.2%}  A {:.2%}  (gap {:+.2%})"
          .format(solo_L.sum() / n_play, solo_A.sum() / n_play,
                  solo_L.sum() / n_play - solo_A.sum() / n_play))

    # --- D5: switch quality -- strict vs a-priori lax criterion, stratified by cause --------
    # strict  : new team had a control episode (d_min < 2 m, ball not flying) in the 2 s BEFORE
    # lax     : new team's d_min < 3 m somewhere within +/-1 s of the switch (a-priori, fixed
    #           before looking at any outcome)
    # causes for strict-failures (multi-label): ball missing at the switch, reception pulse
    # active, switch moved/created by the morphological filter, ball-gap (EMA) edge nearby.
    letters = np.array([hmm.STATES[i] for i in
                        np.concatenate([final_paths[p] for p in final_paths])], dtype=object)
    raw_letters = np.array([hmm.STATES[i] for i in
                            np.concatenate([raw_paths[p] for p in raw_paths])], dtype=object)
    assert len(letters) == len(feat)
    since_end = np.zeros(len(feat), dtype=np.float64)
    cnt = 0.0
    flying_all = flying
    for i in range(len(feat)):
        cnt = 0.0 if flying_all[i] else cnt + 1.0
        since_end[i] = cnt
    zero_zone = (~flying_all) & ~solo_L & ~solo_A
    ball_gap_edge = np.abs(np.diff(ball_ok.astype(np.int8), prepend=0)) > 0

    sw_idx = [i for i in range(1, len(letters))
              if letters[i] != letters[i - 1] and letters[i] in ("L", "A")
              and letters[i - 1] in ("L", "A")]
    raw_sw = [i for i in range(1, len(raw_letters))
              if raw_letters[i] != raw_letters[i - 1] and raw_letters[i] in ("L", "A")
              and raw_letters[i - 1] in ("L", "A")]
    strict_pass = lax_pass = 0
    causes = {"missing_ball": 0, "pulse_active": 0, "filter_moved": 0, "ema_gap_edge": 0}
    strict_fail = 0
    for i in sw_idx:
        new = letters[i]
        ctrl = solo_L if new == "L" else solo_A
        strict = ctrl[max(0, i - 50):i].any()
        dnew = d_L if new == "L" else d_A
        lax = (dnew[max(0, i - 25):min(len(dnew), i + 26)] < 3.0).any()
        strict_pass += bool(strict)
        lax_pass += bool(lax)
        if not strict:
            strict_fail += 1
            if not ball_ok[max(0, i - 5):i + 6].any():
                causes["missing_ball"] += 1
            if cfg.w_duel_pulse > 0 and (zero_zone[i:min(len(feat), i + 13)]
                                         & (since_end[i:min(len(feat), i + 13)] <= 12.0)).any():
                causes["pulse_active"] += 1
            if i not in raw_sw and any(abs(i - r) <= 12 for r in raw_sw):
                causes["filter_moved"] += 1
            if ball_gap_edge[max(0, i - 25):i + 26].any():
                causes["ema_gap_edge"] += 1
    n_sw = len(sw_idx)
    near_edge_rate = float(pd.Series(ball_gap_edge.astype(int))
                           .rolling(51, center=True, min_periods=1).max().mean())
    print("\n-- D5: direct team switches (n={}) --".format(n_sw))
    print("  (base rate: {:.1%} of ALL frames sit within +/-25f of a ball-gap edge --".format(
        near_edge_rate))
    print("   cause percentages must beat that baseline to mean anything)")
    if n_sw:
        print("  strict criterion (control <2m, not flying, prev 2s): {:5.1%} pass".format(
            strict_pass / n_sw))
        print("  LAX criterion (new team d<3m within +/-1s, a priori): {:5.1%} pass".format(
            lax_pass / n_sw))
        print("  of the {} strict-failures, explained by cause (multi-label):".format(strict_fail))
        for name, c in causes.items():
            print("    {:<14} {:4d} ({:.0%})".format(name, c, c / max(1, strict_fail)))


def _ema_valid(raw, valid, ema_fn, alpha=0.35, fill=50.0):
    """Same per-run EMA the notebook emission uses (passed in, not re-imported per call)."""
    out = np.full(len(raw), fill, dtype=np.float64)
    idx = np.flatnonzero(valid)
    if not len(idx):
        return out
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    for run in np.split(idx, splits):
        if len(run):
            out[run] = ema_fn(np.nan_to_num(raw[run], nan=fill), alpha)
    return out


def part_e():
    """IDSSE -> BAR-ATH feature transfer + LOMO variance (the biggest remaining risk).

    IDSSE and the ATD tracking come from different systems: if a feature's distribution shifts,
    a threshold learned on IDSSE does not transfer as-is and must be rescaled by quantiles or
    normalised per match (as the rolling median already does inside D).
    """
    print("\n" + "=" * 78)
    print("PART E: transfer IDSSE -> this match + LOMO variance (no GT)")
    print("=" * 78)
    from select_viterbi_idsse import build_features
    from tuning import idsse

    cache_dir = os.path.join(ROOT, "data", ".cache")
    cached = sorted(f[len("idsse_"):-len(".pkl")] for f in
                    (os.listdir(cache_dir) if os.path.isdir(cache_dir) else [])
                    if f.startswith("idsse_") and f.endswith(".pkl"))
    if not cached:
        print("  no IDSSE cache -- skipped")
        return
    id_frames = [build_features(idsse.load_match(mid)) for mid in cached]
    iddf = pd.concat(id_frames, ignore_index=True)

    cols = ["d_min_L", "d_min_A", "frac_stationary_L", "frac_stationary_A",
            "mean_speed_L", "mean_speed_A", "ball_available", "ball_score",
            "ball_x_m", "ball_y_m"]
    bardf = pd.read_parquet(FEATURES_PATH, columns=cols)
    b_ok_b = bardf["ball_available"].to_numpy(dtype=bool)
    b_spd_b = ball_speed_mps(b_ok_b, bardf["ball_x_m"].to_numpy(float),
                             bardf["ball_y_m"].to_numpy(float), FPS)
    b_ok_i = iddf["ball_available"].to_numpy(dtype=bool)
    b_spd_i = ball_speed_mps(b_ok_i, iddf["ball_x_m"].to_numpy(float),
                             iddf["ball_y_m"].to_numpy(float), FPS)

    def q(s):
        s = pd.Series(s).dropna()
        if not len(s):
            return (np.nan, np.nan, np.nan)
        return (float(s.quantile(0.10)), float(s.median()), float(s.quantile(0.90)))

    rows = []
    for name, idser, barser in [
        ("d_min_L (m)", iddf["d_min_L"], bardf["d_min_L"]),
        ("d_min_A (m)", iddf["d_min_A"], bardf["d_min_A"]),
        ("ball speed (m/s)", b_spd_i, b_spd_b),
        ("frac_stationary_L", iddf["frac_stationary_L"], bardf["frac_stationary_L"]),
        ("frac_stationary_A", iddf["frac_stationary_A"], bardf["frac_stationary_A"]),
        ("mean_speed_L (m/s)", iddf["mean_speed_L"], bardf["mean_speed_L"]),
        ("mean_speed_A (m/s)", iddf["mean_speed_A"], bardf["mean_speed_A"]),
    ]:
        i10, i50, i90 = q(idser)
        b10, b50, b90 = q(barser)
        ratio = b50 / i50 if i50 and np.isfinite(i50) and i50 != 0 else np.nan
        flag = "" if (np.isfinite(ratio) and 0.8 <= ratio <= 1.25) else "  <-- CHECK (re-scale)"
        rows.append((name, i10, i50, i90, b10, b50, b90, ratio, flag))
    print("\n-- feature distributions: IDSSE pooled ({}) vs BAR-ATH --".format(
        len(cached)) + " matches")
    print("  {:22s} {:>18s} {:>18s} {:>7s}".format(
        "feature", "IDSSE p10/50/90", "BAR-ATH p10/50/90", "med ratio"))
    for name, i10, i50, i90, b10, b50, b90, ratio, flag in rows:
        print("  {:22s} {:6.2f}/{:5.2f}/{:5.2f}  {:6.2f}/{:5.2f}/{:5.2f}  {:7.2f}{}".format(
            name, i10, i50, i90, b10, b50, b90, ratio, flag))
    print("  ball_available rate: IDSSE {:.2%}  BAR-ATH {:.2%}".format(
        b_ok_i.mean(), b_ok_b.mean()))
    print("  ball_score: IDSSE fixed 1.0 (provider always has the ball); BAR-ATH mean {:.3f}"
          " (DETECTED/FOT/KINEMATIC/NONE) -- confidence channel has no IDSSE counterpart"
          .format(float(pd.Series(bardf['ball_score']).mean())))

    # --- LOMO variance --------------------------------------------------------------------------
    lomo_path = os.path.join(ROOT, "tuning", "results", "viterbi_idsse_lomo.csv")
    if os.path.exists(lomo_path):
        lomo = pd.read_csv(lomo_path)
        chosen = {"beta": 0.7, "p_switch_play": 0.005, "min_possession_frames": 25,
                  "w_control": 2.6, "w_duel_pulse": 1.0}
        sel = lomo
        for p, v in chosen.items():
            sel = sel[sel[p] == v]
        per = sel.groupby("match_id")["spell_f1"].mean()
        print("\n-- LOMO variance of the chosen config (spellF1 per IDSSE match) --")
        print("  " + "  ".join("{} {:.4f}".format(m, v) for m, v in per.items()))
        print("  mean {:.4f}  std {:.4f}  range {:.4f}..{:.4f}  (spread {:.4f}{})".format(
            per.mean(), per.std(), per.min(), per.max(), per.max() - per.min(),
            "  <-- fragile" if per.max() - per.min() > 0.06 else ""))
    print("\n  transfer verdict: thresholds with CHECK flags were RESCALED by quantile")
    print("  matching into the notebook (ball_slow/fast/flight/stat -- same operating point")
    print("  on the ATD scale); ball availability differs structurally (IDSSE 100%, ours")
    print("  ~79%) -- the missing-ball channel is ATD-specific, no IDSSE counterpart.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--idsse", action="store_true",
                    help="attempt part B even if the IDSSE cache looks absent")
    ap.add_argument("--submission", default=None,
                    help="score this CSV instead of running the notebook decoder")
    args = ap.parse_args()
    stats = part_a(submission=args.submission)
    print("\n" + "-" * 78)
    print("SUMMARY (all numbers above are GT-free)")
    print("-" * 78)
    for k, v in stats.items():
        if k not in ("d_runs",):
            print("  {:<26}: {}".format(k, "{:.4f}".format(v) if isinstance(v, float) else v))
    print("  {:<26}: {}".format("D_runs", run_length_stats(stats["d_runs"])))
    cache_dir = os.path.join(ROOT, "data", ".cache")
    cached = [f for f in (os.listdir(cache_dir) if os.path.isdir(cache_dir) else [])
              if f.startswith("idsse_") and f.endswith(".pkl")]
    if args.idsse or cached:
        part_b()
    else:
        print("\nIDSSE cache not ready yet -- part B deferred "
              "(background: python -m tuning.idsse)")
    part_c()
    part_d()
    part_e()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
