"""Test the Viterbi possession decoder from notebooks/viterbi.ipynb against the ground truth.

The decoder under test is not copied here: this script reads the notebook, finds the cell that
defines ``ViterbiPossessionConfig`` and ``PossessionViterbiHMM`` and executes exactly that cell,
so what gets scored is the logic as it lives in the notebook -- one source of truth, no drift
between the notebook and the test.

What it does, in order:

  1. exec the notebook's decoder cell;
  2. run it on the features parquet that atdProcess.ipynb produced (the same file the notebook
     loads as PARQUET_PATH);
  3. map the notebook's L/A/D/X alphabet onto the evaluation's H/A/D/X alphabet (L is the home
     team, so L -> H; A, D, X are the same letter in both) and write a full-length submission
     covering every video frame -- frames outside the two halves get X, because the model never
     ran there and X is the alphabet's word for "no answer";
  4. score it two ways:
       * frame-level on the graded window, the way the notebook's own evaluation cell meant to
         (it never fired: it looked for GT labels L/A/D while the ground truth spells home as H);
       * the official metrics in possession_metrics.py, including a segment bootstrap;
  5. save a pred-vs-GT timeline figure of the graded window.

Usage:
    python evaluation/run_viterbi_vs_gt.py
    python evaluation/run_viterbi_vs_gt.py --features /path/to/features.parquet --bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

NOTEBOOK_PATH = os.path.join(ROOT, "notebooks", "viterbi.ipynb")
GT_PATH = os.path.join(HERE, "gt", "{}_possession_gt_public.csv".format(games.GAME))
SUBMISSION_PATH = os.path.join(HERE, "results", "viterbi.csv")
FIGURE_PATH = os.path.join(ROOT, "reports", "viterbi_vs_gt_timeline.png")

# The notebook's alphabet uses L for the home team, the evaluation uses H. Same states, one
# rename; everything else is identical.
NOTEBOOK_TO_EVAL = {"L": "H", "A": "A", "D": "D", "X": "X"}


def default_features_path():
    """The parquet atdProcess.ipynb writes, honouring MTAH_DATA_DIR like the rest of the repo."""
    data_dir = os.environ.get("MTAH_DATA_DIR") or r"D:\Proyectos\Barcelona project local\data"
    return os.path.join(data_dir, "processed", "{}_atd_ball_features_LA.parquet".format(games.GAME))


def load_decoder_from_notebook():
    """Execute the notebook cell that defines the decoder and return its namespace."""
    with open(NOTEBOOK_PATH, encoding="utf-8") as handle:
        notebook = json.load(handle)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "class PossessionViterbiHMM" in source:
            namespace = {}
            exec(compile(source, NOTEBOOK_PATH, "exec"), namespace)  # noqa: S102 -- our own notebook
            return namespace
    raise RuntimeError(
        "no code cell defining PossessionViterbiHMM found in {}".format(NOTEBOOK_PATH))


def build_submission(df_viterbi, n_frames):
    """Full-length frame,label table: every video frame, mapped to the evaluation alphabet."""
    labels = np.full(n_frames, "X", dtype=object)
    frames = df_viterbi["video_frame"].to_numpy(dtype=np.int64)
    mapped = df_viterbi["pred_state"].map(NOTEBOOK_TO_EVAL)
    if mapped.isna().any():
        raise ValueError("unknown pred_state values: {}".format(
            sorted(set(df_viterbi.loc[mapped.isna(), "pred_state"]))))
    labels[frames] = mapped.to_numpy(dtype=object)
    return pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int64), "label": labels})


def frame_level_report(gt, pred_labels, title):
    """Accuracy + macro-F1 + per-class report + confusion, on frames where GT is graded."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    graded = gt["label"].isin(list(pm.SCORED))
    y_true = gt.loc[graded, "label"].to_numpy()
    y_pred = pred_labels[graded.to_numpy()]
    labels = list(pm.SCORED)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro")
    print("\n{}".format(title))
    print("  n={}  accuracy={:.4f}  macro-F1={:.4f}".format(len(y_true), acc, macro))
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("  confusion  rows=GT cols=pred  {}".format(
        "".join("{:>8}".format(c) for c in labels)))
    for row, name in enumerate(labels):
        print("  {:>10}    {}".format(name, "".join("{:>8}".format(v) for v in cm[row])))
    return acc, macro


def save_timeline_figure(df_viterbi, gt, path):
    """Three-row timeline of the graded window: Viterbi, ground truth, ball availability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    window = gt[gt["label"].isin(list(pm.SCORED))].copy()
    window = window.merge(df_viterbi[["video_frame", "pred_state", "ball_available"]],
                          left_on="frame", right_on="video_frame", how="left")
    window["pred_eval"] = window["pred_state"].map(NOTEBOOK_TO_EVAL)

    codes = {"H": 0, "A": 1, "D": 2, "X": 3}
    cmap = ListedColormap(["#1f77b4", "#d62728", "#7f7f7f", "#ffffff"])
    t_min = (window["frame"] - window["frame"].iloc[0]) / (games.GAMES[games.GAME]["fps"] * 60.0)

    fig, axes = plt.subplots(3, 1, figsize=(15, 5.6), sharex=True)
    axes[0].imshow(window["pred_eval"].map(codes).to_numpy()[None, :], aspect="auto",
                   cmap=cmap, vmin=0, vmax=3, extent=[t_min.iloc[0], t_min.iloc[-1], 0, 1])
    axes[0].set_yticks([])
    axes[0].set_title("Viterbi (notebook logic)  --  blue=H(L)  red=A  grey=D  white=X")

    axes[1].imshow(window["label"].map(codes).to_numpy()[None, :], aspect="auto",
                   cmap=cmap, vmin=0, vmax=3, extent=[t_min.iloc[0], t_min.iloc[-1], 0, 1])
    axes[1].set_yticks([])
    axes[1].set_title("Ground truth (public window, frames {}-{})".format(
        int(window["frame"].min()), int(window["frame"].max())))

    axes[2].imshow(window["ball_available"].astype(int).to_numpy()[None, :], aspect="auto",
                   cmap=ListedColormap(["#ff9800", "#2ca02c"]), vmin=0, vmax=1,
                   extent=[t_min.iloc[0], t_min.iloc[-1], 0, 1])
    axes[2].set_yticks([])
    axes[2].set_title("Ball available (green=yes, orange=team fallback)")
    axes[2].set_xlabel("minutes into the graded window")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default=default_features_path(),
                        help="features parquet from atdProcess.ipynb (default: %(default)s)")
    parser.add_argument("--gt", default=GT_PATH, help="ground truth CSV (default: public slice)")
    parser.add_argument("--out", default=SUBMISSION_PATH, help="where to write the submission")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="segment-bootstrap draws for the official metrics, 0=off")
    parser.add_argument("--figure", default=FIGURE_PATH, help="timeline figure output path")
    parser.add_argument("--no-figure", action="store_true", help="skip the timeline figure")
    parser.add_argument("--submission-only", action="store_true",
                        help="decode + write the submission, skip ALL ground-truth scoring "
                             "(GT freeze: format validation without opening the GT)")
    args = parser.parse_args()

    # --- 1. the logic under test, straight from the notebook -------------------------------
    ns = load_decoder_from_notebook()
    ViterbiPossessionConfig = ns["ViterbiPossessionConfig"]
    PossessionViterbiHMM = ns["PossessionViterbiHMM"]

    df_features = pd.read_parquet(args.features)
    print("=== VITERBI LOGIC FROM notebooks/viterbi.ipynb ===")
    print("  features : {} ({} rows, video frames {}..{})".format(
        args.features, len(df_features),
        int(df_features["video_frame"].min()), int(df_features["video_frame"].max())))

    hmm = PossessionViterbiHMM(ViterbiPossessionConfig())
    df_viterbi, summary = hmm.predict(df_features)

    print("\n=== DECODE SUMMARY (as the notebook prints it) ===")
    for key, value in summary.items():
        print("  {:<24}: {}".format(key, value))

    # --- 2. submission in the evaluation alphabet ------------------------------------------
    n_frames = games.source_frames()
    submission = build_submission(df_viterbi, n_frames)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    submission.to_csv(args.out, index=False, lineterminator="\n")
    print("\nsubmission -> {}  ({} rows, labels: {})".format(
        args.out, len(submission),
        dict(submission["label"].value_counts())))

    if args.submission_only:
        print("--submission-only: GT scoring skipped (GT freeze); validate with "
              "`python evaluation/validate_submission.py {}`".format(args.out))
        return 0

    # --- 3. score against the ground truth --------------------------------------------------
    gt = pd.read_csv(args.gt)
    gt = gt.rename(columns={gt.columns[1]: "label"})
    gt["label"] = gt["label"].astype(str).str.strip().str.upper()

    print("\n=== GROUND TRUTH ===")
    print("  {}  {} frames, graded {}".format(
        os.path.relpath(args.gt, ROOT), len(gt), dict(gt["label"].value_counts())))

    # 3a. frame-level on the graded window -- the notebook's own evaluation cell, made to work
    #     (it searched for GT labels L/A/D and therefore never matched the real H/A/D).
    pred_on_gt = submission["label"].to_numpy(dtype=object)
    frame_level_report(gt, pred_on_gt,
                       "=== FRAME-LEVEL ON GRADED FRAMES (notebook evaluation cell, H-aware) ===")

    # The notebook also split the score by ball available vs collective fallback.
    graded = gt["label"].isin(list(pm.SCORED))
    ball = df_features.set_index("video_frame")["ball_available"]
    ball_on_gt = gt.loc[graded, "frame"].map(ball).fillna(False).to_numpy(dtype=bool)
    y_true = gt.loc[graded, "label"].to_numpy()
    y_pred = pred_on_gt[graded.to_numpy()]
    from sklearn.metrics import accuracy_score, f1_score
    for flag, desc in [(True, "ball available (Paso 8 + 7)"),
                       (False, "fallback colectivo sin balón (solo Paso 7)")]:
        mask = ball_on_gt == flag
        if mask.sum() == 0:
            continue
        print("  [{}] n={:5d}  acc={:.4f}  macro-F1={:.4f}".format(
            desc, mask.sum(),
            accuracy_score(y_true[mask], y_pred[mask]),
            f1_score(y_true[mask], y_pred[mask], labels=list(pm.SCORED),
                     average="macro", zero_division=0)))

    # 3b. the official metrics -- same numbers the scoreboard ranks on
    gt_codes = pm.label_codes(gt["label"])
    pred_codes = pm.label_codes(submission["label"])
    result = pm.evaluate(gt_codes, pred_codes, bootstrap_n=args.bootstrap)

    print("\n=== OFFICIAL METRICS (possession_metrics.evaluate) ===")
    print("  macro_f1          : {:.4f}".format(result["macro_f1"]))
    print("  spell_f1_mean     : {:.4f}   (IoU 0.3..0.7; at 0.5: {:.4f})".format(
        result["spell_f1_mean"], result["spell_f1_at_headline"]))
    if result["bootstrap_low"] is not None:
        print("    bootstrap macroF1 {:.4f} .. {:.4f}  ({} draws)".format(
            result["bootstrap_low"], result["bootstrap_high"], result["bootstrap_n"]))
    if result["spell_bootstrap_low"] is not None:
        print("    bootstrap spellF1 {:.4f} .. {:.4f}".format(
            result["spell_bootstrap_low"], result["spell_bootstrap_high"]))
    print("  accuracy          : {:.4f}   kappa {:.4f}".format(
        result["accuracy"], result["kappa"]))
    print("  per class         : " + "   ".join(
        "{} F1 {:.4f} (P {:.4f} R {:.4f} n={})".format(
            name, result["per_class"][name]["f1"], result["per_class"][name]["precision"],
            result["per_class"][name]["recall"], result["per_class"][name]["support"])
        for name in pm.SCORED))
    print("  alive/dead F1     : {:.4f}   team acc gt-alive {:.4f}  both-alive {:.4f}".format(
        result["alive_dead_f1"], result["team_acc_gt_alive"], result["team_acc_both_alive"]))
    for tau in sorted(result["transitions"]):
        t = result["transitions"][tau]
        print("  transitions tau={:<3}: F1 {:.4f}  P {:.4f}  R {:.4f}  matched {} missed {} "
              "spurious {}  mean lat {:+.1f} frames".format(
                  tau, t["f1"], t["precision"], t["recall"], t["n_matched"], t["n_missed"],
                  t["n_spurious"], t["mean_latency"] if t["mean_latency"] is not None else 0.0))
    print("  possessions       : gt {} pred {} ({:+d})  mean duration {:.1f}s vs gt {:.1f}s".format(
        result["n_gt_spells"], result["n_pred_spells"], result["spell_count_error"],
        result["mean_spell_duration_s"], result["gt_mean_spell_duration_s"]))
    print("  fragmentation     : {} / {} = {:.2f}".format(
        result["n_pred_segments"], result["n_gt_segments"], result["fragmentation"]))
    print("  abstentions (X)   : {} of {} graded frames ({:.2f}%)".format(
        result["n_abstain_on_graded"], result["n_graded"],
        100.0 * result["n_abstain_on_graded"] / max(1, result["n_graded"])))

    # 3c. diagnostics by predicted state, the notebook's "sin ground truth" cell, now with GT
    diag = df_viterbi.merge(
        df_features[["video_frame", "mean_speed_L", "mean_speed_A",
                     "frac_stationary_L", "frac_stationary_A"]],
        on="video_frame").merge(
            gt.rename(columns={"frame": "video_frame"})[["video_frame", "label"]],
            on="video_frame", how="left")
    diag["mean_spd"] = 0.5 * (diag["mean_speed_L"] + diag["mean_speed_A"])
    diag["frac_stat"] = 0.5 * (diag["frac_stationary_L"] + diag["frac_stationary_A"])
    print("\n=== FEATURE STATS BY PREDICTED STATE (graded frames) ===")
    print(diag[diag["label"].isin(list(pm.SCORED))]
          .groupby("pred_state")[["mean_spd", "frac_stat", "ball_available"]]
          .agg(["mean", "median"]).round(4).to_string())
    print("\n=== PREDICTION vs GT CONTINGENCY ON GRADED FRAMES (rows=GT, cols=pred) ===")
    cross = pd.crosstab(diag.loc[diag["label"].isin(list(pm.SCORED)), "label"],
                        diag.loc[diag["label"].isin(list(pm.SCORED)), "pred_state"])
    print(cross.to_string())

    # --- 4. figure --------------------------------------------------------------------------
    if not args.no_figure:
        path = save_timeline_figure(df_viterbi, gt, args.figure)
        print("\ntimeline figure -> {}".format(path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
