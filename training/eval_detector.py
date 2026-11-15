"""Is a detector checkpoint actually better? Full-frame evaluation on a held-out external game.

Tile-level mAP (what ultralytics prints during training) is not what the pipeline
uses. The pipeline cuts each full frame into tiles, runs the detector on every tile,
merges the detections and keeps *one* ball position per frame. This module measures
exactly that, through the same code path as inference
(``src.fot_video.detect_frame_candidates`` — the tiling, the confidence floor and the
duplicate suppression are not re-implemented here), against the ball annotations of
an external game the checkpoint was **not** trained on.

Two stages, so the expensive one runs once:

1. ``detect`` (GPU): every sampled frame -> all candidates above the low confidence
   floor, written to ``candidates__<label>__g<game>.csv`` (``game, frame, gx, gy,
   conf``) plus the list of frames that were evaluated. Thresholds are *not* applied
   here, so they can be studied afterwards without touching the GPU again. The same
   CSV feeds hard-negative mining (``training.tiles --hard-negatives``) and FoT
   training (``training.fot_data``).
2. ``score`` (numpy only): per confidence threshold, using the top candidate of each
   frame — recall and precision within a pixel radius of the annotated ball, false
   alarms on frames with no visible ball, the any-candidate recall (the ceiling a
   perfect selector could reach) and the localisation error of the hits. Two
   checkpoints are compared on the SAME frames; the interval on the recall difference
   accounts for autocorrelation (adjacent frames are far from independent), and below
   ``MIN_EFFECTIVE_FRAMES`` effectively independent frames no verdict is given.

A cache is only reused for exactly the request that produced it (same checkpoint
file, frame step, frame cap, input size); anything else is an error, not a silent
reuse. Its identity is stored next to it in ``cache__<label>__g<game>.json``.

Rules this tool enforces or relies on:

* external games only — every path goes through ``training.compliance``;
* evaluate on a game that is in neither checkpoint's training split, otherwise the
  comparison flatters whichever model saw it; mine hard negatives on *training*
  games only, for the same reason;
* this is evidence about the *detector*. It says nothing about the possession score,
  and nothing here may be run on the provided match.

    python -m training.eval_detector --manifest data/external/manifest.json --games 9 \
        --weights models/rtdetr_fold0_best.pt runs/rtdetr/fold_0/weights/best.pt \
        --labels kaggle local --out runs/eval/fold0_vs_local
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from training.compliance import assert_external
from training.manifest import load_annotations, load_manifest

DEFAULT_RADII_PX = (5.0, 10.0, 20.0)
DEFAULT_CONF_GRID = (0.07, 0.18, 0.30, 0.40, 0.50, 0.68)
DEFAULT_FRAME_STEP = 5
DEFAULT_MAX_CANDIDATES = 20
CANDIDATE_COLUMNS = ["game", "frame", "gx", "gy", "conf"]


# --------------------------------------------------------------------------- #
# Stage 2 — scoring (pure numpy / pandas)
# --------------------------------------------------------------------------- #
def top_candidates(cands: pd.DataFrame, min_conf: float) -> pd.DataFrame:
    """The most confident candidate per frame among those with ``conf >= min_conf``.

    Ties are broken by position so the result does not depend on row order.
    """
    kept = cands[cands["conf"] >= min_conf]
    if kept.empty:
        return kept.assign(frame=kept["frame"].astype(np.int64)).set_index("frame")
    ordered = kept.sort_values(["frame", "conf", "gx", "gy"],
                               ascending=[True, False, True, True])
    return ordered.drop_duplicates("frame").set_index("frame")


def frame_hits(cands: pd.DataFrame, truth: pd.DataFrame, frames,
               min_conf: float, radius_px: float) -> pd.DataFrame:
    """Per evaluated frame: is there a ball, a top detection, and is it a hit.

    Columns: ``has_ball``, ``has_det``, ``hit`` (top detection within ``radius_px``
    of the ball), ``any_hit`` (some candidate within the radius), ``err_px`` (distance
    of the top detection to the ball; NaN without both).
    """
    frames = np.asarray(sorted(set(int(f) for f in frames)), dtype=np.int64)
    out = pd.DataFrame(index=pd.Index(frames, name="frame"))
    truth_xy = truth.set_index("frame")[["x", "y"]].reindex(frames)
    out["has_ball"] = truth_xy["x"].notna().to_numpy()

    top = top_candidates(cands, min_conf).reindex(frames)
    out["has_det"] = top["conf"].notna().to_numpy()
    err = np.hypot(top["gx"].to_numpy(dtype=float) - truth_xy["x"].to_numpy(dtype=float),
                   top["gy"].to_numpy(dtype=float) - truth_xy["y"].to_numpy(dtype=float))
    out["err_px"] = err
    with np.errstate(invalid="ignore"):
        out["hit"] = (err <= radius_px) & out["has_ball"].to_numpy() & out["has_det"].to_numpy()

    any_hit = np.zeros(len(frames), dtype=bool)
    kept = cands[cands["conf"] >= min_conf]
    if not kept.empty:
        merged = kept.merge(truth[["frame", "x", "y"]], on="frame", how="inner")
        close = np.hypot(merged["gx"] - merged["x"], merged["gy"] - merged["y"]) <= radius_px
        hit_frames = set(merged.loc[close, "frame"].astype(int))
        any_hit = np.array([int(f) in hit_frames for f in frames], dtype=bool)
    out["any_hit"] = any_hit
    return out


def summarise(hits: pd.DataFrame) -> dict:
    """Detector metrics from a :func:`frame_hits` table. Ratios are NaN on empty sets."""
    n_frames = int(len(hits))
    n_ball = int(hits["has_ball"].sum())
    n_empty = n_frames - n_ball
    n_det = int(hits["has_det"].sum())
    n_hit = int(hits["hit"].sum())
    false_alarms = int((hits["has_det"] & ~hits["has_ball"]).sum())
    hit_err = hits.loc[hits["hit"], "err_px"]

    def ratio(a: int, b: int) -> float:
        return float(a) / b if b else float("nan")

    return {
        "frames": n_frames,
        "frames_with_ball": n_ball,
        "recall": ratio(n_hit, n_ball),
        "precision": ratio(n_hit, n_det),
        "any_candidate_recall": ratio(int((hits["any_hit"] & hits["has_ball"]).sum()),
                                      n_ball),
        "false_alarm_rate_empty": ratio(false_alarms, n_empty),
        "detection_rate": ratio(n_det, n_frames),
        "median_err_px": float(hit_err.median()) if len(hit_err) else float("nan"),
        "p90_err_px": float(hit_err.quantile(0.9)) if len(hit_err) else float("nan"),
    }


def score_table(cands: pd.DataFrame, truth: pd.DataFrame, frames,
                radii=DEFAULT_RADII_PX, conf_grid=DEFAULT_CONF_GRID) -> pd.DataFrame:
    """One row per (confidence threshold, radius) with the :func:`summarise` metrics."""
    rows = []
    for min_conf in conf_grid:
        for radius in radii:
            row = {"min_conf": float(min_conf), "radius_px": float(radius)}
            row.update(summarise(frame_hits(cands, truth, frames, min_conf, radius)))
            rows.append(row)
    return pd.DataFrame(rows)


MIN_EFFECTIVE_FRAMES = 100


def integrated_autocorrelation_time(x, window_factor: float = 5.0) -> float:
    """Integrated autocorrelation time of a series (Sokal's automatic window).

    ``1`` for independent values; ``k`` means the series carries about as much
    information as ``n / k`` independent values. Estimated with FFT autocorrelation
    and the smallest window ``m >= window_factor * tau(m)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 1.0
    x = x - x.mean()
    if not np.any(x):
        return 1.0
    spectrum = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    acf = acf / acf[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * float(acf[1:m + 1].sum())
        if m >= window_factor * tau:
            break
    return max(tau, 1.0)


def paired_delta(hit_a: np.ndarray, hit_b: np.ndarray, has_ball: np.ndarray,
                 min_effective: int = MIN_EFFECTIVE_FRAMES) -> dict:
    """Recall of B minus recall of A on the same frames, with an honest interval.

    The per-frame difference on frames with a ball is strongly autocorrelated (a
    detector that loses the ball loses it for a while). The standard error is
    therefore inflated by the integrated autocorrelation time, which gives the
    effective number of independent frames ``n_eff``. A naive block bootstrap was
    measured to call a difference "significant" in 15-45 % of simulated runs where
    none existed; this estimator stays near the nominal 5 % once ``n_eff`` is about
    100, and below ``min_effective`` the verdict is ``insufficient data``.

    ``verdict`` is one of ``B better``, ``A better``, ``no evidence``,
    ``insufficient data``. One game is still one game: confirm on a second fold.
    """
    hit_a = np.asarray(hit_a, dtype=bool)
    hit_b = np.asarray(hit_b, dtype=bool)
    has_ball = np.asarray(has_ball, dtype=bool)
    if not (len(hit_a) == len(hit_b) == len(has_ball)):
        raise ValueError("hit_a, hit_b and has_ball must be aligned on the same frames")
    diff = hit_b[has_ball].astype(np.float64) - hit_a[has_ball].astype(np.float64)
    n = len(diff)
    nan = float("nan")
    if n == 0:
        return {"delta_recall": nan, "ci95": [nan, nan], "frames_with_ball": 0,
                "n_eff": 0.0, "tau": nan, "verdict": "insufficient data"}
    delta = float(diff.mean())
    # The windowed estimate is biased low when the series holds only a few long runs;
    # a two-state chain with mean run length L has tau ~ 2L - 1, used as a floor.
    var = float(diff.var(ddof=1)) if n > 1 else 0.0
    if var > 0:
        changes = int(np.count_nonzero(np.diff(diff)))
        mean_run = n / (changes + 1)
        tau = max(integrated_autocorrelation_time(diff), 2.0 * mean_run - 1.0)
    else:
        tau = 1.0      # a constant difference (identical outputs, or one always better)
    n_eff = n / tau
    half = 1.96 * float(np.sqrt(var * tau / n)) if var > 0 else 0.0
    low, high = delta - half, delta + half
    if n_eff < min_effective:
        verdict = "insufficient data"
    elif low > 0:
        verdict = "B better"
    elif high < 0:
        verdict = "A better"
    else:
        verdict = "no evidence"
    return {"delta_recall": delta, "ci95": [low, high], "frames_with_ball": n,
            "n_eff": float(n_eff), "tau": float(tau), "verdict": verdict}


def markdown_report(tables: dict[str, pd.DataFrame], comparisons: list[dict],
                    header: dict) -> str:
    """Human-readable summary of a scoring run."""
    lines = ["# Detector evaluation — held-out external game(s)", ""]
    for key, value in header.items():
        lines.append(f"- **{key}:** {value}")
    lines += ["", "Top-1 detection per frame after tiling and duplicate suppression.",
              "`recall` = frames with a ball whose top detection is within the radius;"
              " `precision` = top detections that are within the radius;"
              " `any_candidate_recall` = ceiling if the best candidate were always"
              " picked; `false_alarm_rate_empty` = frames with no visible ball that"
              " still got a detection.", ""]
    columns = ["min_conf", "radius_px", "recall", "precision", "any_candidate_recall",
               "false_alarm_rate_empty", "median_err_px", "p90_err_px"]
    for label, table in tables.items():
        frames = int(table["frames"].iloc[0]) if len(table) else 0
        with_ball = int(table["frames_with_ball"].iloc[0]) if len(table) else 0
        lines += [f"## {label}", "",
                  f"{frames} frames evaluated, {with_ball} with a visible ball.", "",
                  "| " + " | ".join(columns) + " |",
                  "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in table.itertuples(index=False):
            cells = [f"{getattr(row, c):.3f}" if c not in ("min_conf", "radius_px")
                     else f"{getattr(row, c):g}" for c in columns]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if comparisons:
        lines += ["## Paired comparison (same frames, autocorrelation-corrected)", "",
                  "`n_eff` = effectively independent frames; below"
                  f" {MIN_EFFECTIVE_FRAMES} no verdict is given.", "",
                  "| A | B | min_conf | radius_px | frames | recall B - A | 95% CI |"
                  " n_eff | verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in comparisons:
            lines.append(f"| {c['a']} | {c['b']} | {c['min_conf']:g} |"
                         f" {c['radius_px']:g} | {c['frames_with_ball']} |"
                         f" {c['delta_recall']:+.3f} |"
                         f" [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |"
                         f" {c['n_eff']:.0f} | {c['verdict']} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Stage 1 — detection (GPU, cv2, ultralytics; imported lazily)
# --------------------------------------------------------------------------- #
def candidates_path(out_dir: Path, label: str, game_id: str) -> Path:
    """Where the candidates of one (checkpoint label, game) pair are cached."""
    return out_dir / f"candidates__{label}__g{game_id}.csv"


def frames_path(out_dir: Path, label: str, game_id: str) -> Path:
    """The list of frames one checkpoint was evaluated on, for one game."""
    return out_dir / f"frames__{label}__g{game_id}.csv"


def cache_meta_path(out_dir: Path, label: str, game_id: str) -> Path:
    """The identity of a (label, game) cache: which request produced it."""
    return out_dir / f"cache__{label}__g{game_id}.json"


def cache_identity(weights, frame_step: int, max_frames, imgsz: int) -> dict:
    """What a cache must match to be reused: the exact checkpoint file and settings."""
    weights = Path(weights).expanduser().resolve()
    stat = weights.stat()
    return {"weights": str(weights), "weights_bytes": stat.st_size,
            "weights_mtime_ns": stat.st_mtime_ns, "frame_step": int(frame_step),
            "max_frames": max_frames, "imgsz": int(imgsz)}


def read_cache_identity(out_dir: Path, label: str, game_id: str) -> dict | None:
    """The stored identity of a cache, or None when there is no complete cache."""
    meta = cache_meta_path(out_dir, label, game_id)
    if not (meta.is_file() and candidates_path(out_dir, label, game_id).is_file()
            and frames_path(out_dir, label, game_id).is_file()):
        return None
    return json.loads(meta.read_text(encoding="utf-8"))


def recorded_imgsz(weights) -> int:
    """The input size a checkpoint was trained at, read from its recorded train_args.

    A detector must be run at the size it was trained on (a 640 px model fed 320 px
    tiles sees a ball half the size it learned). Falls back to the tile size when the
    file records nothing.
    """
    import src.fot_video as F
    from training.inspect_checkpoint import read_train_args

    try:
        recorded = read_train_args(weights) or {}
    except (OSError, ValueError):
        recorded = {}
    value = recorded.get("imgsz")
    return int(value) if isinstance(value, int) and value > 0 else F.TILE_SIZE


def detect_game(rt_model, game, out_csv: Path, frames_csv: Path, device,
                frame_step: int = DEFAULT_FRAME_STEP, max_frames: int | None = None,
                max_candidates: int = DEFAULT_MAX_CANDIDATES,
                progress_every: int = 200, imgsz: int | None = None) -> int:
    """Run the tiled detector over one external game and cache every candidate.

    Uses ``src.fot_video.detect_frame_candidates`` with no pitch mask (external games
    have no homography). Writes atomically (temp file then rename), so an interrupted
    run never leaves a half-written cache that a later run would trust.
    """
    import time

    import src.fot_video as F
    from training.frames import frame_size, iter_frames

    frame_w, frame_h = frame_size(game)
    tiles = F.compute_tile_coords(frame_w, frame_h, F.TILE_SIZE, F.TILE_STRIDE)
    predict_device = F.yolo_device(device)
    imgsz = int(imgsz or F.TILE_SIZE)
    rows, frames = [], []
    started = time.perf_counter()
    for n_seen, (frame, image) in enumerate(
            iter_frames(game, step=frame_step, max_frames=max_frames), 1):
        cands = F.detect_frame_candidates(rt_model, image, tiles, None,
                                          predict_device, imgsz)
        frames.append(int(frame))
        for gx, gy, conf in cands[:max_candidates]:
            rows.append((game.id, int(frame), float(gx), float(gy), float(conf)))
        if progress_every and n_seen % progress_every == 0:
            rate = n_seen / max(time.perf_counter() - started, 1e-6)
            print(f"  game {game.id}: {n_seen} frames, {rate:.2f} frames/s", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".csv.tmp")
    pd.DataFrame(rows, columns=CANDIDATE_COLUMNS).to_csv(tmp, index=False)
    tmp.replace(out_csv)
    tmp = frames_csv.with_suffix(".csv.tmp")
    pd.DataFrame({"frame": frames}).to_csv(tmp, index=False)
    tmp.replace(frames_csv)
    return len(frames)


def load_candidates(path) -> pd.DataFrame:
    """A candidates CSV with the game id kept as a string."""
    return pd.read_csv(path, dtype={"game": str})


def main(argv: list[str] | None = None) -> int:
    """Detect (cached) and score one or more checkpoints on held-out external games."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m training.eval_detector",
        description="Full-frame evaluation of detector checkpoints on external games.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--games", nargs="+", required=True,
                        help="held-out game ids (not in any checkpoint's train split)")
    parser.add_argument("--weights", nargs="+", required=True, help="checkpoint(s)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="short name per checkpoint (default: file stem)")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="cap on evaluated frames per game")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="detector input size for every checkpoint (default: the"
                             " size each checkpoint recorded at training time)")
    parser.add_argument("--radii", nargs="+", type=float, default=list(DEFAULT_RADII_PX))
    parser.add_argument("--conf-grid", nargs="+", type=float,
                        default=list(DEFAULT_CONF_GRID))
    parser.add_argument("--force", action="store_true",
                        help="re-run detection even if a matching cache exists, and"
                             " replace a cache made with other settings")
    parser.add_argument("--score-only", action="store_true",
                        help="skip detection; score the cached candidates")
    args = parser.parse_args(argv)

    labels = args.labels or [Path(w).stem for w in args.weights]
    if len(labels) != len(args.weights) or len(set(labels)) != len(labels):
        parser.error("--labels must give one distinct name per --weights entry (two"
                     " runs' best.pt share the default label 'best')")
    out_dir = Path(args.out).expanduser()
    assert_external(out_dir, what="evaluation output")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    games = [manifest.game(g) for g in args.games]

    if not args.score_only:
        import torch
        from ultralytics import RTDETR

        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        for label, weights in zip(labels, args.weights):
            model = None
            imgsz = args.imgsz or recorded_imgsz(weights)
            wanted = cache_identity(weights, args.frame_step, args.max_frames, imgsz)
            for game in games:
                target = candidates_path(out_dir, label, game.id)
                stored = read_cache_identity(out_dir, label, game.id)
                if stored is not None and not args.force:
                    if stored != wanted:
                        changed = sorted(k for k in wanted if stored.get(k) != wanted[k])
                        raise SystemExit(
                            f"[{label}] game {game.id}: a cache made with different"
                            f" {changed} exists in {out_dir}. Use another --out or"
                            " --labels, or --force to replace it.")
                    print(f"[{label}] game {game.id}: cached ({target.name})")
                    continue
                if model is None:
                    model = RTDETR(str(weights))
                print(f"[{label}] game {game.id}: detecting on {device}"
                      f" at imgsz {imgsz}", flush=True)
                n = detect_game(model, game, target,
                                frames_path(out_dir, label, game.id),
                                device, args.frame_step, args.max_frames, imgsz=imgsz)
                cache_meta_path(out_dir, label, game.id).write_text(
                    json.dumps(wanted, indent=2), encoding="utf-8")
                print(f"[{label}] game {game.id}: {n} frames -> {target.name}")

    identities = {}
    for label in labels:
        for game in games:
            stored = read_cache_identity(out_dir, label, game.id)
            if stored is None:
                raise SystemExit(f"[{label}] game {game.id}: no complete cache in"
                                 f" {out_dir} — run without --score-only first")
            identities[(label, game.id)] = stored
    # Compare checkpoints on the frames every one of them was evaluated on.
    common_frames = {}
    for game in games:
        sets = [set(pd.read_csv(frames_path(out_dir, label, game.id))["frame"])
                for label in labels]
        common = set.intersection(*sets)
        if any(len(s) != len(common) for s in sets):
            print(f"NOTE game {game.id}: checkpoints were evaluated on different frames;"
                  f" scoring the {len(common)} frames they share")
        common_frames[game.id] = np.array(sorted(common), dtype=np.int64)

    tables: dict[str, pd.DataFrame] = {}
    per_frame: dict[tuple, pd.DataFrame] = {}
    for label in labels:
        parts = []
        for game in games:
            cands = load_candidates(candidates_path(out_dir, label, game.id))
            frames = common_frames[game.id]
            truth = load_annotations(game)
            part = score_table(cands, truth, frames, args.radii, args.conf_grid)
            part.insert(0, "game", game.id)
            parts.append(part)
            for min_conf in args.conf_grid:
                for radius in args.radii:
                    per_frame[(label, game.id, min_conf, radius)] = frame_hits(
                        cands, truth, frames, min_conf, radius)
        full = pd.concat(parts, ignore_index=True)
        full.to_csv(out_dir / f"metrics__{label}.csv", index=False)
        # one table per label for the report: frame-weighted pooling over the games
        pooled = []
        for min_conf in args.conf_grid:
            for radius in args.radii:
                hits = pd.concat([per_frame[(label, g.id, min_conf, radius)]
                                  for g in games], ignore_index=True)
                row = {"min_conf": float(min_conf), "radius_px": float(radius)}
                row.update(summarise(hits))
                pooled.append(row)
        tables[label] = pd.DataFrame(pooled)

    comparisons = []
    if len(labels) >= 2:
        base = labels[0]
        for other in labels[1:]:
            for min_conf in args.conf_grid:
                for radius in args.radii:
                    a = pd.concat([per_frame[(base, g.id, min_conf, radius)]
                                   for g in games], ignore_index=True)
                    b = pd.concat([per_frame[(other, g.id, min_conf, radius)]
                                   for g in games], ignore_index=True)
                    result = paired_delta(a["hit"].to_numpy(), b["hit"].to_numpy(),
                                          a["has_ball"].to_numpy())
                    comparisons.append({"a": base, "b": other,
                                        "min_conf": float(min_conf),
                                        "radius_px": float(radius), **result})

    steps = sorted({identities[k]["frame_step"] for k in identities})
    header = {"dataset": f"{manifest.dataset} ({manifest.license})",
              "games": ", ".join(g.id for g in games),
              "checkpoints": ", ".join(
                  f"{label}={identities[(label, games[0].id)]['weights']}"
                  f" (imgsz {identities[(label, games[0].id)]['imgsz']})"
                  for label in labels),
              "frame step (from the caches)": ", ".join(str(x) for x in steps)}
    report = markdown_report(tables, comparisons, header)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    (out_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2),
                                              encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
