"""Acceptance test for a detector checkpoint: the real inference chain on a synthetic clip.

``tests/test_fot_video.py`` replaces the detector, the video and the fusion with
fakes, so nothing in the unit tests ever runs RT-DETR, OpenCV decoding or FoT v2 for
real. This script does, without touching any match data: it renders a short synthetic
clip (a small bright ball crossing a drawn pitch — ``training.synth``), then runs the
*unmodified* ``src.fot_video.detect_window`` and ``fuse_predictions`` on it with real
checkpoints and a synthetic homography, and compares the detections with the known
ball positions.

Both the detector's choice per frame (``hit_rate``) and the fused, smoothed output
(``fused_hit_rate``) are compared with the known ball positions; both must pass.

It answers three questions in under a minute on a GPU:

* does the environment work end to end (video decode -> tiling -> RT-DETR on the
  device -> coordinate mapping -> kinematic gate -> FoT v2 -> RTS/Hermite fusion)?
* does this checkpoint still find a ball-like object at the scale inference uses, and
  does it place it within a few pixels?
* is a retrained checkpoint a drop-in replacement for the committed one (same call
  path, same output contract)?

Passing is necessary, not sufficient: a synthetic dot is far easier than a real ball.
Real quality is measured by ``training.eval_detector`` on a held-out external game.

    python -m training.smoke_inference                              # committed weights
    python -m training.smoke_inference --rtdetr runs/rtdetr/fold_0/weights/best.pt
    python -m training.smoke_inference --frames 12 --width 640 --height 480 --device cpu

Exit code 0 when at least ``--min-hit-rate`` of the frames with a visible ball are
detected within ``--radius`` pixels.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from training.compliance import REPO_ROOT

DEFAULT_RTDETR = REPO_ROOT / "models" / "rtdetr_fold0_best.pt"
DEFAULT_FOT = REPO_ROOT / "models" / "fot_v2_fold0.pt"


def synthetic_homography(n_frames: int, width: int, height: int):
    """Per-frame homography mapping the normalised pitch onto the drawn pitch.

    Same convention as ``starter/load_homography`` (normalised pitch -> pixels): the
    pitch occupies the frame minus the 8 % border ``training.synth`` draws.
    """
    pad_x, pad_y = 0.08 * width, 0.08 * height
    matrix = np.array([[width - 2 * pad_x, 0.0, pad_x],
                       [0.0, height - 2 * pad_y, pad_y],
                       [0.0, 0.0, 1.0]])
    return (np.tile(matrix, (n_frames, 1, 1)), np.zeros(n_frames),
            np.ones(n_frames, dtype=bool))


def run(rtdetr_weights=DEFAULT_RTDETR, fot_weights=DEFAULT_FOT, n_frames: int = 30,
        width: int = 1920, height: int = 1080, device=None, radius_px: float = 5.0,
        seed: int = 0, workdir=None, imgsz: int | None = None) -> dict:
    """Render the clip, run detection + fusion, return the comparison as a dict."""
    import cv2
    import torch
    from ultralytics import RTDETR

    import src.fot_video as F
    from training import synth
    from training.eval_detector import recorded_imgsz

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    imgsz = int(imgsz or recorded_imgsz(rtdetr_weights))
    rng = np.random.default_rng(seed)
    xs, ys = synth.ball_path(n_frames, width, height, rng)
    # keep the ball on the pitch, clear of the lines the detector could latch on to
    xs = np.clip(xs, 0.15 * width, 0.85 * width)
    ys = np.clip(ys, 0.15 * height, 0.85 * height)

    with tempfile.TemporaryDirectory(dir=workdir) as folder:
        video = str(Path(folder) / "clip.mp4")
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 25.0,
                                 (width, height))
        if not writer.isOpened():
            raise IOError("cv2.VideoWriter could not open an mp4v stream")
        for index in range(n_frames):
            writer.write(synth.render_frame(width, height, (xs[index], ys[index]),
                                            (), rng))
        writer.release()

        matrices, reproj, valid = synthetic_homography(n_frames, width, height)
        cfg = {**F.CFG, "fot_device": device, "fot_verbose": False,
               "fot_imgsz": imgsz}
        rt_model = RTDETR(str(rtdetr_weights))
        fot_model = F.build_fot_model(F.WINDOW_SIZE, weights=fot_weights, device=device)

        started = time.perf_counter()
        det_pass = F.detect_window(rt_model, video, matrices, reproj, valid,
                                   0, n_frames - 1, cfg)
        detect_s = time.perf_counter() - started
        preds = F.fuse_predictions(fot_model, det_pass, matrices, reproj, valid,
                                   cfg, device)

    errors = []
    for frame in range(n_frames):
        _, gx, gy, _ = det_pass.history.get(frame, (frame, np.nan, np.nan, 0.0))
        errors.append(float(np.hypot(gx - xs[frame], gy - ys[frame]))
                      if not np.isnan(gx) else float("nan"))
    errors = np.array(errors)
    hits = np.nan_to_num(errors, nan=np.inf) <= radius_px
    fused_errors = []
    for frame in range(n_frames):
        px, py, _, state, _ = preds.get(frame, (np.nan, np.nan, 0.0, "NONE", 0))
        fused_errors.append(float(np.hypot(px - xs[frame], py - ys[frame]))
                            if state != "NONE" and np.isfinite(px) else float("nan"))
    fused_errors = np.array(fused_errors)
    fused_hits = np.nan_to_num(fused_errors, nan=np.inf) <= radius_px
    states: dict[str, int] = {}
    for value in preds.values():
        states[value[3]] = states.get(value[3], 0) + 1
    tiles = len(F.compute_tile_coords(width, height))
    return {
        "device": str(device),
        "rtdetr_weights": str(rtdetr_weights),
        "fot_weights": str(fot_weights),
        "frames": n_frames,
        "frame_size": [width, height],
        "imgsz": imgsz,
        "tiles_per_frame": tiles,
        "detect_seconds": round(detect_s, 3),
        "frames_per_second": round(n_frames / max(detect_s, 1e-9), 3),
        "frames_detected": int(np.isfinite(errors).sum()),
        "hit_rate": float(hits.mean()),
        "radius_px": radius_px,
        "median_error_px": float(np.nanmedian(errors)) if np.isfinite(errors).any()
        else None,
        "max_error_px": float(np.nanmax(errors)) if np.isfinite(errors).any() else None,
        "fused_states": states,
        "fused_frames": len(preds),
        "fused_hit_rate": float(fused_hits.mean()),
        "fused_median_error_px": float(np.nanmedian(fused_errors))
        if np.isfinite(fused_errors).any() else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the synthetic-clip acceptance test and print the result as JSON."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m training.smoke_inference",
        description="Real inference chain on a synthetic clip (no match data).")
    parser.add_argument("--rtdetr", default=str(DEFAULT_RTDETR))
    parser.add_argument("--fot", default=str(DEFAULT_FOT))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="detector input size (default: recorded in the checkpoint)")
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--min-hit-rate", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    result = run(args.rtdetr, args.fot, args.frames, args.width, args.height,
                 args.device, args.radius, args.seed, imgsz=args.imgsz)
    result["min_hit_rate"] = args.min_hit_rate
    result["passed"] = bool(result["hit_rate"] >= args.min_hit_rate
                            and result["fused_hit_rate"] >= args.min_hit_rate
                            and result["fused_frames"] == args.frames)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
