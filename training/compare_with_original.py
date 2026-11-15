"""Does the reviewed src/fot_video.py still produce exactly what the author's version did?

The review branch changed ``src/fot_video.py`` (bug fixes, opt-in switches, helpers
extracted for training). Every change is meant to leave the default output identical
for valid inputs, and the unit tests check that with fakes. This script checks it with
the REAL models: it loads the author's original module straight from git (default ref
``0e7b9af``, the head of the author's branch), renders the synthetic clip of
:mod:`training.smoke_inference`, runs ``detect_window`` and ``fuse_predictions`` of
both versions on it with the same committed checkpoints, and compares the results
value for value.

No match data is involved. Heavy on CPU (every frame is 32 RT-DETR tiles at 1920x1080,
run twice), fast on a GPU — it is one of the checks handed to the RTX PC.

    python -m training.compare_with_original                    # 1920x1080, 30 frames
    python -m training.compare_with_original --frames 4 --width 640 --height 480 --device cpu

Exit code 0 when detections and fused predictions are identical.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from training.compliance import REPO_ROOT

ORIGINAL_REF = "0e7b9af"


def load_original(ref: str = ORIGINAL_REF, workdir=None):
    """Import ``src/fot_video.py`` as it is at ``ref``, under its own module name."""
    source = subprocess.run(["git", "show", f"{ref}:src/fot_video.py"], cwd=REPO_ROOT,
                            check=True, capture_output=True, text=True).stdout
    folder = Path(tempfile.mkdtemp(prefix="fot_original_", dir=workdir))
    path = folder / "fot_video_original.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("fot_video_original", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fot_video_original"] = module
    spec.loader.exec_module(module)
    return module


def _same(a, b) -> bool:
    """Equality that treats NaN == NaN (both versions write NaN for 'no ball')."""
    if isinstance(a, float) and isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or a == b
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def compare(frames: int = 30, width: int = 1920, height: int = 1080, device=None,
            seed: int = 0, ref: str = ORIGINAL_REF) -> dict:
    """Run both versions on the same synthetic clip and list every difference."""
    import cv2
    import torch
    from ultralytics import RTDETR

    import src.fot_video as new
    from training import synth
    from training.smoke_inference import DEFAULT_FOT, DEFAULT_RTDETR, synthetic_homography

    old = load_original(ref)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    xs, ys = synth.ball_path(frames, width, height, rng)
    xs = np.clip(xs, 0.15 * width, 0.85 * width)
    ys = np.clip(ys, 0.15 * height, 0.85 * height)
    # a few frames without the ball, so the bridging code runs too
    hidden = {frames // 3, frames // 3 + 1, (2 * frames) // 3} if frames >= 6 else set()

    with tempfile.TemporaryDirectory() as folder:
        video = str(Path(folder) / "clip.mp4")
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 25.0,
                                 (width, height))
        for index in range(frames):
            ball = None if index in hidden else (xs[index], ys[index])
            writer.write(synth.render_frame(width, height, ball, (), rng))
        writer.release()

        matrices, reproj, valid = synthetic_homography(frames, width, height)
        cfg = {**new.CFG, "fot_device": device, "fot_verbose": False}
        rt_model = RTDETR(str(DEFAULT_RTDETR))
        fot_new = new.build_fot_model(new.WINDOW_SIZE, weights=DEFAULT_FOT, device=device)
        fot_old = old.build_fot_model(old.WINDOW_SIZE, weights=DEFAULT_FOT, device=device)

        det_new = new.detect_window(rt_model, video, matrices, reproj, valid, 0,
                                    frames - 1, cfg)
        det_old = old.detect_window(rt_model, video, matrices, reproj, valid, 0,
                                    frames - 1, cfg)
        preds_new = new.fuse_predictions(fot_new, det_new, matrices, reproj, valid, cfg,
                                         device)
        preds_old = old.fuse_predictions(fot_old, det_old, matrices, reproj, valid, cfg,
                                         device)

    history_diffs = [f for f in sorted(set(det_new.history) | set(det_old.history))
                     if not _same(det_new.history.get(f), det_old.history.get(f))]
    pred_diffs = [f for f in sorted(set(preds_new) | set(preds_old))
                  if not _same(preds_new.get(f), preds_old.get(f))]
    states: dict[str, int] = {}
    for value in preds_new.values():
        states[value[3]] = states.get(value[3], 0) + 1
    return {
        "original_ref": ref, "device": str(device), "frames": frames,
        "frame_size": [width, height],
        "history_differences": history_diffs[:20],
        "prediction_differences": pred_diffs[:20],
        "n_history_differences": len(history_diffs),
        "n_prediction_differences": len(pred_diffs),
        "fused_states": states,
        "identical": not history_diffs and not pred_diffs,
    }


def main(argv: list[str] | None = None) -> int:
    """Compare the reviewed module with the author's original on a synthetic clip."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m training.compare_with_original",
        description="Reviewed vs original src/fot_video.py on a synthetic clip.")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref", default=ORIGINAL_REF,
                        help=f"git ref of the original (default {ORIGINAL_REF})")
    args = parser.parse_args(argv)

    result = compare(args.frames, args.width, args.height, args.device, args.seed,
                     args.ref)
    print(json.dumps(result, indent=2))
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
