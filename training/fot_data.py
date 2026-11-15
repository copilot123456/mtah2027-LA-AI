"""Training windows for FoT v2, built with the inference code's own feature functions.

FoT v2 (``src.fot_video.build_fot_model``) refines a sequence of detector outputs: it
sees 64 steps of 12 features and predicts a corrected ball position plus a "hit"
logit per step. The model can only be as good as the match between what it is trained
on and what inference feeds it, so nothing about the input is re-implemented here:

* the per-frame detection is the top candidate above ``FAR_CONF_FLOOR`` — exactly what
  ``PitchKinematicGate.step`` returns when a frame has no homography, which is the
  situation of an external game (:func:`detections_from_candidates`);
* thresholding and isolated-anchor pruning come from
  ``src.fot_video.prepare_fot_observations`` and the 12 features from
  ``src.fot_video.build_window_features`` — the same two calls inference makes;
* sequences are sub-sampled by ``frame_step`` (2, as recorded in
  ``models/fold0_train_meta.json``), one sequence per phase, mirroring the even / odd
  passes of ``fuse_predictions``.

The targets are the annotated ball positions in the same normalised image
coordinates (x / frame width, y / frame height), with a mask for the steps where the
ball is annotated.

Pure numpy; torch is only needed by ``training.train_fot``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import src.fot_video as F


@dataclass(frozen=True)
class WindowSet:
    """Stacked FoT training windows of one game."""

    feats: np.ndarray        # (n, W, 12) float32 — model input
    baseline: np.ndarray     # (n, W, 2) float32 — interpolation the model corrects
    target: np.ndarray       # (n, W, 2) float32 — annotated position, normalised
    target_mask: np.ndarray  # (n, W) bool — step has an annotated ball
    observed: np.ndarray     # (n, W) bool — step had a usable detection
    frame_w: int
    frame_h: int

    def __len__(self) -> int:
        return int(self.feats.shape[0])


def detections_from_candidates(cands: pd.DataFrame, frames,
                               min_conf: float = F.FAR_CONF_FLOOR) -> list[tuple]:
    """Per evaluated frame, the detection inference would keep without a homography.

    Returns ``[(frame, gx, gy, conf), ...]`` in frame order, with NaN / 0.0 where no
    candidate reaches ``min_conf`` — the same shape as ``DetectionPass.history``
    values. ``frames`` is the full list of frames the detector was run on, so frames
    with no candidate at all are still present.
    """
    frames = sorted(int(f) for f in frames)
    kept = cands[cands["conf"] >= min_conf]
    top = (kept.sort_values(["frame", "conf", "gx", "gy"],
                            ascending=[True, False, True, True])
               .drop_duplicates("frame").set_index("frame"))
    out = []
    for frame in frames:
        if frame in top.index:
            row = top.loc[frame]
            out.append((frame, float(row["gx"]), float(row["gy"]), float(row["conf"])))
        else:
            out.append((frame, float("nan"), float("nan"), 0.0))
    return out


def drop_observations(valid: np.ndarray, rng: np.random.Generator,
                      gap_prob: float = 0.5, max_gap: int = 6) -> np.ndarray:
    """Copy of ``valid`` with one random run of observations removed (an occlusion).

    With probability ``gap_prob`` a run of 1..``max_gap`` consecutive steps is
    switched off, so the model sees — and is supervised on — gaps it must bridge.
    The first and last observation are never removed: a gap with no anchor on one
    side is not something inference asks the model to fill.
    """
    out = valid.copy()
    observed = np.flatnonzero(valid)
    if len(observed) < 3 or rng.random() >= gap_prob:
        return out
    length = int(rng.integers(1, max_gap + 1))
    start = int(rng.integers(observed[0] + 1, max(observed[0] + 2, observed[-1])))
    stop = min(start + length, int(observed[-1]))
    out[start:stop] = False
    return out


def windows_from_sequence(dets: list[tuple], truth: pd.DataFrame, frame_w: int,
                          frame_h: int, profile: str = "max_coverage",
                          window: int = F.WINDOW_SIZE, hop: int = 16,
                          rng: np.random.Generator | None = None,
                          gap_prob: float = 0.0, max_gap: int = 6) -> dict:
    """Windows of one already sub-sampled detection sequence.

    ``dets`` is ``[(frame, gx, gy, conf), ...]``. Windows slide by ``hop`` steps (16,
    the stride of inference's overlap-add). A window with no annotated ball is
    skipped: it carries no supervision. With ``rng`` and ``gap_prob > 0`` each window
    gets a random occlusion (training only).
    """
    obs_x, obs_y, scores, valid = F.prepare_fot_observations(dets, frame_w, frame_h,
                                                            profile)
    n = len(dets)
    truth_xy = truth.set_index("frame")[["x", "y"]]
    frames = [int(d[0]) for d in dets]
    target = np.full((n, 2), np.nan, dtype=np.float32)
    known = [i for i, f in enumerate(frames) if f in truth_xy.index]
    if known:
        rows = truth_xy.loc[[frames[i] for i in known]].to_numpy(dtype=np.float64)
        target[known, 0] = rows[:, 0] / frame_w
        target[known, 1] = rows[:, 1] / frame_h
    target_mask = ~np.isnan(target).any(axis=1)

    feats, base, tgt, tmask, observed = [], [], [], [], []
    if n < window:
        starts = []
    else:
        starts = list(range(0, n - window + 1, hop))
        if starts[-1] != n - window:
            starts.append(n - window)
    for s0 in starts:
        sl = slice(s0, s0 + window)
        if not target_mask[sl].any():
            continue
        v = valid[sl]
        if rng is not None and gap_prob > 0:
            v = drop_observations(v, rng, gap_prob, max_gap)
        f, b = F.build_window_features(obs_x[sl], obs_y[sl], scores[sl], v)
        feats.append(f.astype(np.float32))
        base.append(b.astype(np.float32))
        tgt.append(np.nan_to_num(target[sl], nan=0.0))
        tmask.append(target_mask[sl])
        observed.append(v.copy())
    return {"feats": feats, "baseline": base, "target": tgt, "target_mask": tmask,
            "observed": observed}


def windows_for_game(cands: pd.DataFrame, frames, truth: pd.DataFrame, frame_w: int,
                     frame_h: int, frame_step: int = 2, profile: str = "max_coverage",
                     window: int = F.WINDOW_SIZE, hop: int = 16, seed: int | None = None,
                     gap_prob: float = 0.0, max_gap: int = 6) -> WindowSet:
    """All FoT windows of one game, over every sub-sampling phase.

    ``frames`` are the frames the detector was run on (consecutive frames give the
    faithful picture; if detection itself was sub-sampled, pass ``frame_step=1`` and
    say so in the run notes). ``seed`` switches the occlusion augmentation on.
    """
    dets = detections_from_candidates(cands, frames)
    rng = np.random.default_rng(seed) if seed is not None else None
    parts = {"feats": [], "baseline": [], "target": [], "target_mask": [], "observed": []}
    for phase in range(max(1, int(frame_step))):
        piece = windows_from_sequence(dets[phase::frame_step], truth, frame_w, frame_h,
                                      profile, window, hop, rng, gap_prob, max_gap)
        for key in parts:
            parts[key].extend(piece[key])

    def stack(key, shape, dtype):
        if not parts[key]:
            return np.zeros((0,) + shape, dtype=dtype)
        return np.stack(parts[key]).astype(dtype)

    return WindowSet(
        feats=stack("feats", (window, 12), np.float32),
        baseline=stack("baseline", (window, 2), np.float32),
        target=stack("target", (window, 2), np.float32),
        target_mask=stack("target_mask", (window,), bool),
        observed=stack("observed", (window,), bool),
        frame_w=int(frame_w), frame_h=int(frame_h))


def concat(sets: list[WindowSet]) -> WindowSet:
    """Join the windows of several games (they must share a frame size)."""
    sets = [s for s in sets if len(s)]
    if not sets:
        raise ValueError("no training windows: check the candidates and annotations")
    sizes = {(s.frame_w, s.frame_h) for s in sets}
    if len(sizes) != 1:
        raise ValueError(f"games have different frame sizes {sorted(sizes)}; the hit"
                         " radius is defined in pixels, so train per frame size")
    return WindowSet(
        feats=np.concatenate([s.feats for s in sets]),
        baseline=np.concatenate([s.baseline for s in sets]),
        target=np.concatenate([s.target for s in sets]),
        target_mask=np.concatenate([s.target_mask for s in sets]),
        observed=np.concatenate([s.observed for s in sets]),
        frame_w=sets[0].frame_w, frame_h=sets[0].frame_h)


def bridgeable_mask(observed: np.ndarray, max_gap: int) -> np.ndarray:
    """Steps inference could bridge: inside a run of at most ``max_gap`` unobserved
    steps with an observation on BOTH sides (``run_fot_v2_sequence``'s rule).

    ``observed`` is ``(n, W)`` or ``(W,)``. Leading / trailing gaps and gaps longer
    than ``max_gap`` are never bridged by FoT, so judging the model on them measures
    something inference never asks of it.
    """
    observed = np.asarray(observed, dtype=bool)
    flat = observed.reshape(-1, observed.shape[-1])
    out = np.zeros_like(flat)
    for row, obs in enumerate(flat):
        index = 0
        width = len(obs)
        while index < width:
            if obs[index]:
                index += 1
                continue
            start = index
            while index < width and not obs[index]:
                index += 1
            if start > 0 and index < width and index - start <= max_gap:
                out[row, start:index] = True
    return out.reshape(observed.shape)


def pixel_error(pred_xy: np.ndarray, target_xy: np.ndarray, frame_w: int,
                frame_h: int) -> np.ndarray:
    """Per-step distance in pixels between normalised predictions and targets."""
    dx = (pred_xy[..., 0] - target_xy[..., 0]) * frame_w
    dy = (pred_xy[..., 1] - target_xy[..., 1]) * frame_h
    return np.hypot(dx, dy)
