"""Ball reconstruction with provenance — P1.

Tiers (the raw ATD is never modified):

  SOURCE_ATD        0  raw tracking observation (full evidence)
  SOURCE_DETECTOR   1  reserved for the stretch detector tier (not implemented)
  SOURCE_KINEMATIC  2  cubic-Hermite fill across bracketed gaps <= fill_long_gap_s
  SOURCE_NONE       3  no ball evidence — the decoder carries state through these

Rules:
  * Fills happen only inside the graded halves; a fill never crosses a half boundary.
  * Only estimated positions can be nulled by drop_out_of_pitch; ATD positions are
    flagged only (R: corrections come from code, never by removing raw evidence).
  * Provenance columns: `source`, `age_s` (seconds to the nearest ATD observation),
    `out_of_pitch` (computed before nulling, so it survives the drop).

The fill is Jose's tested design (damped cubic Hermite between bracket anchors with
boundary velocities from the neighbouring known frames, overshoot clipped to the
anchors' bounding box); constants live in config.py with origin=prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CFG

SOURCE_ATD = 0
SOURCE_DETECTOR = 1  # reserved (stretch tier)
SOURCE_KINEMATIC = 2
SOURCE_NONE = 3

SOURCE_NAMES = {
    SOURCE_ATD: "atd",
    SOURCE_DETECTOR: "detector",
    SOURCE_KINEMATIC: "kinematic",
    SOURCE_NONE: "none",
}


@dataclass(frozen=True)
class BallTrack:
    """Reconstructed ball with per-frame provenance. NaN xy => tier `none`."""

    xy: np.ndarray        # (N, 2) float64
    source: np.ndarray    # (N,) int8, one of SOURCE_*
    age_s: np.ndarray     # (N,) float64, seconds to nearest ATD observation (0 on ATD frames)
    out_of_pitch: np.ndarray  # (N,) bool, position outside the margin (pre-null)


def _velocity(sel: np.ndarray, xy: np.ndarray, fps: int) -> np.ndarray:
    """Velocity (units/s) across a chronological span of known frames (>= 2)."""
    if sel.size < 2:
        return np.zeros(2)
    dt = (sel[-1] - sel[0]) / float(fps)
    if dt <= 0:
        return np.zeros(2)
    return (xy[sel[-1]] - xy[sel[0]]) / dt


def build_ball_track(ball: np.ndarray, half: np.ndarray, cfg: dict = CFG) -> BallTrack:
    """Fill bracketed gaps and attach provenance. Pure: inputs are not modified."""
    n = len(ball)
    fps = int(cfg["fps"])
    known = ~np.isnan(ball).any(axis=1)
    known_idx = np.flatnonzero(known)

    xy = ball.copy()                      # ATD values ride along untouched
    source = np.full(n, SOURCE_NONE, dtype=np.int8)
    source[known] = SOURCE_ATD

    # age_s: seconds to the nearest ATD observation in either direction (offline decoder)
    idx = np.arange(n)
    if known_idx.size:
        pos = np.searchsorted(known_idx, idx)
        right = known_idx[np.clip(pos, 0, known_idx.size - 1)]
        left = known_idx[np.clip(pos - 1, 0, known_idx.size - 1)]
        age_s = np.minimum(np.abs(idx - left), np.abs(idx - right)) / float(fps)
    else:
        age_s = np.full(n, np.inf)

    # --- kinematic fill, per half, only bracketed gaps within the time limit ---
    window = int(cfg["fill_vel_window"])
    tau_s = float(cfg["fill_tau_s"])
    long_gap = float(cfg["fill_long_gap_s"])
    pad = float(cfg["fill_pad"])
    pad_frac = float(cfg["fill_pad_frac"])

    if known_idx.size >= 2:
        for h in (1, 2):
            half_frames = np.flatnonzero(half == h)
            if half_frames.size == 0:
                continue
            k = known_idx[(known_idx >= half_frames[0]) & (known_idx <= half_frames[-1])]
            if k.size < 2:
                continue
            for i0, i1 in zip(k[:-1], k[1:]):
                if i1 - i0 < 2:
                    continue
                gap_s = (i1 - i0) / float(fps)
                if gap_s > long_gap:
                    continue  # too old to interpolate: stays `none` for the decoder to carry

                p0, p1 = xy[i0], xy[i1]
                back = k[k <= i0][-window:]   # same half only, chronological, ends at i0
                fwd = k[k >= i1][:window]     # same half only, chronological, starts at i1
                v0 = _velocity(back, ball, fps)
                v1 = _velocity(fwd, ball, fps)
                damp = min(1.0, tau_s / gap_s)

                interior = np.arange(i0 + 1, i1)
                t = ((interior - i0) / float(i1 - i0))[:, None]
                t2, t3 = t * t, t * t * t
                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2
                path = (h00 * p0 + h10 * gap_s * (v0 * damp)
                        + h01 * p1 + h11 * gap_s * (v1 * damp))

                span = np.abs(p1 - p0)
                margin = np.maximum(pad, pad_frac * span)
                path = np.clip(path, np.minimum(p0, p1) - margin, np.maximum(p0, p1) + margin)

                xy[interior] = path
                source[interior] = SOURCE_KINEMATIC

    # --- out-of-pitch flag (computed before nulling, on every positioned frame) ---
    margin = float(cfg["pitch_margin"])
    has_pos = ~np.isnan(xy).any(axis=1)
    outside = ((xy[:, 0] < -margin) | (xy[:, 0] > 1 + margin)
               | (xy[:, 1] < -margin) | (xy[:, 1] > 1 + margin)) & has_pos

    # --- drop only ESTIMATED out-of-pitch positions (ATD is flagged, never dropped) ---
    drop = (source == SOURCE_KINEMATIC) & outside
    xy[drop] = np.nan
    source[drop] = SOURCE_NONE

    return BallTrack(xy=xy, source=source, age_s=age_s, out_of_pitch=outside)


def coverage(track: BallTrack, graded: np.ndarray | None = None) -> dict:
    """Source shares (fractions summing to 1) over all frames and graded frames."""
    out = {}
    for scope, mask in (("all", np.ones(len(track.source), bool)),
                        ("graded", graded if graded is not None else np.ones(len(track.source), bool))):
        total = int(mask.sum()) or 1
        for code, name in SOURCE_NAMES.items():
            out[f"{scope}_{name}"] = float(((track.source == code) & mask).sum()) / total
    return out


def describe(track: BallTrack, graded: np.ndarray | None = None) -> str:
    cov = coverage(track, graded)
    lines = ["ball source shares:"]
    for scope in ("all", "graded"):
        parts = "  ".join(f"{name} {100 * cov[f'{scope}_{name}']:.1f}%"
                          for name in ("atd", "kinematic", "none"))
        lines.append(f"  {scope:6s} {parts}")
    lines.append(f"  out_of_pitch flags: {int(track.out_of_pitch.sum())} frames "
                 f"(atd flagged: {int((track.out_of_pitch & (track.source == SOURCE_ATD)).sum())})")
    return "\n".join(lines)
