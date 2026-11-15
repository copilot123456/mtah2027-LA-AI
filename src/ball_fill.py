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

from __future__ import annotations  # postponed type hints

from dataclasses import dataclass  # frozen result container

import numpy as np  # array maths

from config import CFG  # default config values

SOURCE_ATD = 0  # observed tracking
SOURCE_DETECTOR = 1  # reserved (stretch tier)
SOURCE_KINEMATIC = 2  # interpolated fill
SOURCE_NONE = 3  # no ball evidence

# tier codes mapped to report names
SOURCE_NAMES = {
    SOURCE_ATD: "atd",
    SOURCE_DETECTOR: "detector",
    SOURCE_KINEMATIC: "kinematic",
    SOURCE_NONE: "none",
}


@dataclass(frozen=True)  # read-only per-frame provenance bundle
class BallTrack:
    """Reconstructed ball with per-frame provenance. NaN xy => tier `none`."""

    xy: np.ndarray        # (N, 2) float64
    source: np.ndarray    # (N,) int8, one of SOURCE_*
    age_s: np.ndarray     # (N,) float64, seconds to nearest ATD observation (0 on ATD frames)
    out_of_pitch: np.ndarray  # (N,) bool, position outside the margin (pre-null)


def _velocity(sel: np.ndarray, xy: np.ndarray, fps: int) -> np.ndarray:
    """Velocity (units/s) across a chronological span of known frames (>= 2)."""
    if sel.size < 2:
        return np.zeros(2)  # fewer than 2 frames: no motion
    dt = (sel[-1] - sel[0]) / float(fps)  # elapsed seconds across the span
    if dt <= 0:
        return np.zeros(2)  # degenerate span: no motion
    return (xy[sel[-1]] - xy[sel[0]]) / dt  # displacement / elapsed time


def build_ball_track(ball: np.ndarray, half: np.ndarray, cfg: dict = CFG) -> BallTrack:
    """Fill bracketed gaps and attach provenance. Pure: inputs are not modified."""
    n = len(ball)                        # frame count
    fps = int(cfg["fps"])                # frames per second
    known = ~np.isnan(ball).any(axis=1)  # frames with a real ATD position
    known_idx = np.flatnonzero(known)    # their indices

    xy = ball.copy()                      # ATD values ride along untouched
    source = np.full(n, SOURCE_NONE, dtype=np.int8)  # start: no evidence anywhere
    source[known] = SOURCE_ATD            # observed frames get the ATD tier

    # age_s: seconds to the nearest ATD observation in either direction (offline decoder)
    idx = np.arange(n)                   # frame numbers
    if known_idx.size:
        pos = np.searchsorted(known_idx, idx)  # first ATD at or after each frame
        right = known_idx[np.clip(pos, 0, known_idx.size - 1)]  # nearest ATD on the right
        left = known_idx[np.clip(pos - 1, 0, known_idx.size - 1)]  # nearest ATD on the left
        age_s = np.minimum(np.abs(idx - left), np.abs(idx - right)) / float(fps)  # closer side, in seconds
    else:
        age_s = np.full(n, np.inf)        # no ATD at all: age is infinite

    # --- kinematic fill, per half, only bracketed gaps within the time limit ---
    window = int(cfg["fill_vel_window"])   # frames used for endpoint velocity
    tau_s = float(cfg["fill_tau_s"])       # velocity damping time (s)
    long_gap = float(cfg["fill_long_gap_s"])  # longest gap still interpolated
    pad = float(cfg["fill_pad"])           # absolute clip margin (normalized)
    pad_frac = float(cfg["fill_pad_frac"])  # clip margin as share of gap span

    if known_idx.size >= 2:               # need two observations to fill anything
        for h in (1, 2):                  # process each half separately
            half_frames = np.flatnonzero(half == h)  # frames belonging to this half
            if half_frames.size == 0:
                continue                  # empty half: nothing to do
            k = known_idx[(known_idx >= half_frames[0]) & (known_idx <= half_frames[-1])]  # ATD inside this half
            if k.size < 2:
                continue                  # fewer than 2 anchors: no bracket
            for i0, i1 in zip(k[:-1], k[1:]):  # each consecutive pair brackets a gap
                if i1 - i0 < 2:
                    continue              # neighbours: no interior frames to fill
                gap_s = (i1 - i0) / float(fps)  # gap length in seconds
                if gap_s > long_gap:
                    continue  # too old to interpolate: stays `none` for the decoder to carry

                p0, p1 = xy[i0], xy[i1]   # gap endpoint positions
                back = k[k <= i0][-window:]   # same half only, chronological, ends at i0
                fwd = k[k >= i1][:window]     # same half only, chronological, starts at i1
                v0 = _velocity(back, ball, fps)  # velocity entering the gap
                v1 = _velocity(fwd, ball, fps)   # velocity leaving the gap
                damp = min(1.0, tau_s / gap_s)   # long gaps use gentler velocities

                interior = np.arange(i0 + 1, i1)  # frames to interpolate
                t = ((interior - i0) / float(i1 - i0))[:, None]  # normalized time 0..1
                t2, t3 = t * t, t * t * t   # t squared and cubed
                h00 = 2 * t3 - 3 * t2 + 1   # Hermite basis: weight of p0
                h10 = t3 - 2 * t2 + t       # weight of left velocity
                h01 = -2 * t3 + 3 * t2      # weight of p1
                h11 = t3 - t2               # weight of right velocity
                path = (h00 * p0 + h10 * gap_s * (v0 * damp)  # blend endpoints
                        + h01 * p1 + h11 * gap_s * (v1 * damp))  # and velocities

                span = np.abs(p1 - p0)     # per-axis gap size
                margin = np.maximum(pad, pad_frac * span)  # clip box grows with the gap
                path = np.clip(path, np.minimum(p0, p1) - margin, np.maximum(p0, p1) + margin)  # clip overshoot

                xy[interior] = path        # write estimated positions
                source[interior] = SOURCE_KINEMATIC  # mark them kinematic tier

    # --- out-of-pitch flag (computed before nulling, on every positioned frame) ---
    margin = float(cfg["pitch_margin"])   # allowed overhang past pitch edge
    has_pos = ~np.isnan(xy).any(axis=1)   # frames that hold a position
    outside = ((xy[:, 0] < -margin) | (xy[:, 0] > 1 + margin)  # past x bounds
               | (xy[:, 1] < -margin) | (xy[:, 1] > 1 + margin)) & has_pos  # or y bounds

    # --- drop only ESTIMATED out-of-pitch positions (ATD is flagged, never dropped) ---
    drop = (source == SOURCE_KINEMATIC) & outside  # estimated and outside
    xy[drop] = np.nan                   # null the position
    source[drop] = SOURCE_NONE          # back to no evidence

    return BallTrack(xy=xy, source=source, age_s=age_s, out_of_pitch=outside)  # bundle provenance


def coverage(track: BallTrack, graded: np.ndarray | None = None) -> dict:
    """Source shares (fractions summing to 1) over all frames and graded frames."""
    out = {}                             # share per scope and tier
    for scope, mask in (("all", np.ones(len(track.source), bool)),  # every frame
                        ("graded", graded if graded is not None else np.ones(len(track.source), bool))):  # graded frames only
        total = int(mask.sum()) or 1     # guard against empty mask
        for code, name in SOURCE_NAMES.items():
            out[f"{scope}_{name}"] = float(((track.source == code) & mask).sum()) / total  # tier share
    return out


def describe(track: BallTrack, graded: np.ndarray | None = None) -> str:
    cov = coverage(track, graded)         # shares for both scopes
    lines = ["ball source shares:"]       # report header
    for scope in ("all", "graded"):       # one row per scope
        parts = "  ".join(f"{name} {100 * cov[f'{scope}_{name}']:.1f}%"  # three percentages
                          for name in ("atd", "kinematic", "none"))
        lines.append(f"  {scope:6s} {parts}")
    lines.append(f"  out_of_pitch flags: {int(track.out_of_pitch.sum())} frames "  # flag totals
                 f"(atd flagged: {int((track.out_of_pitch & (track.source == SOURCE_ATD)).sum())})")  # ATD subset
    return "\n".join(lines)               # join rows into one string
