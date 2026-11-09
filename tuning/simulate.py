"""Missingness simulation: degrade IDSSE ball tracking to challenge-like quality.

The run-length distribution of missing-ball runs is a structural diagnostic of the
provided tracking (GREEN under the output classification: missing counts and gap
distributions, no GT involved). Applying it to external data is how the test bed
exercises our fill + decoder under the input quality the submission will face.

Placement of mask runs on IDSSE is UNIFORM and seeded per match — a documented
simplification (the real missingness correlates with attacking/dead phases, which we
cannot quantify without using the provided match statistics beyond the coarse
distribution).

Stage 4 (E2, pre-declared dossier §7 2026-11-09) adds *phase-weighted* placement on the
same mechanics: start frames are sampled ∝ `w_t = 1 + α·cue_t` where the three cues are
computed from IDSSE itself (possession changes, crowded ball, TacklingGame events).
α = 2 primary, α = 3 sensitivity — both a priori; no provided-match value enters the
weights. The uniform path is unchanged and still reproduces the P5 masks bit-for-bit.
"""

from __future__ import annotations

import numpy as np


def empirical_run_lengths(graded_missing: np.ndarray) -> np.ndarray:
    """Lengths (frames) of True runs in a missing-ball mask."""
    edges = np.diff(np.concatenate(([0], graded_missing.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return stops - starts


def mask_ball(ball: np.ndarray, run_lengths: np.ndarray, rng: np.random.Generator,
              target_fraction: float = 0.431,
              weights: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Copy of `ball` with non-overlapping runs masked to NaN.

    Run lengths are sampled with replacement from the empirical distribution and placed
    at random (rejecting overlaps) until the masked frame fraction reaches
    `target_fraction` or a bounded number of attempts fails. With `weights` (stage 4,
    dossier §7) the start frame is sampled ∝ `weights` instead of uniformly; the default
    `None` keeps the P5 behaviour byte-identical.
    """
    n = len(ball)
    masked = ball.copy()
    blocked = np.zeros(n, dtype=bool)
    n_masked = 0
    n_runs = 0
    target = int(target_fraction * n)
    attempts = 0
    max_attempts = 200_000
    while n_masked < target and attempts < max_attempts:
        attempts += 1
        length = int(rng.choice(run_lengths))
        if length <= 1:
            continue
        if weights is None:
            start = int(rng.integers(0, n - length))
        else:
            seg = np.asarray(weights, dtype=np.float64)[: n - length]
            start = int(rng.choice(n - length, p=seg / seg.sum()))
        if blocked[start:start + length].any():
            continue
        blocked[start:start + length] = True
        masked[start:start + length] = np.nan
        n_masked += length
        n_runs += 1
    stats = {"masked_frames": n_masked, "masked_fraction": n_masked / n,
             "n_runs": n_runs, "attempts": attempts}
    return masked, stats


# ---------------- stage-4 phase cues (E2, dossier §7 pre-declared 2026-11-09) ----------------

def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """True where `mask` is true within ±radius frames."""
    out = mask.copy()
    for k in range(1, max(int(radius), 0) + 1):
        out[k:] |= mask[:-k]
        out[:-k] |= mask[k:]
    return out


def possession_change_cue(letters: np.ndarray, *, half_s: float = 2.0,
                          fps: float = 25.0) -> np.ndarray:
    """cue 1: ±half_s around external possession changes on the *original* letters.

    A change is an alive (H/A) frame that starts the array, follows a non-alive frame,
    or directly follows the other team. Original letters are used on purpose: they are
    the provider-faithful change points; E1's aligned set is a scoring interpretation.
    """
    letters = np.asarray(letters, dtype=object)
    alive = np.isin(letters, ("H", "A"))
    change = np.zeros(len(letters), dtype=bool)
    if len(letters):
        change[0] = alive[0]
        if len(letters) > 1:
            change[1:] = alive[1:] & (~alive[:-1] | (letters[1:] != letters[:-1]))
    return _dilate(change, int(round(half_s * fps)))


def crowding_cue(players: np.ndarray, ball: np.ndarray, *, length_m: float,
                 width_m: float, min_players: int = 3,
                 radius_m: float = 3.0) -> np.ndarray:
    """cue 2: >= `min_players` finite players within `radius_m` of a finite ball.

    Metres use the CFG pitch dimensions — the same convention `emission._team_evidence`
    already applies to IDSSE. Ball missing -> false (the cue is a ball-anchored state).
    """
    n = len(ball)
    cue = np.zeros(n, dtype=bool)
    has_ball = ~np.isnan(ball).any(axis=1)
    if not has_ball.any():
        return cue
    idx = np.flatnonzero(has_ball)
    b = ball[idx]
    p = players[idx]
    dx = (p[..., 0] - b[:, None, 0]) * length_m
    dy = (p[..., 1] - b[:, None, 1]) * width_m
    dist = np.hypot(dx, dy)
    close = np.isfinite(dist) & (dist <= radius_m)
    cue[idx] = close.sum(axis=1) >= min_players
    return cue


def event_cue(rows: np.ndarray, n_frames: int, *, half_s: float = 2.0,
              fps: float = 25.0) -> np.ndarray:
    """cue 3: ±half_s around external event rows (TacklingGame; same a-priori window)."""
    base = np.zeros(n_frames, dtype=bool)
    r = np.asarray(rows, dtype=np.int64)
    r = r[(r >= 0) & (r < n_frames)]
    base[r] = True
    return _dilate(base, int(round(half_s * fps)))


def phase_weights(cues: list[np.ndarray], alpha: float) -> np.ndarray:
    """w_t = 1 + α·cue_t with cue_t = sum of the three indicators (α = 2 primary, 3 sensitivity)."""
    cue = np.zeros(len(cues[0]), dtype=np.int64)
    for c in cues:
        cue += np.asarray(c, dtype=np.int64)
    return 1.0 + float(alpha) * cue.astype(np.float64)


def map_event_rows(ts_us: np.ndarray, frame_id: np.ndarray, period: np.ndarray,
                   anchors: list[tuple[float, int]],
                   event_epochs: list[float]) -> np.ndarray:
    """Map absolute event epochs (s) to tracking rows — pure alignment maths.

    `ts_us` is µs since period start (the IDSSE tracking timestamp, which resets per
    period); `anchors` are (epoch_s, provider_frame) pairs from events carrying
    `StartFrame`. Per-period median offset; nearest in-period timestamp; -1 when no
    anchor exists for an event's period. J03WPY anchor residual std ≈ 1.3 s (dossier §7).
    """
    ts_us = np.asarray(ts_us, dtype=np.float64)
    frame_id = np.asarray(frame_id)
    period = np.asarray(period)
    offsets: dict[int, list[float]] = {}
    for epoch_s, provider_frame in anchors:
        i = int(np.searchsorted(frame_id, provider_frame))
        if i >= len(frame_id) or frame_id[i] != provider_frame:
            continue
        offsets.setdefault(int(period[i]), []).append(float(epoch_s) - ts_us[i] / 1e6)
    out = np.full(len(event_epochs), -1, dtype=np.int64)
    if not offsets:
        return out
    median = {p: float(np.median(v)) for p, v in offsets.items()}
    period_rows = {p: np.flatnonzero(period == p) for p in median}
    period_ts = {p: ts_us[rows] for p, rows in period_rows.items()}
    for k, epoch_s in enumerate(event_epochs):
        best_d, best_row = np.inf, -1
        for p, off in median.items():
            t_rel = (float(epoch_s) - off) * 1e6
            j = int(np.searchsorted(period_ts[p], t_rel))
            for cand in (j - 1, j):                # true nearest, robust to float noise
                if 0 <= cand < len(period_rows[p]):
                    d = abs(float(period_ts[p][cand]) - t_rel)
                    if d < best_d:
                        best_d, best_row = d, int(period_rows[p][cand])
        out[k] = best_row
    return out
