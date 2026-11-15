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

from __future__ import annotations  # postponed annotation evaluation (PEP 563)

import numpy as np  # array maths for masks and phase cues


def empirical_run_lengths(graded_missing: np.ndarray) -> np.ndarray:
    """Lengths (frames) of True runs in a missing-ball mask."""
    edges = np.diff(np.concatenate(([0], graded_missing.astype(np.int8), [0])))  # +1 start, -1 end
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)  # boundary frame indices
    return stops - starts  # each run's length in frames


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
    n = len(ball)  # frames in this match
    masked = ball.copy()  # work on a copy; input stays untouched
    blocked = np.zeros(n, dtype=bool)  # frames already taken by a run
    n_masked = 0  # frames hidden so far
    n_runs = 0  # runs placed so far
    target = int(target_fraction * n)  # frames to hide before stopping
    attempts = 0  # loop iterations used
    max_attempts = 200_000  # budget so an unreachable target can't hang
    while n_masked < target and attempts < max_attempts:  # fill until target or budget spent
        attempts += 1  # count this attempt
        length = int(rng.choice(run_lengths))  # sample one run length (with replacement)
        if length <= 1:
            continue  # skip trivial runs
        if weights is None:
            start = int(rng.integers(0, n - length))  # uniform start (P5 behaviour)
        else:
            seg = np.asarray(weights, dtype=np.float64)[: n - length]  # legal starts for this length
            start = int(rng.choice(n - length, p=seg / seg.sum()))  # start sampled ∝ weight
        if blocked[start:start + length].any():
            continue  # reject overlaps with earlier runs
        blocked[start:start + length] = True  # claim these frames
        masked[start:start + length] = np.nan  # hide the ball there
        n_masked += length  # frames are non-overlapping, so count them fully
        n_runs += 1  # one more run placed
    stats = {"masked_frames": n_masked, "masked_fraction": n_masked / n,  # summary of the mask
             "n_runs": n_runs, "attempts": attempts}
    return masked, stats  # masked ball plus its stats


# ---------------- stage-4 phase cues (E2, dossier §7 pre-declared 2026-11-09) ----------------

def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """True where `mask` is true within ±radius frames."""
    out = mask.copy()  # start from the input flags
    for k in range(1, max(int(radius), 0) + 1):  # offsets 1..radius frames
        out[k:] |= mask[:-k]  # spread each hit k frames forward
        out[:-k] |= mask[k:]  # spread each hit k frames backward
    return out  # widened flag array


def possession_change_cue(letters: np.ndarray, *, half_s: float = 2.0,
                          fps: float = 25.0) -> np.ndarray:
    """cue 1: ±half_s around external possession changes on the *original* letters.

    A change is an alive (H/A) frame that starts the array, follows a non-alive frame,
    or directly follows the other team. Original letters are used on purpose: they are
    the provider-faithful change points; E1's aligned set is a scoring interpretation.
    """
    letters = np.asarray(letters, dtype=object)  # normalise letter array
    alive = np.isin(letters, ("H", "A"))  # frames with a live team
    change = np.zeros(len(letters), dtype=bool)  # raw change points, before widening
    if len(letters):
        change[0] = alive[0]  # a live first frame starts a spell
        if len(letters) > 1:
            change[1:] = alive[1:] & (~alive[:-1] | (letters[1:] != letters[:-1]))  # alive after dead, or team flip
    return _dilate(change, int(round(half_s * fps)))  # widen to a ±half_s window


def crowding_cue(players: np.ndarray, ball: np.ndarray, *, length_m: float,
                 width_m: float, min_players: int = 3,
                 radius_m: float = 3.0) -> np.ndarray:
    """cue 2: >= `min_players` finite players within `radius_m` of a finite ball.

    Metres use the CFG pitch dimensions — the same convention `emission._team_evidence`
    already applies to IDSSE. Ball missing -> false (the cue is a ball-anchored state).
    """
    n = len(ball)  # frames in the match
    cue = np.zeros(n, dtype=bool)  # default: no crowding
    has_ball = ~np.isnan(ball).any(axis=1)  # frames where a ball position exists
    if not has_ball.any():
        return cue  # no ball anywhere -> cue stays all-False
    idx = np.flatnonzero(has_ball)  # indices of those frames
    b = ball[idx]  # ball positions there
    p = players[idx]  # player positions there
    dx = (p[..., 0] - b[:, None, 0]) * length_m  # x gap scaled to metres
    dy = (p[..., 1] - b[:, None, 1]) * width_m  # y gap scaled to metres
    dist = np.hypot(dx, dy)  # player-to-ball distance in metres
    close = np.isfinite(dist) & (dist <= radius_m)  # players inside the radius
    cue[idx] = close.sum(axis=1) >= min_players  # crowding = enough close players
    return cue  # crowding flags for this match


def event_cue(rows: np.ndarray, n_frames: int, *, half_s: float = 2.0,
              fps: float = 25.0) -> np.ndarray:
    """cue 3: ±half_s around external event rows (TacklingGame; same a-priori window)."""
    base = np.zeros(n_frames, dtype=bool)  # no events by default
    r = np.asarray(rows, dtype=np.int64)  # event rows as integers
    r = r[(r >= 0) & (r < n_frames)]  # drop rows outside the match
    base[r] = True  # flag the event frames
    return _dilate(base, int(round(half_s * fps)))  # widen to a ±half_s window


def phase_weights(cues: list[np.ndarray], alpha: float) -> np.ndarray:
    """w_t = 1 + α·cue_t with cue_t = sum of the three indicators (α = 2 primary, 3 sensitivity)."""
    cue = np.zeros(len(cues[0]), dtype=np.int64)  # accumulator over frames
    for c in cues:
        cue += np.asarray(c, dtype=np.int64)  # sum the three indicators
    return 1.0 + float(alpha) * cue.astype(np.float64)  # w_t = 1 + α·cue_t


def map_event_rows(ts_us: np.ndarray, frame_id: np.ndarray, period: np.ndarray,
                   anchors: list[tuple[float, int]],
                   event_epochs: list[float]) -> np.ndarray:
    """Map absolute event epochs (s) to tracking rows — pure alignment maths.

    `ts_us` is µs since period start (the IDSSE tracking timestamp, which resets per
    period); `anchors` are (epoch_s, provider_frame) pairs from events carrying
    `StartFrame`. Per-period median offset; nearest in-period timestamp; -1 when no
    anchor exists for an event's period. J03WPY anchor residual std ≈ 1.3 s (dossier §7).
    """
    ts_us = np.asarray(ts_us, dtype=np.float64)  # timestamps as float µs
    frame_id = np.asarray(frame_id)  # provider frame numbers
    period = np.asarray(period)  # period id per row
    offsets: dict[int, list[float]] = {}  # per-period epoch-minus-timestamp offsets
    for epoch_s, provider_frame in anchors:
        i = int(np.searchsorted(frame_id, provider_frame))  # row of this anchor frame
        if i >= len(frame_id) or frame_id[i] != provider_frame:
            continue  # anchor frame not present -> skip it
        offsets.setdefault(int(period[i]), []).append(float(epoch_s) - ts_us[i] / 1e6)  # record offset
    out = np.full(len(event_epochs), -1, dtype=np.int64)  # default: unmapped (-1)
    if not offsets:
        return out  # no anchors at all -> nothing maps
    median = {p: float(np.median(v)) for p, v in offsets.items()}  # robust offset per period
    period_rows = {p: np.flatnonzero(period == p) for p in median}  # row indices per period
    period_ts = {p: ts_us[rows] for p, rows in period_rows.items()}  # their timestamps
    for k, epoch_s in enumerate(event_epochs):
        best_d, best_row = np.inf, -1  # nearest row found so far
        for p, off in median.items():
            t_rel = (float(epoch_s) - off) * 1e6  # event time on this period's clock
            j = int(np.searchsorted(period_ts[p], t_rel))  # insertion point
            for cand in (j - 1, j):                # true nearest, robust to float noise
                if 0 <= cand < len(period_rows[p]):
                    d = abs(float(period_ts[p][cand]) - t_rel)  # gap to that row
                    if d < best_d:
                        best_d, best_row = d, int(period_rows[p][cand])  # keep the closer row
        out[k] = best_row  # store nearest row (or -1)
    return out  # row per event, -1 if none


# ---------------- stage-5 dead-cover placement (dossier §7 P7, pre-declared 2026-11-14) -----

def mask_ball_dead_cover(ball: np.ndarray, letters: np.ndarray, run_lengths: np.ndarray,
                         rng: np.random.Generator, q: float, *,
                         target_fraction: float = 0.431, min_dead_frames: int = 50,
                         max_attempts: int = 200_000) -> tuple[np.ndarray, dict]:
    """Copy of `ball` with ball loss coupled to dead ball.

    Every maximal dead run (`letters == "D"`) of at least `min_dead_frames` is masked
    entirely with probability `q` — one draw per run, in chronological order. P5 run
    lengths are then placed uniformly at random, rejecting overlaps, until
    `target_fraction` of the in-match frames (`letters != "X"`) is masked. `q` is a priori
    (0.5 moderate, 0.9 strong); it models a tracker that loses the ball when it leaves
    play, which the uniform and phase-weighted placements do not.
    """
    letters = np.asarray(letters, dtype=object)  # normalise letter array
    n = len(ball)  # frames in the match
    dead = letters == "D"  # dead-ball frames
    in_match = letters != "X"  # frames that count toward the target
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))  # dead-run boundaries
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)  # run start/stop frames
    blocked = np.zeros(n, dtype=bool)  # every frame to hide (covers + P5 runs)
    covered_runs = 0  # dead runs actually covered
    for start, stop in zip(starts, stops):
        if stop - start >= min_dead_frames and rng.random() < q:  # long enough and drawn
            blocked[start:stop] = True  # cover the whole dead run
            covered_runs += 1  # one more covered run
    goal = int(target_fraction * int(in_match.sum()))  # in-match frames left to hide
    n_masked = int((blocked & in_match).sum())  # already covered by dead runs
    attempts = 0  # loop iterations used
    while n_masked < goal and attempts < max_attempts:  # fill until goal or budget spent
        attempts += 1  # count this attempt
        length = int(rng.choice(run_lengths))  # sample a P5 run length
        if length <= 1:
            continue  # skip trivial runs
        start = int(rng.integers(0, n - length))  # uniform start (P5 placement)
        if blocked[start:start + length].any():
            continue  # reject overlaps with existing coverage
        blocked[start:start + length] = True  # claim these frames
        n_masked += int(in_match[start:start + length].sum())  # count in-match frames only
    masked = ball.copy()  # work on a copy
    masked[blocked] = np.nan  # hide the ball on every blocked frame
    stats = {"masked_frames": int(blocked.sum()),  # all blocked frames
             "masked_fraction": float(blocked[in_match].mean()) if in_match.any() else 0.0,  # in-match share
             "n_runs": covered_runs, "attempts": attempts}  # dead runs covered, attempts used
    return masked, stats  # masked ball plus its stats
