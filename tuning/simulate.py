"""Missingness simulation: degrade IDSSE ball tracking to challenge-like quality.

The run-length distribution of missing-ball runs is a structural diagnostic of the
provided tracking (GREEN under the output classification: missing counts and gap
distributions, no GT involved). Applying it to external data is how the test bed
exercises our fill + decoder under the input quality the submission will face.

Placement of mask runs on IDSSE is UNIFORM and seeded per match — a documented
simplification (the real missingness correlates with attacking/dead phases, which we
cannot quantify without using the provided match statistics beyond the coarse
distribution).
"""

from __future__ import annotations

import numpy as np


def empirical_run_lengths(graded_missing: np.ndarray) -> np.ndarray:
    """Lengths (frames) of True runs in a missing-ball mask."""
    edges = np.diff(np.concatenate(([0], graded_missing.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return stops - starts


def mask_ball(ball: np.ndarray, run_lengths: np.ndarray, rng: np.random.Generator,
              target_fraction: float = 0.431) -> tuple[np.ndarray, dict]:
    """Copy of `ball` with non-overlapping runs masked to NaN.

    Run lengths are sampled with replacement from the empirical distribution and placed
    uniformly at random (rejecting overlaps) until the masked frame fraction reaches
    `target_fraction` or a bounded number of attempts fails.
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
        start = int(rng.integers(0, n - length))
        if blocked[start:start + length].any():
            continue
        blocked[start:start + length] = True
        masked[start:start + length] = np.nan
        n_masked += length
        n_runs += 1
    stats = {"masked_frames": n_masked, "masked_fraction": n_masked / n,
             "n_runs": n_runs, "attempts": attempts}
    return masked, stats
