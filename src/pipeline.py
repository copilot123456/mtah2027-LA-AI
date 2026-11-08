"""End-to-end pipeline: data -> ball fill -> emission -> decode -> X policy."""

from __future__ import annotations

import numpy as np

from config import CFG
from src import data as D
from src.ball_fill import build_ball_track, coverage
from src.decoder import decode, segment_stats
from src.emission import build_emission
from src.x_policy import apply_x_policy


def run(cfg: dict = CFG) -> tuple[np.ndarray, dict]:
    """Labels for all 199,319 frames (X outside graded spans) + a summary dict.

    The only inputs are the provided match files and cfg; nothing here reads
    ground truth or the video.
    """
    arrays = D.load_arrays()
    track = build_ball_track(arrays.ball, arrays.half, cfg)
    emission = build_emission(track, arrays.players, arrays.players_s, arrays.slot_team, cfg)
    decoded = decode(emission, track.source, arrays.half, cfg)
    labels = apply_x_policy(decoded.labels, emission.conf, arrays.half, cfg)

    summary = {
        **segment_stats(labels, cfg["fps"]),
        "reverted_switches": decoded.reverted_switches,
        "coverage": coverage(track, arrays.graded),
        "graded_frames": int(arrays.graded.sum()),
    }
    return labels, summary
