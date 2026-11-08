"""Default pipeline step handlers — thin adapters over the pure stage functions.

Each handler is `fn(ctx, cfg) -> None`: it pulls its inputs from the `Ctx`,
calls the corresponding pure function (unchanged, independently unit-tested)
and stores the result back. Handlers must carry a docstring — enforced by
`src.pipeline.validate_spec` before any step runs.

Swapping a handler for an alternative implementation (e.g. ViT instead of the
current gap fill) happens in `pipeline_spec.py`, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src import data as D
from src.ball_fill import build_ball_track, coverage
from src.decoder import decode, segment_stats
from src.emission import build_emission
from src.x_policy import apply_x_policy

if TYPE_CHECKING:
    from src.pipeline import Ctx


def load(ctx: "Ctx", cfg: dict) -> None:
    """Load and align the match arrays (ATD, players, homography, graded spans)."""
    ctx.arrays = D.load_arrays()


def ball_track(ctx: "Ctx", cfg: dict) -> None:
    """Reconstruct the ball track with provenance tiers (raw ATD + fill tiers)."""
    ctx.track = build_ball_track(ctx.arrays.ball, ctx.arrays.half, cfg)


def emission(ctx: "Ctx", cfg: dict) -> None:
    """Build the per-frame H/A/D log-emission (team kernel, confidence, dead-ball)."""
    ctx.emission = build_emission(
        ctx.track, ctx.arrays.players, ctx.arrays.players_s, ctx.arrays.slot_team, cfg
    )


def decode_step(ctx: "Ctx", cfg: dict) -> None:
    """Per-half Viterbi decode with switch gating and the min-dwell post-pass."""
    ctx.decoded = decode(ctx.emission, ctx.track.source, ctx.arrays.half, cfg)


def x_policy(ctx: "Ctx", cfg: dict) -> None:
    """Apply the abstention policy (a no-op while cfg['x_use'] is False)."""
    ctx.labels = apply_x_policy(
        ctx.decoded.labels, ctx.emission.conf, ctx.arrays.half, cfg
    )


def summarize(ctx: "Ctx", cfg: dict) -> None:
    """Build the run summary: segment stats, gated switches, coverage, graded frames."""
    ctx.summary = {
        **segment_stats(ctx.labels, cfg["fps"]),
        "reverted_switches": ctx.decoded.reverted_switches,
        "coverage": coverage(ctx.track, ctx.arrays.graded),
        "graded_frames": int(ctx.arrays.graded.sum()),
    }
