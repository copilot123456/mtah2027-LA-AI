"""Default pipeline step handlers — thin adapters over the pure stage functions.

Each handler is `fn(ctx, cfg) -> None`: it pulls its inputs from the `Ctx`,
calls the corresponding pure function (unchanged, independently unit-tested)
and stores the result back. Handlers must carry a docstring — enforced by
`src.pipeline.validate_spec` before any step runs.

Swapping a handler for an alternative implementation (e.g. ViT instead of the
current gap fill) happens in `pipeline_spec.py`, not here.
"""

from __future__ import annotations    # string annotations for the Ctx hint

from dataclasses import replace       # copy a dataclass with one field swapped
from typing import TYPE_CHECKING      # guards the import-only-for-hints below

from src import data as D             # match file loading
from src.ball_fill import build_ball_track, coverage  # ball reconstruction + coverage
from src.decoder import apply_kickoff_rule, decode, segment_stats  # Viterbi + post-passes
from src.emission import build_emission  # per-frame H/A/D scores
from src.x_policy import apply_x_policy  # abstention policy

if TYPE_CHECKING:
    from src.pipeline import Ctx      # type hint only; never imported at runtime


def load(ctx: "Ctx", cfg: dict) -> None:
    """Load and align the match arrays (ATD, players, homography, graded spans)."""
    ctx.arrays = D.load_arrays()     # read and align all match files


def ball_track(ctx: "Ctx", cfg: dict) -> None:
    """Reconstruct the ball track with provenance tiers (raw ATD + fill tiers)."""
    ctx.track = build_ball_track(ctx.arrays.ball, ctx.arrays.half, cfg)  # raw ATD + fills


def emission(ctx: "Ctx", cfg: dict) -> None:
    """Build the per-frame H/A/D log-emission (team kernel, confidence, dead-ball); the
    player-only kinematic terms are computed only when one of their weights is positive."""
    kin = None                       # player-only terms off unless a weight is positive
    if cfg.get("kin_w_dead", 0.0) > 0 or cfg.get("kin_w_team", 0.0) > 0:
        from src.kinematics import build_kinematics  # imported only when needed

        kin = build_kinematics(ctx.arrays.players, ctx.arrays.players_s,  # ball-less evidence
                               ctx.arrays.slot_team, ctx.arrays.half, cfg)
    ctx.emission = build_emission(   # assemble per-frame H/A/D log-scores
        ctx.track, ctx.arrays.players, ctx.arrays.players_s, ctx.arrays.slot_team, cfg,
        setpiece_starts=ctx.arrays.setpiece_starts, kin=kin,
    )


def decode_step(ctx: "Ctx", cfg: dict) -> None:
    """Per-half Viterbi decode with switch gating and the min-dwell post-pass, then the
    kick-off tag rule (D before a half's kick-off tag; no-op while cfg['kickoff_rule'] is False)."""
    ctx.decoded = decode(ctx.emission, ctx.track.source, ctx.arrays.half, cfg)  # Viterbi + gating
    if cfg["kickoff_rule"]:          # K1: D before a half's kick-off tag
        ctx.decoded = replace(ctx.decoded, labels=apply_kickoff_rule(  # copy, labels rewritten
            ctx.decoded.labels, ctx.arrays.half, ctx.arrays.kickoff_starts, cfg))


def x_policy(ctx: "Ctx", cfg: dict) -> None:
    """Apply the abstention policy (a no-op while cfg['x_use'] is False)."""
    ctx.labels = apply_x_policy(     # decoded labels plus X marks where allowed
        ctx.decoded.labels, ctx.emission.conf, ctx.arrays.half, cfg
    )


def summarize(ctx: "Ctx", cfg: dict) -> None:
    """Build the run summary: segment stats, gated switches, coverage, graded frames."""
    ctx.summary = {
        **segment_stats(ctx.labels, cfg["fps"]),  # segment counts and lengths
        "reverted_switches": ctx.decoded.reverted_switches,  # switches the gate undid
        "coverage": coverage(ctx.track, ctx.arrays.graded),  # track coverage, graded frames
        "graded_frames": int(ctx.arrays.graded.sum()),  # frames under evaluation
    }
