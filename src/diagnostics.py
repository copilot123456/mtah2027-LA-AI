"""Aggregate diagnostics on the provided data — GREEN output (data-only, no GT, no video).

Run:  python -m src.diagnostics
Prints coarse aggregates only; nothing frame-level leaves this machine (see dossier.md).
Targets below are the numbers recorded in the tech deep-dive (2026-11-07); the test suite
asserts the same values with tolerances.
"""

from __future__ import annotations

import numpy as np

from src import data as D


def missing_runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of True runs (frame counts)."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def main() -> None:
    m = D.load_arrays()
    print(f"rows: {m.n_frames}  slots: {m.players.shape[1]}")

    ball_vis = ~np.isnan(m.ball).any(axis=1)
    print(f"ball missing overall: {100 * (~ball_vis).mean():.1f}%")
    print(f"ball missing on graded frames: {100 * ((~ball_vis) & m.graded).mean() / m.graded.mean():.1f}%")

    runs = missing_runs((~ball_vis) & m.graded)
    if len(runs):
        print(
            "missing runs on graded frames: n=%d median=%d p90=%d max=%d frames"
            % (len(runs), np.median(runs), np.percentile(runs, 90), runs.max())
        )
        for lo, hi, label in [(1, 25, "<=1s"), (26, 125, "1-5s"), (126, 750, "5-30s"), (751, 10**9, ">30s")]:
            share = 100 * ((runs >= lo) & (runs <= hi)).mean()
            print(f"  {label}: {share:.0f}% of runs")

    nvis = (~np.isnan(m.players)).sum(axis=1)
    g = nvis[m.graded]
    print(f"players visible (graded): mean {g.mean():.1f}  p10 {np.percentile(g, 10):.0f}"
          f"  p50 {np.median(g):.0f}  p90 {np.percentile(g, 90):.0f}  max {g.max()}")

    delta = np.diff(m.ball, axis=0)
    both_present = ~np.isnan(m.ball).any(axis=1)
    pair_ok = both_present[:-1] & both_present[1:]
    speed = np.sqrt(np.nansum(delta[pair_ok] ** 2, axis=1)) * D.FPS
    print(f"ball speed (norm units/s): p50 {np.percentile(speed, 50):.3f}"
          f"  p90 {np.percentile(speed, 90):.3f}  p99 {np.percentile(speed, 99):.3f}")

    df = D.load_tracking()
    for col in ("ball_state", "ball_owning_team_id"):
        print(f"{col}: non-null {100 * df[col].notna().mean():.1f}%")

    hg = D.load_homography()
    print(f"homography: rows {len(hg)}  valid {100 * hg['valid'].mean():.1f}%")

    tags = D.load_smart_tagging()
    print(f"smart tagging: {len(tags)} instances  SET PIECES {int((tags['code'] == 'SET PIECES').sum())}")

    print(f"graded frames: {int(m.graded.sum())}  half1 {int((m.half == 1).sum())}  half2 {int((m.half == 2).sum())}")

    from src.ball_fill import build_ball_track, describe
    from src.emission import build_emission

    track = build_ball_track(m.ball, m.half)
    print(describe(track, m.graded))

    em = build_emission(track, m.players, m.players_s, m.slot_team)
    for code, name in ((0, "atd"), (2, "kinematic"), (3, "none")):
        sel = track.source == code
        if sel.any():
            print(f"  conf[{name}]: mean {em.conf[sel].mean():.3f}")
    print(f"  p_d: p10 {np.percentile(em.p_d, 10):.3f}  p50 {np.percentile(em.p_d, 50):.3f}"
          f"  p90 {np.percentile(em.p_d, 90):.3f}  share>0.5 {100 * (em.p_d > 0.5).mean():.1f}%")
    vis = em.ev.sum(axis=1) > 0
    print(f"  team margin (ball visible): H ahead {100 * (em.margin[vis] > 0).mean():.1f}%"
          f"  A ahead {100 * (em.margin[vis] < 0).mean():.1f}%")

    from src.decoder import decode, segment_stats

    decoded = decode(em, track.source, m.half)
    stats = segment_stats(decoded.labels)
    print(f"  decoded: segments {stats['n_segments']} (possessions {stats['n_possessions']})"
          f"  median {stats['median_len_s']:.1f}s  sub-second {100 * stats['subsecond_share']:.1f}%"
          f"  D share {100 * stats['d_share']:.1f}%  reverted switches {decoded.reverted_switches}")


if __name__ == "__main__":
    main()
