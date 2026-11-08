"""Abstention policy — P4.

Why X exists (EVALUATION.md/FAQ): under spellF1 an X is forward-filled with the last
label in the same half, so it costs whatever repeating that answer costs — free inside
a continuing possession, expensive across a hand-over. Under macroF1 it is a miss.

Why ours is OFF by default (`x_use=False`):

  For a path already smoothed by switch penalties + gating + min-dwell, emitting the
  decoded label and abstaining produce the **same filled array** inside a segment
  (fill copies exactly that label), so spellF1 cannot improve — while macroF1 charges
  a miss where emitting could score a true positive. Emitting therefore weakly
  dominates on both metrics for our architecture; X is the insurance of flicker-prone,
  per-frame systems. Evidence-free stretches (the case X was designed for) are already
  collapsed by switch gating. The rule below is kept because it is cheap to ablate:
  P5 can enable it on IDSSE; if it wins there, the flag flips with evidence.

The rule when enabled (docs-derived, a priori): X on a graded frame iff
  * confidence <= x_conf_max (no trustworthy ball evidence), AND
  * the frame is >= x_stable_s from any decoded switch, half start or half end
    (never near a hand-over — the fill would bleed the old label across it),
and never on a whole half (a half the submission abstains on entirely has nothing to
carry and scores zero spells).
"""

from __future__ import annotations

import numpy as np

from config import CFG
from src.decoder import CODE_X


def _distance_to_nearest_boundary(t_len: int, switches: np.ndarray) -> np.ndarray:
    """Frames to the nearest switch or span edge, whichever is closer.

    A span without switches is handled by the same code path: the switch terms stay
    infinite and the edge terms take over (an early `return inf` here used to skip the
    edges, so a switch-free half abstained right up to its first and last frame —
    dormant under the shipped x_use=False, fixed in the 2026-11-07 audit).
    """
    switches = np.asarray(switches, dtype=np.int64)
    pos = np.arange(t_len)
    left_idx = np.searchsorted(switches, pos, side="right") - 1
    right_idx = np.searchsorted(switches, pos, side="left")
    dist = np.full(t_len, np.inf)
    has_left = left_idx >= 0
    has_right = right_idx < len(switches)
    dist[has_left] = np.minimum(dist[has_left], pos[has_left] - switches[left_idx[has_left]])
    dist[has_right] = np.minimum(dist[has_right], switches[right_idx[has_right]] - pos[has_right])
    # distance to the span edges
    dist = np.minimum(dist, pos)
    dist = np.minimum(dist, t_len - 1 - pos)
    return dist


def apply_x_policy(labels: np.ndarray, conf: np.ndarray, half: np.ndarray,
                   cfg: dict = CFG) -> np.ndarray:
    """Return labels with X applied per the module rule (no-op unless cfg['x_use'])."""
    if not cfg.get("x_use", False):
        return labels.copy()

    out = labels.copy()
    stable_frames = int(round(cfg["x_stable_s"] * cfg["fps"]))
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        lab = out[idx]
        switches = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        dist = _distance_to_nearest_boundary(len(lab), switches)
        candidate = (conf[idx] <= cfg["x_conf_max"]) & (dist >= stable_frames)
        if candidate.all():
            continue  # never abstain on an entire half
        out[idx[candidate]] = CODE_X
    return out
