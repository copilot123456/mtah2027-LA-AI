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

from __future__ import annotations  # postponed annotation evaluation

import numpy as np  # array math

from config import CFG  # default config (x_use off)
from src.decoder import CODE_X  # integer code for abstain


def _distance_to_nearest_boundary(t_len: int, switches: np.ndarray) -> np.ndarray:
    """Frames to the nearest switch or span edge, whichever is closer.

    A span without switches is handled by the same code path: the switch terms stay
    infinite and the edge terms take over (an early `return inf` here used to skip the
    edges, so a switch-free half abstained right up to its first and last frame —
    dormant under the shipped x_use=False, fixed in the 2026-11-07 audit).
    """
    switches = np.asarray(switches, dtype=np.int64)
    pos = np.arange(t_len)  # frame indices within the span
    left_idx = np.searchsorted(switches, pos, side="right") - 1  # last switch ≤ each frame
    right_idx = np.searchsorted(switches, pos, side="left")  # first switch ≥ each frame
    dist = np.full(t_len, np.inf)  # no boundary found yet
    has_left = left_idx >= 0  # a switch before this frame
    has_right = right_idx < len(switches)  # a switch after this frame
    dist[has_left] = np.minimum(dist[has_left], pos[has_left] - switches[left_idx[has_left]])  # back to switch
    dist[has_right] = np.minimum(dist[has_right], switches[right_idx[has_right]] - pos[has_right])  # forward to switch
    # distance to the span edges
    dist = np.minimum(dist, pos)  # frames from span start
    dist = np.minimum(dist, t_len - 1 - pos)  # frames from span end
    return dist


def apply_x_policy(labels: np.ndarray, conf: np.ndarray, half: np.ndarray,
                   cfg: dict = CFG) -> np.ndarray:
    """Return labels with X applied per the module rule (no-op unless cfg['x_use']).

    A frame gets X only if its confidence is low and it sits far from every switch
    and span edge.
    """
    if not cfg.get("x_use", False):
        return labels.copy()  # flag off: unchanged copy

    out = labels.copy()  # work on a copy
    stable_frames = int(round(cfg["x_stable_s"] * cfg["fps"]))  # stability seconds → frames
    for h in (1, 2):  # each half independently
        idx = np.flatnonzero(half == h)  # frames of this half
        if not len(idx):
            continue  # half absent from data
        lab = out[idx]  # labels within the half
        switches = np.flatnonzero(lab[1:] != lab[:-1]) + 1  # positions where label changes
        dist = _distance_to_nearest_boundary(len(lab), switches)  # frames to nearest boundary
        candidate = (conf[idx] <= cfg["x_conf_max"]) & (dist >= stable_frames)  # low conf + far away
        if candidate.all():
            continue  # never abstain on an entire half
        out[idx[candidate]] = CODE_X  # mark candidates as abstain
    return out
