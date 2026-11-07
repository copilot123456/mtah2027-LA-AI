"""Segment-coherent decoding — P3.

Per graded span (the two halves, decoded independently, never across the halftime hole):

  1. Viterbi over the emission with a log-space transition matrix:
       stay 0, H<->A -switch_pen, any<->D -dead_pen
  2. Switch gating: a boundary that has no raw-ATD frame with supporting evidence
       within +/- switch_support_s is reverted (merged into the previous segment) —
       tier-2/3 evidence never switches alone. Ablatable (config gate_switches).
  3. Min-dwell post-pass: segments shorter than min_dwell_s merge into the longer
       neighbouring segment (edges merge into their only neighbour), shortest first
       until stable.

Frames outside the graded spans are CODE_X (free for the scorer).
Pure function: inputs are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CFG
from src.ball_fill import SOURCE_ATD
from src.emission import CLASS_D, Emission

CODE_H, CODE_A, CODE_D = 0, 1, 2
CODE_X = -1
CODE_NAMES = {CODE_H: "H", CODE_A: "A", CODE_D: "D", CODE_X: "X"}


def transition_log(cfg: dict = CFG) -> np.ndarray:
    """(3,3) additive log-bonus matrix for the score-maximising Viterbi."""
    sw, dw = -float(cfg["switch_pen"]), -float(cfg["dead_pen"])
    return np.array([[0.0, sw, dw],
                     [sw, 0.0, dw],
                     [dw, dw, 0.0]], dtype=np.float64)


def viterbi(log_e: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Best state path for one contiguous span. log_e: (T,3), trans: (3,3) bonuses."""
    t_len, n_states = log_e.shape
    score = log_e[0].copy()
    ptr = np.zeros((t_len, n_states), dtype=np.int8)
    states = np.arange(n_states)
    for t in range(1, t_len):
        cand = score[:, None] + trans          # (prev, new)
        best_prev = np.argmax(cand, axis=0)    # (new,)
        ptr[t] = best_prev
        score = cand[best_prev, states] + log_e[t]
    path = np.empty(t_len, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    return path


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """[(start, stop, label)] constant-label runs."""
    if not len(labels):
        return []
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(labels[bounds[i]]))
            for i in range(len(bounds) - 1)]


def _supported(boundary: int, label: int, source: np.ndarray, log_e: np.ndarray,
               window: int) -> bool:
    """True if an ATD frame within +/- window of the boundary favours `label`."""
    lo = max(0, boundary - window)
    hi = min(len(source), boundary + window)
    atd = source[lo:hi] == SOURCE_ATD
    if not atd.any():
        return False
    winners = np.argmax(log_e[lo:hi][atd], axis=1)
    return bool((winners == label).any())


def apply_gating(labels: np.ndarray, source: np.ndarray, log_e: np.ndarray,
                 cfg: dict = CFG) -> tuple[np.ndarray, int]:
    """Merge segments whose start boundary lacks nearby ATD support. Returns (labels, n_reverted)."""
    if not cfg["gate_switches"]:
        return labels.copy(), 0
    window = int(round(cfg["switch_support_s"] * cfg["fps"]))
    out: list[tuple[int, int, int]] = []
    reverted = 0
    for start, stop, lab in _segments(labels):
        if out and out[-1][2] == lab:
            out[-1] = (out[-1][0], stop, lab)          # no boundary: extend
        elif out and not _supported(start, lab, source, log_e, window):
            out[-1] = (out[-1][0], stop, out[-1][2])   # revert: swallow into previous
            reverted += 1
        else:
            out.append((start, stop, lab))
    return _rebuild(labels, out), reverted


def enforce_min_dwell(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Merge segments shorter than min_len into a neighbour, shortest first, until stable."""
    segs = _segments(labels)
    while len(segs) > 1:
        lengths = [stop - start for start, stop, _ in segs]
        short = [i for i, ln in enumerate(lengths) if ln < min_len]
        if not short:
            break
        i = min(short, key=lambda j: lengths[j])
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:  # merge into the longer neighbour; tie -> left
            j = i - 1 if lengths[i - 1] >= lengths[i + 1] else i + 1
        a, b = (i, j) if i < j else (j, i)
        start, stop = segs[a][0], segs[b][1]
        lab = segs[j][2]                        # the neighbour we merge INTO
        segs[a] = (start, stop, lab)
        del segs[b]
    return _rebuild(labels, segs)


def _rebuild(template: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = template.copy()
    for start, stop, lab in segs:
        out[start:stop] = lab
    return out


@dataclass(frozen=True)
class Decoded:
    labels: np.ndarray        # (N,) int8: CODE_H/A/D on graded frames, CODE_X elsewhere
    reverted_switches: int    # switches the gating merged away (span total)


def decode(emission: Emission, source: np.ndarray, half: np.ndarray,
           cfg: dict = CFG) -> Decoded:
    """Decode each graded half independently; everything outside is CODE_X."""
    n = len(emission.conf)
    labels = np.full(n, CODE_X, dtype=np.int8)
    trans = transition_log(cfg)
    min_len = max(1, int(round(cfg["min_dwell_s"] * cfg["fps"])))
    reverted_total = 0

    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        if not (np.diff(idx) == 1).all():
            raise ValueError(f"half {h} is not a contiguous span")
        path = viterbi(emission.log_e[idx], trans)
        path, reverted = apply_gating(path, source[idx], emission.log_e[idx], cfg)
        reverted_total += reverted
        path = enforce_min_dwell(path, min_len)
        labels[idx] = path

    return Decoded(labels=labels, reverted_switches=reverted_total)


def segment_stats(labels: np.ndarray, fps: int = 25) -> dict:
    """Coach-readable counts over the non-X part: segments, lengths, D share."""
    graded = labels != CODE_X
    segs = _segments(labels[graded])
    lengths = np.array([stop - start for start, stop, _ in segs], dtype=float)
    d_share = float((labels[graded] == CODE_D).mean()) if graded.any() else 0.0
    return {
        "n_segments": len(segs),
        "median_len_s": float(np.median(lengths) / fps) if len(lengths) else 0.0,
        "subsecond_share": float((lengths < fps).mean()) if len(lengths) else 0.0,
        "d_share": d_share,
        "n_possessions": int(sum(1 for _, _, lab in segs if lab in (CODE_H, CODE_A))),
    }
