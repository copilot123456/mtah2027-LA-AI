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
  4. Generic no-play pass (cfg no_play_rule, OFF by default — E10): D over sustained
       low-activity runs, applied per half after gating and min-dwell (dossier §7).

`apply_kickoff_rule` (K1, dossier §7 P7) is a separate post-pass the pipeline step applies
to the decoded labels: D from the start of a graded half up to its kick-off tag.

P8 components (dossier §7 P8, 2026-11-14; all OFF by default, so the default path is
byte-identical to the P7 decoder):
  G  gate_mode "move": an unsupported boundary is moved to the first raw-ATD frame of the
     new segment that favours its label instead of merging the whole segment away;
     gap_switch_frac: an H<->A boundary touching a ball gap is placed inside that gap.
  M  d_min_dwell_s: a longer minimum duration for D segments only.
  S  switch_delay_s: H<->A boundaries placed by ball evidence move later.
Order per half: Viterbi -> gate -> gap placement -> min-dwell -> D minimum -> no-play ->
switch delay.

P9 components (dossier §7 P9, 2026-11-14; all OFF by default, so the default path is
byte-identical to the P8 decoder):
  C  dwell_coalesce: the min-dwell pass joins equal neighbours after every merge.
  K/T gate_kin_support: a ball-less frame that carries an active kinematic term counts as
     support in the switch gate like a raw ball frame (the terms themselves live in the
     emission).
  S  switch_delay_slope: the team-switch delay grows with the ball speed at the boundary.

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


def _supported(boundary: int, label: int, evidence: np.ndarray, log_e: np.ndarray,
               window: int) -> bool:
    """True if an evidence frame within +/- window of the boundary favours `label`.

    `evidence` is a boolean mask of the frames allowed to speak (raw ATD frames; with
    `gate_kin_support` also ball-less frames carrying a kinematic term for this label)."""
    lo = max(0, boundary - window)
    hi = min(len(evidence), boundary + window)
    ev = evidence[lo:hi]
    if not ev.any():
        return False
    winners = np.argmax(log_e[lo:hi][ev], axis=1)
    return bool((winners == label).any())


def _first_support(start: int, stop: int, label: int, evidence: np.ndarray,
                   log_e: np.ndarray) -> int | None:
    """First evidence frame in [start, stop) whose own emission favours `label`, or None."""
    ev = evidence[start:stop]
    if not ev.any():
        return None
    hits = np.flatnonzero(ev & (np.argmax(log_e[start:stop], axis=1) == label))
    return start + int(hits[0]) if len(hits) else None


def apply_gating(labels: np.ndarray, source: np.ndarray, log_e: np.ndarray,
                 cfg: dict = CFG, *, kin_team: np.ndarray | None = None,
                 kin_dead: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Gate segment boundaries on nearby ATD support. Returns (labels, n_reverted).

    `gate_mode = "revert"` (default): a segment whose start boundary lacks support is merged
    whole into the previous one. `gate_mode = "move"` (P8 G): the boundary is moved forward
    to the first raw-ATD frame of the segment that favours its label — the previous label
    holds until the ball shows the new one — and only a segment with no such frame is
    merged. `n_reverted` counts merged segments in both modes.

    `gate_kin_support` (P9, OFF by default): `kin_team` / `kin_dead` mark the ball-less
    frames whose emission carries a kinematic team / dead term; they then count as evidence
    for a team label / for `D` exactly like raw ATD frames.
    """
    if not cfg["gate_switches"]:
        return labels.copy(), 0
    window = int(round(cfg["switch_support_s"] * cfg["fps"]))
    move = cfg.get("gate_mode", "revert") == "move"
    atd = source == SOURCE_ATD
    ev_team = ev_dead = atd
    if cfg.get("gate_kin_support", False):
        if kin_team is not None:
            ev_team = atd | kin_team
        if kin_dead is not None:
            ev_dead = atd | kin_dead
    out: list[tuple[int, int, int]] = []
    reverted = 0
    for start, stop, lab in _segments(labels):
        if out and out[-1][2] == lab:
            out[-1] = (out[-1][0], stop, lab)          # no boundary: extend
        elif out and not _supported(start, lab, ev_dead if lab == CODE_D else ev_team,
                                    log_e, window):
            first = _first_support(start, stop, lab, ev_dead if lab == CODE_D else ev_team,
                                   log_e) if move else None
            if first is None:
                out[-1] = (out[-1][0], stop, out[-1][2])   # revert: swallow into previous
                reverted += 1
            else:
                out[-1] = (out[-1][0], first, out[-1][2])  # previous label holds until support
                out.append((first, stop, lab))
        else:
            out.append((start, stop, lab))
    return _rebuild(labels, out), reverted


def place_gap_switches(labels: np.ndarray, source: np.ndarray, frac: float) -> np.ndarray:
    """P8 G: put every H<->A boundary that touches a ball gap at `frac` of that gap.

    Inside a maximal run of non-ATD frames the emission says nothing about the team, so the
    Viterbi boundary there is a tie-break. `frac = 0.5` places it in the middle of the run
    (1.0 = the first frame after it); the result is clamped so that neither segment
    vanishes. `frac <= 0` is a no-op. Pure.
    """
    if frac <= 0:
        return labels
    segs = _segments(labels)
    atd = source == SOURCE_ATD
    n = len(labels)
    for i in range(1, len(segs)):
        start, stop, lab = segs[i]
        prev_start, _, prev_lab = segs[i - 1]
        if lab not in (CODE_H, CODE_A) or prev_lab not in (CODE_H, CODE_A):
            continue
        if atd[start] and atd[start - 1]:
            continue
        anchor = start if not atd[start] else start - 1     # a frame of the touching gap
        g0 = anchor
        while g0 > 0 and not atd[g0 - 1]:
            g0 -= 1
        g1 = anchor + 1
        while g1 < n and not atd[g1]:
            g1 += 1
        new = g0 + int(round(frac * (g1 - g0)))
        new = min(max(new, prev_start + 1), stop - 1)
        segs[i - 1] = (prev_start, new, prev_lab)
        segs[i] = (new, stop, lab)
    return _rebuild(labels, segs)


def enforce_min_dwell(labels: np.ndarray, min_len: int, coalesce: bool = False) -> np.ndarray:
    """Merge segments shorter than min_len into a neighbour, shortest first, until stable.

    `coalesce` (P9 C, OFF by default): after every merge adjacent segments with the same
    label are joined before lengths are measured again. Without it a merge can leave two
    equal neighbours standing as separate segments, and the shorter of them may then be
    merged into the *other* label although together they are long enough."""
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
        if coalesce:
            joined: list[tuple[int, int, int]] = []
            for seg in segs:
                if joined and joined[-1][2] == seg[2]:
                    joined[-1] = (joined[-1][0], seg[1], seg[2])
                else:
                    joined.append(seg)
            segs = joined
    return _rebuild(labels, segs)


def _rebuild(template: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = template.copy()
    for start, stop, lab in segs:
        out[start:stop] = lab
    return out


def enforce_d_min_dwell(labels: np.ndarray, d_min_len: int) -> np.ndarray:
    """P8 M: merge D segments shorter than `d_min_len` into the longer neighbour.

    Same rule and tie-break as `enforce_min_dwell` (longer neighbour, tie -> left, edges
    into their only neighbour), shortest first until stable; adjacent equal labels are
    coalesced after every merge, so H D H becomes one H segment. Pure.
    """
    segs = _segments(labels)
    while len(segs) > 1:
        short = [i for i, (start, stop, lab) in enumerate(segs)
                 if lab == CODE_D and stop - start < d_min_len]
        if not short:
            break
        i = min(short, key=lambda j: segs[j][1] - segs[j][0])
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:
            left = segs[i - 1][1] - segs[i - 1][0]
            right = segs[i + 1][1] - segs[i + 1][0]
            j = i - 1 if left >= right else i + 1
        a, b = (i, j) if i < j else (j, i)
        segs[a] = (segs[a][0], segs[b][1], segs[j][2])
        del segs[b]
        merged: list[tuple[int, int, int]] = []
        for seg in segs:                                   # coalesce equal neighbours
            if merged and merged[-1][2] == seg[2]:
                merged[-1] = (merged[-1][0], seg[1], seg[2])
            else:
                merged.append(seg)
        segs = merged
    return _rebuild(labels, segs)


def delay_team_switches(labels: np.ndarray, source: np.ndarray, delay: int, *,
                        ball_speed: np.ndarray | None = None, slope_s: float = 0.0,
                        ref_ms: float = 0.0, max_delay_s: float = 0.0,
                        fps: int = 25) -> np.ndarray:
    """P8 S: move ball-placed H<->A boundaries `delay` frames later.

    The kernel hands the ball to the receiving team when it is nearer to it — mid-flight —
    while a change of possession is the moment the receiver controls it. Only a boundary
    whose two adjacent frames are raw ATD is moved (the ball evidence placed it); the move
    is capped at half the following segment. Boundaries touching a ball gap and every
    boundary into or out of D stay. Pure.

    P9 S (`slope_s` > 0, OFF by default): a faster ball is handed over earlier in its
    flight, so the delay of each boundary becomes
    `clip(delay / fps + slope_s * (v - ref_ms), 0, max_delay_s)` seconds, `v` being the mean
    finite ball speed over the six frames ending at the boundary; without a finite speed the
    constant delay applies.
    """
    sloped = slope_s > 0 and ball_speed is not None
    if delay <= 0 and not sloped:
        return labels
    out = labels.copy()
    segs = _segments(labels)
    atd = source == SOURCE_ATD
    for i in range(1, len(segs)):
        start, stop, lab = segs[i]
        prev_lab = segs[i - 1][2]
        if lab not in (CODE_H, CODE_A) or prev_lab not in (CODE_H, CODE_A):
            continue
        if not (atd[start] and atd[start - 1]):
            continue
        frames = int(delay)
        if sloped:
            window = ball_speed[max(start - 5, 0):start + 1]
            finite = np.isfinite(window)
            if finite.any():
                seconds = delay / fps + slope_s * (float(window[finite].mean()) - ref_ms)
                frames = int(round(min(max(seconds, 0.0), max_delay_s) * fps))
        shift = min(frames, (stop - start) // 2)
        out[start:start + shift] = prev_lab
    return out


def apply_no_play(labels: np.ndarray, activity: np.ndarray, cfg: dict = CFG) -> np.ndarray:
    """E10 generic no-play rule (pre-declared: dossier §7 P6, 2026-11-09).

    Mark `CODE_D` over every maximal run of >= `no_play_s` seconds in which the per-frame
    mean player speed stays below `d_activity_thr_ms`; the whole qualifying run is marked,
    including its first `no_play_s`. Non-finite activity breaks a run. No frame numbers,
    match identifiers or ball-position conditions are used; runs never span decoded halves
    (decode() calls this per half span).
    """
    if not cfg["no_play_rule"]:
        return labels
    need = max(1, int(round(cfg["no_play_s"] * cfg["fps"])))
    low = np.isfinite(activity) & (activity < cfg["d_activity_thr_ms"])
    out = labels.copy()
    i, n = 0, len(low)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j]:
            j += 1
        if j - i >= need:
            out[i:j] = CODE_D
        i = j
    return out


def apply_kickoff_rule(labels: np.ndarray, half: np.ndarray, kickoff_starts: np.ndarray,
                       cfg: dict = CFG) -> np.ndarray:
    """K1 kick-off tag rule (pre-declared: dossier §7 P7, 2026-11-14).

    For each graded half, let `k` be the earliest `Kick Off (Start)` tag whose start frame
    lies inside that half's span. Frames from the half's first frame up to `k - 1` become
    `CODE_D`: play has not started, so the ball is dead (FAQ: the second half is graded
    before its kick-off and those frames are all `D`). No tag inside a half -> that half is
    untouched. No frame numbers or match identifiers; the only input is the automatic
    Smart Tagging. Pure: `labels` is not modified.
    """
    if not cfg["kickoff_rule"]:
        return labels
    starts = np.asarray(kickoff_starts, dtype=np.int64)
    out = labels.copy()
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        first, last = int(idx[0]), int(idx[-1])
        inside = starts[(starts >= first) & (starts <= last)]
        if len(inside):
            out[first:int(inside.min())] = CODE_D
    return out


@dataclass(frozen=True)
class Decoded:
    labels: np.ndarray        # (N,) int8: CODE_H/A/D on graded frames, CODE_X elsewhere
    reverted_switches: int    # switches the gating merged away (span total)


def viterbi_paths(emission: Emission, half: np.ndarray, cfg: dict = CFG) -> dict:
    """{half: (frame indices, Viterbi path)} — the part of decoding the P8 keys do not touch."""
    trans = transition_log(cfg)
    paths = {}
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        if not (np.diff(idx) == 1).all():
            raise ValueError(f"half {h} is not a contiguous span")
        paths[h] = (idx, viterbi(emission.log_e[idx], trans))
    return paths


def finish_paths(paths: dict, emission: Emission, source: np.ndarray,
                 cfg: dict = CFG) -> Decoded:
    """Gate, min-dwell and the post-passes on per-half Viterbi paths (see module docstring)."""
    n = len(emission.conf)
    labels = np.full(n, CODE_X, dtype=np.int8)
    fps = cfg["fps"]
    min_len = max(1, int(round(cfg["min_dwell_s"] * fps)))
    d_min_len = int(round(float(cfg.get("d_min_dwell_s", 0.0)) * fps))
    delay = int(round(float(cfg.get("switch_delay_s", 0.0)) * fps))
    gap_frac = float(cfg.get("gap_switch_frac", 0.0))
    coalesce = bool(cfg.get("dwell_coalesce", False))
    slope = float(cfg.get("switch_delay_slope", 0.0))
    kin_gate = bool(cfg.get("gate_kin_support", False))
    reverted_total = 0

    for idx, viterbi_path in paths.values():
        kin_team = emission.kin_team[idx] if kin_gate and emission.kin_team is not None else None
        kin_dead = emission.kin_dead[idx] if kin_gate and emission.kin_dead is not None else None
        path, reverted = apply_gating(viterbi_path, source[idx], emission.log_e[idx], cfg,
                                      kin_team=kin_team, kin_dead=kin_dead)
        reverted_total += reverted
        if gap_frac > 0:
            path = place_gap_switches(path, source[idx], gap_frac)
        path = enforce_min_dwell(path, min_len, coalesce)
        if d_min_len > min_len:
            path = enforce_d_min_dwell(path, d_min_len)
        if cfg["no_play_rule"]:
            path = apply_no_play(path, emission.activity[idx], cfg)
        if delay > 0 or slope > 0:
            speed = emission.ball_speed[idx] if slope > 0 and emission.ball_speed is not None \
                else None
            path = delay_team_switches(path, source[idx], delay, ball_speed=speed,
                                       slope_s=slope,
                                       ref_ms=float(cfg.get("switch_delay_ref_ms", 0.0)),
                                       max_delay_s=float(cfg.get("switch_delay_max_s", 0.0)),
                                       fps=int(fps))
        labels[idx] = path

    return Decoded(labels=labels, reverted_switches=reverted_total)


def decode(emission: Emission, source: np.ndarray, half: np.ndarray,
           cfg: dict = CFG) -> Decoded:
    """Decode each graded half independently; everything outside is CODE_X."""
    return finish_paths(viterbi_paths(emission, half, cfg), emission, source, cfg)


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
