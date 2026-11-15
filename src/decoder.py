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

Note: where a default below disagrees with the shipped config, config.py is the truth
(P8/P9 adoptions are enabled there).
"""

from __future__ import annotations  # postponed annotation evaluation

from dataclasses import dataclass  # frozen result container

import numpy as np  # score arrays and label masks

from config import CFG  # default config dict
from src.ball_fill import SOURCE_ATD  # source tag marking raw ball frames
from src.emission import CLASS_D, Emission  # emission types; CLASS_D unused here

CODE_H, CODE_A, CODE_D = 0, 1, 2  # integer codes for the three labels
CODE_X = -1  # code for frames outside graded spans
CODE_NAMES = {CODE_H: "H", CODE_A: "A", CODE_D: "D", CODE_X: "X"}  # code -> display name


def transition_log(cfg: dict = CFG) -> np.ndarray:
    """(3,3) additive log-bonus matrix for the score-maximising Viterbi."""
    sw, dw = -float(cfg["switch_pen"]), -float(cfg["dead_pen"])  # penalties as negative bonuses
    return np.array([[0.0, sw, dw],       # H->H free; H->A costs sw; H->D costs dw
                     [sw, 0.0, dw],       # A->A free; A->H costs sw; A->D costs dw
                     [dw, dw, 0.0]], dtype=np.float64)  # D->D free; D->team costs dw


def viterbi(log_e: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Best state path for one contiguous span. log_e: (T,3), trans: (3,3) bonuses."""
    t_len, n_states = log_e.shape         # span length and label count
    score = log_e[0].copy()                # best score ending in each state at t=0
    ptr = np.zeros((t_len, n_states), dtype=np.int8)  # backpointer: best previous state
    states = np.arange(n_states)           # index of each current state
    for t in range(1, t_len):              # forward pass: score every state
        cand = score[:, None] + trans          # (prev, new)
        best_prev = np.argmax(cand, axis=0)    # (new,)
        ptr[t] = best_prev                 # remember the winning previous state
        score = cand[best_prev, states] + log_e[t]  # new best scores
    path = np.empty(t_len, dtype=np.int8)  # one label per frame
    path[-1] = int(np.argmax(score))       # start from the best final state
    for t in range(t_len - 1, 0, -1):      # walk backpointers backwards
        path[t - 1] = ptr[t, path[t]]     # fill previous frame
    return path                            # finished label sequence


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """[(start, stop, label)] constant-label runs."""
    if not len(labels):
        return []                         # nothing to segment
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1  # indices where the label changes
    bounds = np.concatenate(([0], breaks, [len(labels)]))   # run edges
    return [(int(bounds[i]), int(bounds[i + 1]), int(labels[bounds[i]]))
            for i in range(len(bounds) - 1)]  # one tuple per run


def _supported(boundary: int, label: int, evidence: np.ndarray, log_e: np.ndarray,
               window: int) -> bool:
    """True if an evidence frame within +/- window of the boundary favours `label`.

    `evidence` is a boolean mask of the frames allowed to speak (raw ATD frames; with
    `gate_kin_support` also ball-less frames carrying a kinematic term for this label)."""
    lo = max(0, boundary - window)         # clamp window to the span
    hi = min(len(evidence), boundary + window)  # clamp upper edge too
    ev = evidence[lo:hi]                   # evidence frames inside the window
    if not ev.any():
        return False                       # no evidence nearby: unsupported
    winners = np.argmax(log_e[lo:hi][ev], axis=1)  # each frame's own favourite label
    return bool((winners == label).any())  # any of them favouring the label?


def _first_support(start: int, stop: int, label: int, evidence: np.ndarray,
                   log_e: np.ndarray) -> int | None:
    """First evidence frame in [start, stop) whose own emission favours `label`, or None."""
    ev = evidence[start:stop]              # evidence frames in this segment
    if not ev.any():
        return None                        # no evidence at all in the segment
    hits = np.flatnonzero(ev & (np.argmax(log_e[start:stop], axis=1) == label))  # winning frames
    return start + int(hits[0]) if len(hits) else None  # first hit's index, else None


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
        return labels.copy(), 0            # gate ablated: labels unchanged
    window = int(round(cfg["switch_support_s"] * cfg["fps"]))  # support window in frames
    move = cfg.get("gate_mode", "revert") == "move"  # P8 G: move boundary, don't revert
    atd = source == SOURCE_ATD             # raw ball frames
    ev_team = ev_dead = atd                # default evidence: raw ATD only
    if cfg.get("gate_kin_support", False):  # P9 K/T: kinematic frames may also count
        if kin_team is not None:
            ev_team = atd | kin_team       # ball + kinematic team evidence
        if kin_dead is not None:
            ev_dead = atd | kin_dead       # ball + kinematic dead evidence
    out: list[tuple[int, int, int]] = []   # rebuilt segments
    reverted = 0                           # count of merged-away segments
    for start, stop, lab in _segments(labels):  # walk segments in order
        if out and out[-1][2] == lab:
            out[-1] = (out[-1][0], stop, lab)          # no boundary: extend
        elif out and not _supported(start, lab, ev_dead if lab == CODE_D else ev_team,
                                    log_e, window):    # boundary lacks support
            first = _first_support(start, stop, lab, ev_dead if lab == CODE_D else ev_team,
                                   log_e) if move else None  # P8: first supporting frame
            if first is None:
                out[-1] = (out[-1][0], stop, out[-1][2])   # revert: swallow into previous
                reverted += 1             # count this merge
            else:
                out[-1] = (out[-1][0], first, out[-1][2])  # previous label holds until support
                out.append((first, stop, lab))  # start new segment
        else:
            out.append((start, stop, lab))  # supported boundary: keep it
    return _rebuild(labels, out), reverted  # labels plus count


def place_gap_switches(labels: np.ndarray, source: np.ndarray, frac: float) -> np.ndarray:
    """P8 G: put every H<->A boundary that touches a ball gap at `frac` of that gap.

    Inside a maximal run of non-ATD frames the emission says nothing about the team, so the
    Viterbi boundary there is a tie-break. `frac = 0.5` places it in the middle of the run
    (1.0 = the first frame after it); the result is clamped so that neither segment
    vanishes. `frac <= 0` is a no-op. Pure.
    """
    if frac <= 0:
        return labels                      # feature disabled: no-op
    segs = _segments(labels)               # current runs
    atd = source == SOURCE_ATD             # ball-present mask
    n = len(labels)                        # frame count, for clamping
    for i in range(1, len(segs)):          # examine each boundary
        start, stop, lab = segs[i]         # segment after the boundary
        prev_start, _, prev_lab = segs[i - 1]  # segment before it
        if lab not in (CODE_H, CODE_A) or prev_lab not in (CODE_H, CODE_A):
            continue                       # only H<->A boundaries move
        if atd[start] and atd[start - 1]:
            continue                       # both sides have ball: no gap touched
        anchor = start if not atd[start] else start - 1     # a frame of the touching gap
        g0 = anchor                        # walk left to the gap start
        while g0 > 0 and not atd[g0 - 1]:  # still inside gap
            g0 -= 1                        # step one frame left
        g1 = anchor + 1                    # walk right to the gap end
        while g1 < n and not atd[g1]:      # still inside gap
            g1 += 1                        # step one frame right
        new = g0 + int(round(frac * (g1 - g0)))  # boundary at frac of the gap
        new = min(max(new, prev_start + 1), stop - 1)  # keep both segments non-empty
        segs[i - 1] = (prev_start, new, prev_lab)  # re-split at the new point
        segs[i] = (new, stop, lab)         # right side of split
    return _rebuild(labels, segs)          # labels with moved boundary


def enforce_min_dwell(labels: np.ndarray, min_len: int, coalesce: bool = False) -> np.ndarray:
    """Merge segments shorter than min_len into a neighbour, shortest first, until stable.

    `coalesce` (P9 C, OFF by default): after every merge adjacent segments with the same
    label are joined before lengths are measured again. Without it a merge can leave two
    equal neighbours standing as separate segments, and the shorter of them may then be
    merged into the *other* label although together they are long enough."""
    segs = _segments(labels)               # split into runs
    while len(segs) > 1:                   # repeat until no short segment remains
        lengths = [stop - start for start, stop, _ in segs]  # current run lengths
        short = [i for i, ln in enumerate(lengths) if ln < min_len]  # below the minimum
        if not short:
            break                          # all segments long enough
        i = min(short, key=lambda j: lengths[j])  # merge the shortest first
        if i == 0:                         # edge segment: only one neighbour
            j = 1                          # right neighbour
        elif i == len(segs) - 1:
            j = i - 1                      # left neighbour
        else:  # merge into the longer neighbour; tie -> left
            j = i - 1 if lengths[i - 1] >= lengths[i + 1] else i + 1  # longer neighbour wins
        a, b = (i, j) if i < j else (j, i)  # order left-to-right
        start, stop = segs[a][0], segs[b][1]  # span covering both segments
        lab = segs[j][2]                        # the neighbour we merge INTO
        segs[a] = (start, stop, lab)        # neighbour's label wins
        del segs[b]                        # drop merged segment
        if coalesce:                        # P9 C: re-join equal neighbours
            joined: list[tuple[int, int, int]] = []  # rebuilt run list
            for seg in segs:               # scan runs
                if joined and joined[-1][2] == seg[2]:  # same label as previous
                    joined[-1] = (joined[-1][0], seg[1], seg[2])  # extend previous run
                else:
                    joined.append(seg)     # keep separate run
            segs = joined                  # use coalesced runs
    return _rebuild(labels, segs)          # labels after merging


def _rebuild(template: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = template.copy()                  # start from a copy (X-filled)
    for start, stop, lab in segs:
        out[start:stop] = lab              # paint each segment's label
    return out                             # painted copy


def enforce_d_min_dwell(labels: np.ndarray, d_min_len: int) -> np.ndarray:
    """P8 M: merge D segments shorter than `d_min_len` into the longer neighbour.

    Same rule and tie-break as `enforce_min_dwell` (longer neighbour, tie -> left, edges
    into their only neighbour), shortest first until stable; adjacent equal labels are
    coalesced after every merge, so H D H becomes one H segment. Pure.
    """
    segs = _segments(labels)               # split into runs
    while len(segs) > 1:                   # repeat until no short D remains
        short = [i for i, (start, stop, lab) in enumerate(segs)
                 if lab == CODE_D and stop - start < d_min_len]  # short D segments only
        if not short:
            break                          # no D segment below the minimum
        i = min(short, key=lambda j: segs[j][1] - segs[j][0])  # shortest D first
        if i == 0:                         # edge segment: only one neighbour
            j = 1                          # right neighbour
        elif i == len(segs) - 1:
            j = i - 1                      # left neighbour
        else:
            left = segs[i - 1][1] - segs[i - 1][0]  # neighbour lengths
            right = segs[i + 1][1] - segs[i + 1][0]  # right neighbour length
            j = i - 1 if left >= right else i + 1  # longer neighbour; tie -> left
        a, b = (i, j) if i < j else (j, i)  # order left-to-right
        segs[a] = (segs[a][0], segs[b][1], segs[j][2])  # neighbour's label wins
        del segs[b]                        # drop merged segment
        merged: list[tuple[int, int, int]] = []  # rebuilt run list
        for seg in segs:                                   # coalesce equal neighbours
            if merged and merged[-1][2] == seg[2]:  # same label as previous
                merged[-1] = (merged[-1][0], seg[1], seg[2])  # extend previous run
            else:
                merged.append(seg)        # keep separate run
        segs = merged                      # use coalesced runs
    return _rebuild(labels, segs)          # labels after merging


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
    sloped = slope_s > 0 and ball_speed is not None  # P9 S active?
    if delay <= 0 and not sloped:
        return labels                      # nothing to delay: no-op
    out = labels.copy()                    # work on a copy (pure)
    segs = _segments(labels)               # split into runs
    atd = source == SOURCE_ATD             # raw ball frames
    for i in range(1, len(segs)):          # examine each boundary
        start, stop, lab = segs[i]         # segment after the boundary
        prev_lab = segs[i - 1][2]          # label it would inherit
        if lab not in (CODE_H, CODE_A) or prev_lab not in (CODE_H, CODE_A):
            continue                       # only H<->A boundaries move
        if not (atd[start] and atd[start - 1]):
            continue                       # boundary must be ball-placed
        frames = int(delay)                # constant delay in frames
        if sloped:                         # P9 S: scale delay with ball speed
            window = ball_speed[max(start - 5, 0):start + 1]  # 6 frames ending here
            finite = np.isfinite(window)   # ignore missing speeds
            if finite.any():
                seconds = delay / fps + slope_s * (float(window[finite].mean()) - ref_ms)  # scaled
                frames = int(round(min(max(seconds, 0.0), max_delay_s) * fps))  # clamp to frames
        shift = min(frames, (stop - start) // 2)  # cap at half the following segment
        out[start:start + shift] = prev_lab  # previous team holds through the shift
    return out                             # delayed labels


def apply_no_play(labels: np.ndarray, activity: np.ndarray, cfg: dict = CFG) -> np.ndarray:
    """E10 generic no-play rule (pre-declared: dossier §7 P6, 2026-11-09).

    Mark `CODE_D` over every maximal run of >= `no_play_s` seconds in which the per-frame
    mean player speed stays below `d_activity_thr_ms`; the whole qualifying run is marked,
    including its first `no_play_s`. Non-finite activity breaks a run. No frame numbers,
    match identifiers or ball-position conditions are used; runs never span decoded halves
    (decode() calls this per half span).
    """
    if not cfg["no_play_rule"]:
        return labels                      # rule disabled: no-op
    need = max(1, int(round(cfg["no_play_s"] * cfg["fps"])))  # minimum run length in frames
    low = np.isfinite(activity) & (activity < cfg["d_activity_thr_ms"])  # inactive frames
    out = labels.copy()                    # work on a copy (pure)
    i, n = 0, len(low)                     # scan cursor
    while i < n:                           # walk the low-activity runs
        if not low[i]:
            i += 1                         # skip active frames
            continue                       # end this iteration
        j = i                              # extend to the end of this run
        while j < n and low[j]:            # extend over low frames
            j += 1                         # step to next frame
        if j - i >= need:
            out[i:j] = CODE_D              # long enough: label the whole run D
        i = j                              # jump past the run
    return out                             # labels with D runs


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
        return labels                      # rule disabled: no-op
    starts = np.asarray(kickoff_starts, dtype=np.int64)  # kick-off tag start frames
    out = labels.copy()                    # work on a copy (pure)
    for h in (1, 2):                       # handle each half separately
        idx = np.flatnonzero(half == h)    # frames graded in this half
        if not len(idx):
            continue                       # half absent from the grading
        first, last = int(idx[0]), int(idx[-1])  # half span edges
        inside = starts[(starts >= first) & (starts <= last)]  # tags inside the half
        if len(inside):
            out[first:int(inside.min())] = CODE_D  # pre-kick-off frames are D
    return out                             # labels with pre-kick-off D


@dataclass(frozen=True)
class Decoded:
    labels: np.ndarray        # (N,) int8: CODE_H/A/D on graded frames, CODE_X elsewhere
    reverted_switches: int    # switches the gating merged away (span total)


def viterbi_paths(emission: Emission, half: np.ndarray, cfg: dict = CFG) -> dict:
    """{half: (frame indices, Viterbi path)} — the part of decoding the P8 keys do not touch."""
    trans = transition_log(cfg)            # switch/death penalty matrix
    paths = {}                             # half -> (indices, path)
    for h in (1, 2):                       # decode each half on its own
        idx = np.flatnonzero(half == h)    # graded frames of this half
        if not len(idx):
            continue                       # half absent from the grading
        if not (np.diff(idx) == 1).all():  # frames must be contiguous
            raise ValueError(f"half {h} is not a contiguous span")  # refuse gappy half
        paths[h] = (idx, viterbi(emission.log_e[idx], trans))  # Viterbi over this half
    return paths                           # both halves mapped


def finish_paths(paths: dict, emission: Emission, source: np.ndarray,
                 cfg: dict = CFG) -> Decoded:
    """Gate, min-dwell and the post-passes on per-half Viterbi paths (see module docstring)."""
    n = len(emission.conf)                 # total frames
    labels = np.full(n, CODE_X, dtype=np.int8)  # X everywhere until a half is written in
    fps = cfg["fps"]                       # frames per second
    min_len = max(1, int(round(cfg["min_dwell_s"] * fps)))  # min-dwell in frames
    d_min_len = int(round(float(cfg.get("d_min_dwell_s", 0.0)) * fps))  # P8 M: D minimum
    delay = int(round(float(cfg.get("switch_delay_s", 0.0)) * fps))  # P8 S: delay in frames
    gap_frac = float(cfg.get("gap_switch_frac", 0.0))  # P8 G: gap placement fraction
    coalesce = bool(cfg.get("dwell_coalesce", False))  # P9 C: join equal neighbours
    slope = float(cfg.get("switch_delay_slope", 0.0))  # P9 S: speed-scaled delay
    kin_gate = bool(cfg.get("gate_kin_support", False))  # P9 K/T: kinematic evidence
    reverted_total = 0                     # gate merges across both halves

    for idx, viterbi_path in paths.values():  # process each half
        # kinematics only when P9 K/T on
        kin_team = emission.kin_team[idx] if kin_gate and emission.kin_team is not None else None
        # dead kinematics for the gate (P9); None when the tier is off
        kin_dead = emission.kin_dead[idx] if kin_gate and emission.kin_dead is not None else None
        path, reverted = apply_gating(viterbi_path, source[idx], emission.log_e[idx], cfg,
                                      kin_team=kin_team, kin_dead=kin_dead)  # switch gate
        reverted_total += reverted        # accumulate across halves
        if gap_frac > 0:
            path = place_gap_switches(path, source[idx], gap_frac)  # P8 G gap placement
        path = enforce_min_dwell(path, min_len, coalesce)  # min-dwell pass
        if d_min_len > min_len:            # only if P8 M set longer D min
            path = enforce_d_min_dwell(path, d_min_len)  # D-only dwell pass
        if cfg["no_play_rule"]:
            path = apply_no_play(path, emission.activity[idx], cfg)  # E10 no-play pass
        if delay > 0 or slope > 0:         # P8 S / P9 S active
            speed = emission.ball_speed[idx] if slope > 0 and emission.ball_speed is not None \
                else None                  # speed only needed for P9 S
            path = delay_team_switches(path, source[idx], delay, ball_speed=speed,
                                       slope_s=slope,
                                       ref_ms=float(cfg.get("switch_delay_ref_ms", 0.0)),
                                       max_delay_s=float(cfg.get("switch_delay_max_s", 0.0)),
                                       fps=int(fps))  # switch-delay pass
        labels[idx] = path                 # write the half back in place

    return Decoded(labels=labels, reverted_switches=reverted_total)  # final labels and count


def decode(emission: Emission, source: np.ndarray, half: np.ndarray,
           cfg: dict = CFG) -> Decoded:
    """Decode each graded half independently; everything outside is CODE_X."""
    return finish_paths(viterbi_paths(emission, half, cfg), emission, source, cfg)  # full pipeline


def segment_stats(labels: np.ndarray, fps: int = 25) -> dict:
    """Coach-readable counts over the non-X part: segments, lengths, D share."""
    graded = labels != CODE_X              # ignore free frames
    segs = _segments(labels[graded])       # runs over graded frames only
    lengths = np.array([stop - start for start, stop, _ in segs], dtype=float)  # run lengths
    d_share = float((labels[graded] == CODE_D).mean()) if graded.any() else 0.0  # D fraction
    return {
        "n_segments": len(segs),           # number of runs
        "median_len_s": float(np.median(lengths) / fps) if len(lengths) else 0.0,  # typical length
        "subsecond_share": float((lengths < fps).mean()) if len(lengths) else 0.0,  # sub-1s runs
        "d_share": d_share,                # dead-ball share
        "n_possessions": int(sum(1 for _, _, lab in segs if lab in (CODE_H, CODE_A))),  # H/A runs
    }
