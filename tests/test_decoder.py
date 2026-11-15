"""P3 acceptance: differential tests against brute force and invariants.

  * zero penalties  -> per-frame argmax
  * huge penalties  -> one constant segment
  * T <= 8          -> identical score to exhaustive enumeration over 3^T paths
  * flat gap        -> state carried through, switch at the later confident block
  * gating          -> unsupported switch reverted, ATD-supported switch kept
  * min-dwell       -> short segments merged into the longer neighbour
  * integration     -> full match decodes, only H/A/D inside halves, X outside
"""

import itertools

import numpy as np
import pytest

from config import CFG
from src.ball_fill import SOURCE_ATD, SOURCE_KINEMATIC, SOURCE_NONE
from src.decoder import (
    CODE_A,
    CODE_D,
    CODE_H,
    CODE_X,
    apply_gating,
    decode,
    enforce_min_dwell,
    segment_stats,
    transition_log,
    viterbi,
)
from src.emission import Emission


def _emission_from_labels(codes, spread=6.0):
    """Fake (T,3) log-emissions that strongly prefer the given codes."""
    log_e = np.full((len(codes), 3), -spread)     # everything starts low
    log_e[np.arange(len(codes)), codes] = 0.0     # chosen label scores top
    return log_e                                  # fake emission block


def _brute_force_score(log_e, trans):
    """Best score over every possible 3^T path."""
    best = -np.inf                                # no path scored yet
    for path in itertools.product(range(3), repeat=len(log_e)): # try every possible path
        score = log_e[0, path[0]]                 # seed with first frame
        for t in range(1, len(log_e)):            # walk the frames forward
            score += trans[path[t - 1], path[t]] + log_e[t, path[t]] # transition plus emission
        best = max(best, score)                   # keep the best so far
    return best                                   # best exhaustive score


def _path_score(log_e, trans, path):
    """Score of one fixed label path."""
    score = log_e[0, path[0]]                     # seed with first frame
    for t in range(1, len(log_e)):                # walk the frames forward
        score += trans[path[t - 1], path[t]] + log_e[t, path[t]] # transition plus emission
    return score                                  # score of this path


def test_zero_penalties_equals_argmax():
    """No switch cost: each frame keeps its own best label."""
    rng = np.random.default_rng(7)                # fixed seed
    log_e = rng.normal(size=(60, 3)) * 3.0        # random 60-frame emissions
    trans = np.zeros((3, 3))                      # zero switch cost
    path = viterbi(log_e, trans)                  # decode the sequence
    np.testing.assert_array_equal(path, np.argmax(log_e, axis=1)) # matches per-frame argmax


def test_huge_penalties_give_one_segment():
    """Huge switch cost: the whole match becomes one segment."""
    rng = np.random.default_rng(3)                # fixed seed
    log_e = rng.normal(size=(200, 3)) * 4.0       # random 200-frame emissions
    trans = transition_log({"switch_pen": 1e9, "dead_pen": 1e9}) # prohibitive switch penalties
    path = viterbi(log_e, trans)                  # decode the sequence
    assert len(np.unique(path)) == 1              # one segment only
    # the constant label with the best total evidence
    assert path[0] == np.argmax(log_e.sum(axis=0))


def test_matches_brute_force_small_T():
    """Decoder score matches exhaustive search on short sequences."""
    rng = np.random.default_rng(11)               # fixed seed
    cfg = CFG                                     # use shipped config
    trans = transition_log(cfg)                   # shipped transition costs
    for t_len in (4, 6, 8):                       # short lengths only
        log_e = rng.normal(size=(t_len, 3)) * 2.5 # random emissions
        path = viterbi(log_e, trans)              # decode the sequence
        assert _path_score(log_e, trans, path) == pytest.approx( # score matches exhaustive best
            _brute_force_score(log_e, trans), abs=1e-9
        ), f"T={t_len}"


def test_flat_gap_carries_state():
    """Confident H | flat | confident A: holds H through the flat, switches at A."""
    h_codes = [CODE_H] * 10                       # confident H block
    flat_codes = [CODE_H] * 15   # labels only build the mask; emissions are flat there
    a_codes = [CODE_A] * 10                       # confident A block
    log_e = np.zeros((35, 3))                     # start flat everywhere
    log_e[:10] = _emission_from_labels(h_codes)   # H evidence at start
    log_e[10:25] = 0.0           # totally uninformative block
    log_e[25:] = _emission_from_labels(a_codes)   # A evidence at end
    trans = transition_log(CFG)                   # shipped transition costs
    path = viterbi(log_e, trans)                  # decode the sequence
    assert (path[:10] == CODE_H).all()            # opening stays H
    assert (path[10:25] == CODE_H).all(), "must not flip inside the flat block" # flat block holds H
    assert (path[25:] == CODE_A).all(), "switch belongs at the confident block" # switch happens at A
    assert len(np.unique(path)) == 2, "exactly one switch overall" # exactly one switch


def test_gating_reverts_unsupported_switch():
    """A switch surrounded only by non-ATD frames gets merged back."""
    labels = np.array([CODE_H] * 10 + [CODE_A] * 10, dtype=np.int8) # H then A labels
    source = np.full(20, SOURCE_KINEMATIC, dtype=np.int8) # no ATD frames here
    log_e = _emission_from_labels([CODE_H] * 10 + [CODE_A] * 10) # emissions follow labels
    out, reverted = apply_gating(labels, source, log_e, CFG) # run the switch gate
    assert reverted == 1                          # one switch reverted
    assert (out == CODE_H).all()                  # everything merged to H


def test_gating_keeps_atd_supported_switch():
    """Switch gate keeps a boundary backed by ATD frames."""
    labels = np.array([CODE_H] * 10 + [CODE_A] * 10, dtype=np.int8) # H then A labels
    source = np.full(20, SOURCE_ATD, dtype=np.int8)   # ATD frames favouring A at the boundary
    log_e = _emission_from_labels([CODE_H] * 10 + [CODE_A] * 10) # emissions follow labels
    out, reverted = apply_gating(labels, source, log_e, CFG) # run the switch gate
    assert reverted == 0                          # switch kept
    np.testing.assert_array_equal(out, labels)    # labels unchanged


def test_gating_support_looks_at_emission_not_just_presence():
    """ATD frames present but all favouring the OLD label => switch still reverted."""
    labels = np.array([CODE_H] * 10 + [CODE_A] * 10, dtype=np.int8) # H then A labels
    source = np.full(20, SOURCE_ATD, dtype=np.int8) # ATD frames everywhere
    log_e = _emission_from_labels([CODE_H] * 20)      # emission says H everywhere
    out, reverted = apply_gating(labels, source, log_e, CFG) # run the switch gate
    assert reverted == 1                          # unsupported switch reverted
    assert (out == CODE_H).all()                  # everything merged to H


def test_gating_disabled_passthrough():
    """Gate off: labels pass through untouched."""
    labels = np.array([CODE_H] * 10 + [CODE_A] * 10, dtype=np.int8) # H then A labels
    source = np.full(20, SOURCE_NONE, dtype=np.int8) # no source frames
    log_e = _emission_from_labels([CODE_H] * 10 + [CODE_A] * 10) # emissions follow labels
    out, reverted = apply_gating(labels, source, log_e, {**CFG, "gate_switches": False}) # gate turned off
    assert reverted == 0                          # nothing reverted
    np.testing.assert_array_equal(out, labels)    # labels unchanged


def test_gating_support_window_is_half_open():
    """Support exactly `window` frames after the boundary is outside [lo, hi)."""
    window = int(round(CFG["switch_support_s"] * CFG["fps"])) # support window in frames
    t_len = 2 * window + 10                       # room around boundary
    boundary = window                             # boundary at window mark
    labels = np.array([CODE_H] * boundary + [CODE_A] * (t_len - boundary), dtype=np.int8) # H then A at boundary
    log_e = _emission_from_labels([CODE_H] * t_len)       # nobody favours A ...
    source = np.full(t_len, SOURCE_NONE, dtype=np.int8) # no source frames

    inside = log_e.copy(); inside[boundary + window - 1] = _emission_from_labels([CODE_A])[0] # A support last frame inside
    src_in = source.copy(); src_in[boundary + window - 1] = SOURCE_ATD # ATD on that frame
    out, reverted = apply_gating(labels, src_in, inside, CFG) # run gate, default mode
    assert reverted == 0 and (out == labels).all(), "support on the last frame inside the window" # inside window counts

    outside = log_e.copy(); outside[boundary + window] = _emission_from_labels([CODE_A])[0] # A support one frame later
    src_out = source.copy(); src_out[boundary + window] = SOURCE_ATD # ATD on that frame
    out, reverted = apply_gating(labels, src_out, outside, {**CFG, "gate_mode": "revert"}) # run gate in revert mode
    assert reverted == 1 and (out == CODE_H).all(), "one frame further is outside the window" # outside window rejected
    # P8 G (shipped): the boundary is not supported where Viterbi put it, so it moves to the
    # supporting frame instead of the segment being merged away.
    out, reverted = apply_gating(labels, src_out, outside, {**CFG, "gate_mode": "move"})
    assert reverted == 0                          # move keeps the switch
    assert (out[:boundary + window] == CODE_H).all() and (out[boundary + window:] == CODE_A).all() # boundary sits at support


def test_gating_first_segment_is_never_reverted():
    """The opening segment is never reverted away."""
    labels = np.array([CODE_A] * 5 + [CODE_H] * 30, dtype=np.int8) # tiny opening A run
    source = np.full(35, SOURCE_NONE, dtype=np.int8) # no source frames
    log_e = _emission_from_labels([CODE_H] * 35)  # emissions favour H
    out, reverted = apply_gating(labels, source, log_e, CFG) # run the switch gate
    assert reverted == 1                          # unsupported H switch reverted
    assert (out == CODE_A).all(), "the unsupported H switch folds into the opening A segment" # opening A survives


def test_min_dwell_all_short_collapses_to_one_segment():
    """Min-dwell: every spell too short merges into one."""
    labels = np.array([CODE_H] * 3 + [CODE_A] * 4 + [CODE_D] * 2 + [CODE_H] * 3, dtype=np.int8) # four short spells
    out = enforce_min_dwell(labels, min_len=25)   # minimum 25 frames
    assert len(np.unique(out)) == 1               # collapsed to one


def test_min_dwell_one_is_a_noop():
    """Min-dwell of 1 frame changes nothing."""
    labels = np.array([CODE_H, CODE_A, CODE_D, CODE_H], dtype=np.int8) # one-frame spells
    np.testing.assert_array_equal(enforce_min_dwell(labels, min_len=1), labels) # minimum one: no change


def test_decode_rejects_non_contiguous_half():
    """A half split by a hole is rejected."""
    t_len = 40                                    # 40-frame sequence
    log_e = _emission_from_labels([CODE_H] * t_len) # all-H emissions
    em = Emission(log_e=log_e, conf=np.ones(t_len), p_d=np.full(t_len, 0.2), ev=np.ones((t_len, 2)), # wrap in Emission
                  activity=np.full(t_len, 5.0))
    half = np.zeros(t_len, dtype=np.int8)         # no half yet
    half[:10] = 1                                 # half one starts
    half[20:30] = 1                                       # half 1 split by a hole
    with pytest.raises(ValueError, match="not a contiguous span"): # span must be contiguous
        decode(em, np.full(t_len, SOURCE_ATD, dtype=np.int8), half, CFG) # decode call must raise


def test_min_dwell_merges_short_into_longer_neighbour():
    """Min-dwell merges a short spell into its longer neighbour."""
    # H(60) A(10) D(30) -> A (10 < 25) merges into the longer neighbour H(60)
    labels = np.array([CODE_H] * 60 + [CODE_A] * 10 + [CODE_D] * 30, dtype=np.int8)
    out = enforce_min_dwell(labels, min_len=25)   # minimum 25 frames
    assert CODE_A not in np.unique(out)           # short A gone
    assert (out[:70] == CODE_H).all()             # first part merged to H
    assert (out[70:] == CODE_D).all()             # D tail untouched


def test_min_dwell_edge_segment_merges_inward():
    """A short spell at the edge merges inward."""
    labels = np.array([CODE_A] * 5 + [CODE_H] * 80, dtype=np.int8) # short edge A run
    out = enforce_min_dwell(labels, min_len=25)   # minimum 25 frames
    assert (out == CODE_H).all()                  # A folded into H


def test_min_dwell_noop_when_all_long():
    """Min-dwell leaves already-long spells alone."""
    labels = np.array([CODE_H] * 40 + [CODE_A] * 40, dtype=np.int8) # two long spells
    out = enforce_min_dwell(labels, min_len=25)   # minimum 25 frames
    np.testing.assert_array_equal(out, labels)    # nothing to merge


def test_segment_stats_counts():
    """Stats count segments, possessions and shares."""
    labels = np.array([CODE_H] * 50 + [CODE_A] * 12 + [CODE_D] * 40, dtype=np.int8) # three spells: 50/12/40
    stats = segment_stats(labels, fps=25)         # stats at 25 fps
    assert stats["n_segments"] == 3               # three segments counted
    assert stats["n_possessions"] == 2            # two possessions counted
    assert stats["median_len_s"] == pytest.approx(40 / 25) # middle spell length
    # one of three segments is sub-second
    assert stats["subsecond_share"] == pytest.approx(1 / 3)
    # D frames over all frames
    assert stats["d_share"] == pytest.approx(40 / 102)


def test_full_match_decode(arrays):
    """Integration: halves decoded, X outside, valid labels, stats sane."""
    from src.ball_fill import build_ball_track
    from src.emission import build_emission

    track = build_ball_track(arrays.ball, arrays.half) # fill ball track
    em = build_emission(track, arrays.players, arrays.players_s, arrays.slot_team) # build frame emissions
    result = decode(em, track.source, arrays.half) # decode full match

    assert len(result.labels) == arrays.n_frames  # one label per frame
    outside = arrays.half == 0                    # ungraded frames mask
    inside = ~outside                             # graded frames mask
    assert (result.labels[outside] == CODE_X).all(), "ungraded frames must be X" # outside stays X
    assert set(np.unique(result.labels[inside])) <= {CODE_H, CODE_A, CODE_D} # only valid labels inside
    assert arrays.graded[inside].all()            # inside all graded

    stats = segment_stats(result.labels)          # stats over the decode
    assert 10 <= stats["n_segments"] <= 5000, stats # sane segment count
    assert stats["subsecond_share"] <= 0.20, stats # few sub-second segments
    assert 0.05 <= stats["d_share"] <= 0.75, stats # plausible dead share
