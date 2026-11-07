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
    log_e = np.full((len(codes), 3), -spread)
    log_e[np.arange(len(codes)), codes] = 0.0
    return log_e


def _brute_force_score(log_e, trans):
    best = -np.inf
    for path in itertools.product(range(3), repeat=len(log_e)):
        score = log_e[0, path[0]]
        for t in range(1, len(log_e)):
            score += trans[path[t - 1], path[t]] + log_e[t, path[t]]
        best = max(best, score)
    return best


def _path_score(log_e, trans, path):
    score = log_e[0, path[0]]
    for t in range(1, len(log_e)):
        score += trans[path[t - 1], path[t]] + log_e[t, path[t]]
    return score


def test_zero_penalties_equals_argmax():
    rng = np.random.default_rng(7)
    log_e = rng.normal(size=(60, 3)) * 3.0
    trans = np.zeros((3, 3))
    path = viterbi(log_e, trans)
    np.testing.assert_array_equal(path, np.argmax(log_e, axis=1))


def test_huge_penalties_give_one_segment():
    rng = np.random.default_rng(3)
    log_e = rng.normal(size=(200, 3)) * 4.0
    trans = transition_log({"switch_pen": 1e9, "dead_pen": 1e9})
    path = viterbi(log_e, trans)
    assert len(np.unique(path)) == 1
    assert path[0] == np.argmax(log_e.sum(axis=0))


def test_matches_brute_force_small_T():
    rng = np.random.default_rng(11)
    cfg = CFG
    trans = transition_log(cfg)
    for t_len in (4, 6, 8):
        log_e = rng.normal(size=(t_len, 3)) * 2.5
        path = viterbi(log_e, trans)
        assert _path_score(log_e, trans, path) == pytest.approx(
            _brute_force_score(log_e, trans), abs=1e-9
        ), f"T={t_len}"


def test_flat_gap_carries_state():
    """Confident H | flat | confident A: holds H through the flat, switches at A."""
    h_codes = [CODE_H] * 10
    flat_codes = [CODE_H] * 15   # labels only build the mask; emissions are flat there
    a_codes = [CODE_A] * 10
    log_e = np.zeros((35, 3))
    log_e[:10] = _emission_from_labels(h_codes)
    log_e[10:25] = 0.0           # totally uninformative block
    log_e[25:] = _emission_from_labels(a_codes)
    trans = transition_log(CFG)
    path = viterbi(log_e, trans)
    assert (path[:10] == CODE_H).all()
    assert (path[10:25] == CODE_H).all(), "must not flip inside the flat block"
    assert (path[25:] == CODE_A).all(), "switch belongs at the confident block"
    assert len(np.unique(path)) == 2, "exactly one switch overall"


def test_gating_reverts_unsupported_switch():
    """A switch surrounded only by non-ATD frames gets merged back."""
    labels = np.array([CODE_H] * 10 + [CODE_A] * 10, dtype=np.int8)
    source = np.full(20, SOURCE_KINEMATIC, dtype=np.int8)
    log_e = _emission_from_labels([CODE_H] * 10 + [CODE_A] * 10)
    out, reverted = apply_gating(labels, source, log_e, CFG)
    assert reverted == 1
    assert (out == CODE_H).all()


def test_gating_keeps_atd_supported_switch():
    labels = np.array([CODE_H] * 10 + [CODE_A] * 10, dtype=np.int8)
    source = np.full(20, SOURCE_ATD, dtype=np.int8)   # ATD frames favouring A at the boundary
    log_e = _emission_from_labels([CODE_H] * 10 + [CODE_A] * 10)
    out, reverted = apply_gating(labels, source, log_e, CFG)
    assert reverted == 0
    np.testing.assert_array_equal(out, labels)


def test_gating_support_looks_at_emission_not_just_presence():
    """ATD frames present but all favouring the OLD label => switch still reverted."""
    labels = np.array([CODE_H] * 10 + [CODE_A] * 10, dtype=np.int8)
    source = np.full(20, SOURCE_ATD, dtype=np.int8)
    log_e = _emission_from_labels([CODE_H] * 20)      # emission says H everywhere
    out, reverted = apply_gating(labels, source, log_e, CFG)
    assert reverted == 1
    assert (out == CODE_H).all()


def test_gating_disabled_passthrough():
    labels = np.array([CODE_H] * 10 + [CODE_A] * 10, dtype=np.int8)
    source = np.full(20, SOURCE_NONE, dtype=np.int8)
    log_e = _emission_from_labels([CODE_H] * 10 + [CODE_A] * 10)
    out, reverted = apply_gating(labels, source, log_e, {**CFG, "gate_switches": False})
    assert reverted == 0
    np.testing.assert_array_equal(out, labels)


def test_min_dwell_merges_short_into_longer_neighbour():
    # H(60) A(10) D(30) -> A (10 < 25) merges into the longer neighbour H(60)
    labels = np.array([CODE_H] * 60 + [CODE_A] * 10 + [CODE_D] * 30, dtype=np.int8)
    out = enforce_min_dwell(labels, min_len=25)
    assert CODE_A not in np.unique(out)
    assert (out[:70] == CODE_H).all()
    assert (out[70:] == CODE_D).all()


def test_min_dwell_edge_segment_merges_inward():
    labels = np.array([CODE_A] * 5 + [CODE_H] * 80, dtype=np.int8)
    out = enforce_min_dwell(labels, min_len=25)
    assert (out == CODE_H).all()


def test_min_dwell_noop_when_all_long():
    labels = np.array([CODE_H] * 40 + [CODE_A] * 40, dtype=np.int8)
    out = enforce_min_dwell(labels, min_len=25)
    np.testing.assert_array_equal(out, labels)


def test_segment_stats_counts():
    labels = np.array([CODE_H] * 50 + [CODE_A] * 12 + [CODE_D] * 40, dtype=np.int8)
    stats = segment_stats(labels, fps=25)
    assert stats["n_segments"] == 3
    assert stats["n_possessions"] == 2
    assert stats["median_len_s"] == pytest.approx(40 / 25)
    assert stats["subsecond_share"] == pytest.approx(1 / 3)
    assert stats["d_share"] == pytest.approx(40 / 102)


def test_full_match_decode(arrays):
    """Integration: halves decoded, X outside, valid labels, stats sane."""
    from src.ball_fill import build_ball_track
    from src.emission import build_emission

    track = build_ball_track(arrays.ball, arrays.half)
    em = build_emission(track, arrays.players, arrays.players_s, arrays.slot_team)
    result = decode(em, track.source, arrays.half)

    assert len(result.labels) == arrays.n_frames
    outside = arrays.half == 0
    inside = ~outside
    assert (result.labels[outside] == CODE_X).all(), "ungraded frames must be X"
    assert set(np.unique(result.labels[inside])) <= {CODE_H, CODE_A, CODE_D}
    assert arrays.graded[inside].all()

    stats = segment_stats(result.labels)
    assert 10 <= stats["n_segments"] <= 5000, stats
    assert stats["subsecond_share"] <= 0.20, stats
    assert 0.05 <= stats["d_share"] <= 0.75, stats
