"""E10 generic no-play rule (pre-declared: dossier §7 P6, 2026-11-09).

Unit tests on the pure post-pass + the real-window integration check. The rule's
default is OFF, so the frozen output is guarded by the byte-identity test in
tests/test_submission.py; the no-harm gate on IDSSE is recorded in
tuning/results/no_harm_no_play.csv.
"""

import numpy as np

from config import CFG
from src.decoder import CODE_A, CODE_D, CODE_H, apply_no_play

FPS = 25
T = 10 * FPS  # 250 frames = cfg no_play_s * fps


def test_run_at_threshold_becomes_d_including_its_first_seconds():
    """A run exactly at the threshold becomes D in full."""
    n = 500
    activity = np.full(n, 5.0)
    activity[100:100 + T] = 0.5                      # exactly no_play_s of low activity
    labels = np.full(n, CODE_H, dtype=np.int8)

    out = apply_no_play(labels, activity, {**CFG, "no_play_rule": True})

    assert (out[100:100 + T] == CODE_D).all(), "whole qualifying run is marked"
    assert (out[:100] == CODE_H).all() and (out[100 + T:] == CODE_H).all()
    assert (labels == CODE_H).all(), "input must not be mutated"


def test_short_low_activity_run_is_untouched():
    """One frame short of the threshold: nothing changes."""
    n = 300
    activity = np.full(n, 5.0)
    activity[50:50 + T - 1] = 0.5                    # one frame short of the threshold
    labels = np.full(n, CODE_A, dtype=np.int8)

    out = apply_no_play(labels, activity, {**CFG, "no_play_rule": True})
    np.testing.assert_array_equal(out, labels)


def test_flag_off_is_a_noop():
    """Rule off: no relabelling at all."""
    activity = np.zeros(300)                         # low everywhere would qualify
    labels = np.full(300, CODE_H, dtype=np.int8)

    out = apply_no_play(labels, activity, {**CFG, "no_play_rule": False})
    np.testing.assert_array_equal(out, labels)


def test_nonfinite_activity_breaks_runs():
    """A NaN splits the run; neither half reaches the threshold."""
    n = 600
    activity = np.full(n, 5.0)
    activity[100:100 + T - 10] = 0.5                 # 240 frames (9.6 s) low
    activity[100 + T - 10] = np.nan                  # break ...
    activity[100 + T - 9:100 + 2 * T - 19] = 0.5     # ... then another 240 frames low
    labels = np.full(n, CODE_H, dtype=np.int8)

    out = apply_no_play(labels, activity, {**CFG, "no_play_rule": True})
    np.testing.assert_array_equal(out, labels)


def test_real_pre_kickoff_window_is_mostly_d_when_enabled(arrays):
    """The point of E10: with the rule on, the FAQ-documented pre-kickoff window
    (frames 110,808-113,741 are graded and all D) flips to mostly D.

    Measured at implementation (2026-11-09): 88.7 % D with the rule on (baseline
    37.4 %); the pre-declared rule marks the 3 sustained-inactivity runs, the decoder
    supplies the rest. Regression floor 85 %. The rule may also fire elsewhere
    (half 1 has qualifying runs) — that is generic behaviour for the IDSSE no-harm
    gate to judge, not something this test may forbid."""
    from src.ball_fill import build_ball_track
    from src.decoder import CODE_X, decode
    from src.emission import build_emission

    track = build_ball_track(arrays.ball, arrays.half, CFG)
    em = build_emission(track, arrays.players, arrays.players_s, arrays.slot_team, CFG)

    on = decode(em, track.source, arrays.half, {**CFG, "no_play_rule": True})
    off = decode(em, track.source, arrays.half, {**CFG, "no_play_rule": False})

    window = slice(110_808, 113_742)                 # FAQ pre-kickoff window
    d_on = float((on.labels[window] == CODE_D).mean())
    d_off = float((off.labels[window] == CODE_D).mean())
    assert d_on >= 0.85, f"pre-kickoff window should be >=85% D with the rule on (got {d_on:.1%})"
    assert d_on >= d_off + 0.4, "the rule must be what flips the window (37 % -> ~89 %)"
    assert (on.labels[190_198:] == CODE_X).all(), "nothing after the graded half changes"
