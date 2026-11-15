"""Alignment contract tests — the mistakes that silently cost points.

Values marked with tolerances are the aggregates recorded in the tech deep-dive
(2026-11-07); they are structural checks, never tuning signals.
"""

import numpy as np
import pandas as pd
import pytest

from src import data as D


def test_row_count(arrays):
    """Total frames match the frozen dataset count."""
    assert arrays.n_frames == D.N_FRAMES == 199_319  # frozen frame count


def test_frame_id_is_one_based_sequence(arrays):
    """frame_id is a gapless 1..N sequence (alignment contract)."""
    np.testing.assert_array_equal(arrays.frame_id, np.arange(1, D.N_FRAMES + 1))  # gapless 1..N ids
    # submission row r corresponds to ATD frame r+1: video_frame = frame_id - 1


def test_graded_spans(arrays):
    """Graded span edges and the halftime hole match recorded spans."""
    graded = arrays.graded  # frames the scorer grades
    assert int(graded.sum()) == 158_857  # total graded frames
    # half 1: 11185..90651 inclusive
    assert graded[11_184] is np.False_ or not graded[11_184]
    assert graded[11_185] and graded[90_651] and not graded[90_652]  # half-1 boundary exact
    # half 2: 110808..190197 inclusive
    assert not graded[110_807] and graded[110_808] and graded[190_197] and not graded[190_198]
    # halftime hole is ungraded
    assert not graded[90_652:110_808].any()


def test_half_label_counts(arrays):
    """Per-half frame counts match the recorded totals."""
    assert int((arrays.half == 1).sum()) == 79_467  # half-1 frame total
    assert int((arrays.half == 2).sum()) == 79_390  # half-2 frame total
    assert int((arrays.half == 0).sum()) == 40_462  # ungraded frames


def test_slot_teams_shape(arrays):
    """Every slot carries a binary team code (0 or 1)."""
    assert arrays.slot_team.shape == (D.N_SLOTS,)  # one code per slot
    assert set(np.unique(arrays.slot_team)) == {0, 1}  # binary codes only


def test_ball_missing_rates(arrays):
    """Ball-missing rates sit within tolerance of the recorded aggregates."""
    ball_missing = np.isnan(arrays.ball).any(axis=1)  # frames missing ball x or y
    overall = 100 * ball_missing.mean()  # percent across all frames
    graded = 100 * (ball_missing & arrays.graded).sum() / arrays.graded.sum()  # percent on graded frames
    assert overall == pytest.approx(43.1, abs=0.2)  # recorded overall rate
    assert graded == pytest.approx(40.7, abs=0.2)  # recorded graded rate


def test_players_visible_mean(arrays):
    """Visible-player mean on graded frames matches the aggregate."""
    nvis = (~np.isnan(arrays.players)).sum(axis=1)  # visible players per frame
    assert nvis[arrays.graded].mean() == pytest.approx(20.5, abs=0.3)  # recorded visible average


def test_provider_team_fields_are_empty(arrays):
    """Documented finding: the automatic ATD carries no team attribution.

    If this ever fails, the data changed — re-check whether a provider field
    became available (it would be an input, not a label).
    """
    df = D.load_tracking()  # raw tracking columns
    assert df["ball_state"].notna().sum() == 0  # no ball_state values
    assert df["ball_owning_team_id"].notna().sum() == 0  # no owning-team values


def test_homography_alignment():
    """Homography covers every frame and stays fully valid."""
    hg = D.load_homography()  # per-frame homography
    assert len(hg) == D.N_FRAMES  # one row per frame
    assert int(hg["frame"].iloc[0]) == 0 and int(hg["frame"].iloc[-1]) == D.N_FRAMES - 1  # spans the whole video
    assert (hg["valid"] == 1).all(), "tactical homography must be valid on every frame"  # every warp usable


def test_smart_tagging_time_base():
    """Tags stay inside the video range and end after they start."""
    tags = D.load_smart_tagging()  # smart-tag table
    assert len(tags) == 638  # fixed tag total
    assert int((tags["code"] == "SET PIECES").sum()) == 59  # set-piece tag count
    assert tags["start_frame"].between(0, D.N_FRAMES - 1).all()  # starts in range
    assert tags["end_frame"].between(0, D.N_FRAMES - 1).all()  # ends in range
    assert (tags["end_frame"] >= tags["start_frame"]).all()  # ends never before starts


def test_arrays_consistent_shapes(arrays):
    """All arrays share the frame axis with the expected shapes."""
    n = arrays.n_frames  # shared frame count
    assert arrays.ball.shape == (n, 2)  # ball xy per frame
    assert arrays.players.shape == (n, D.N_SLOTS, 2)  # player grid per frame
    assert arrays.players_s.shape == (n, D.N_SLOTS)  # one speed per slot
    assert arrays.graded.shape == (n,)  # graded flag per frame
    assert arrays.half.shape == (n,)  # half id per frame
