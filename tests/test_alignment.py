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
    assert arrays.n_frames == D.N_FRAMES == 199_319


def test_frame_id_is_one_based_sequence(arrays):
    """frame_id is a gapless 1..N sequence (alignment contract)."""
    np.testing.assert_array_equal(arrays.frame_id, np.arange(1, D.N_FRAMES + 1))
    # submission row r corresponds to ATD frame r+1: video_frame = frame_id - 1


def test_graded_spans(arrays):
    """Graded span edges and the halftime hole match recorded spans."""
    graded = arrays.graded
    assert int(graded.sum()) == 158_857
    # half 1: 11185..90651 inclusive
    assert graded[11_184] is np.False_ or not graded[11_184]
    assert graded[11_185] and graded[90_651] and not graded[90_652]
    # half 2: 110808..190197 inclusive
    assert not graded[110_807] and graded[110_808] and graded[190_197] and not graded[190_198]
    # halftime hole is ungraded
    assert not graded[90_652:110_808].any()


def test_half_label_counts(arrays):
    """Per-half frame counts match the recorded totals."""
    assert int((arrays.half == 1).sum()) == 79_467
    assert int((arrays.half == 2).sum()) == 79_390
    assert int((arrays.half == 0).sum()) == 40_462  # ungraded frames


def test_slot_teams_shape(arrays):
    """Every slot carries a binary team code (0 or 1)."""
    assert arrays.slot_team.shape == (D.N_SLOTS,)
    assert set(np.unique(arrays.slot_team)) == {0, 1}


def test_ball_missing_rates(arrays):
    """Ball-missing rates sit within tolerance of the recorded aggregates."""
    ball_missing = np.isnan(arrays.ball).any(axis=1)
    overall = 100 * ball_missing.mean()
    graded = 100 * (ball_missing & arrays.graded).sum() / arrays.graded.sum()
    assert overall == pytest.approx(43.1, abs=0.2)
    assert graded == pytest.approx(40.7, abs=0.2)


def test_players_visible_mean(arrays):
    """Visible-player mean on graded frames matches the aggregate."""
    nvis = (~np.isnan(arrays.players)).sum(axis=1)
    assert nvis[arrays.graded].mean() == pytest.approx(20.5, abs=0.3)


def test_provider_team_fields_are_empty(arrays):
    """Documented finding: the automatic ATD carries no team attribution.

    If this ever fails, the data changed — re-check whether a provider field
    became available (it would be an input, not a label).
    """
    df = D.load_tracking()
    assert df["ball_state"].notna().sum() == 0
    assert df["ball_owning_team_id"].notna().sum() == 0


def test_homography_alignment():
    """Homography covers every frame and stays fully valid."""
    hg = D.load_homography()
    assert len(hg) == D.N_FRAMES
    assert int(hg["frame"].iloc[0]) == 0 and int(hg["frame"].iloc[-1]) == D.N_FRAMES - 1
    assert (hg["valid"] == 1).all(), "tactical homography must be valid on every frame"


def test_smart_tagging_time_base():
    """Tags stay inside the video range and end after they start."""
    tags = D.load_smart_tagging()
    assert len(tags) == 638
    assert int((tags["code"] == "SET PIECES").sum()) == 59
    assert tags["start_frame"].between(0, D.N_FRAMES - 1).all()
    assert tags["end_frame"].between(0, D.N_FRAMES - 1).all()
    assert (tags["end_frame"] >= tags["start_frame"]).all()


def test_arrays_consistent_shapes(arrays):
    """All arrays share the frame axis with the expected shapes."""
    n = arrays.n_frames
    assert arrays.ball.shape == (n, 2)
    assert arrays.players.shape == (n, D.N_SLOTS, 2)
    assert arrays.players_s.shape == (n, D.N_SLOTS)
    assert arrays.graded.shape == (n,)
    assert arrays.half.shape == (n,)
