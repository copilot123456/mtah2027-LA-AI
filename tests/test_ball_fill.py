"""P1 acceptance: ATD never mutated, one source per frame, fills stay inside
halves and within the time limit, overshoot guarded, drop only touches estimates."""

import numpy as np
import pytest

from config import CFG
from src import data as D
from src.ball_fill import (
    SOURCE_ATD,
    SOURCE_KINEMATIC,
    SOURCE_NONE,
    build_ball_track,
    coverage,
)


FILL_ON = {**CFG, "fill_long_gap_s": 10.0}   # mechanism tests: tier enabled


@pytest.fixture(scope="session")
def track(arrays):
    """Fill-enabled track for the mechanism tests (the shipped config has the tier off;
    see test_fill_tier_ships_disabled)."""
    return build_ball_track(arrays.ball, arrays.half, FILL_ON)


def test_fill_tier_ships_disabled(arrays):
    """Shipped default: the kinematic tier is off (IDSSE ablation; config provenance)."""
    default = build_ball_track(arrays.ball, arrays.half)
    assert (default.source != SOURCE_KINEMATIC).all()


def test_atd_never_overwritten(arrays, track):
    """Raw observations ride through bit-exactly; only their provenance label is added."""
    known = ~np.isnan(arrays.ball).any(axis=1)
    assert (track.source[known] == SOURCE_ATD).all()
    np.testing.assert_array_equal(track.xy[known], arrays.ball[known])


def test_every_frame_has_exactly_one_source(track):
    """Only allowed source codes appear, and they cover all frames."""
    assert set(np.unique(track.source)) <= {SOURCE_ATD, SOURCE_KINEMATIC, SOURCE_NONE}
    cov = coverage(track)
    assert cov["all_atd"] + cov["all_kinematic"] + cov["all_none"] == pytest.approx(1.0)


def test_fill_stays_inside_halves(arrays, track):
    """Fills stay inside a real half, never in ungraded frames."""
    kin = track.source == SOURCE_KINEMATIC
    assert kin.any(), "expected some fills"
    assert (arrays.half[kin] > 0).all(), "a fill crossed a half boundary"


def _kinematic_runs(track):
    kin = track.source == SOURCE_KINEMATIC
    edges = np.flatnonzero(np.diff(np.concatenate(([0], kin.view(np.int8), [0]))))  # rising/falling fill edges
    return zip(edges[::2], edges[1::2])


def _brackets(known_idx, a, b):
    """Original ATD anchors surrounding a filled run (edge fills may have been dropped)."""
    i0 = known_idx[known_idx < a][-1]
    i1 = known_idx[known_idx >= b][0]
    return i0, i1


def test_filled_gaps_within_time_limit(arrays, track):
    """No filled gap is longer than the configured time limit."""
    known = ~np.isnan(arrays.ball).any(axis=1)
    known_idx = np.flatnonzero(known)
    assert known_idx.size > 0
    limit = FILL_ON["fill_long_gap_s"] * FILL_ON["fps"]  # seconds → frames
    runs = list(_kinematic_runs(track))
    assert runs, "expected some fills"
    for a, b in runs:
        i0, i1 = _brackets(known_idx, a, b)
        assert (i1 - i0) <= limit + 1, f"gap {(i1 - i0) / CFG['fps']:.1f}s exceeds limit"


def test_overshoot_guard(arrays, track):
    """Filled positions never leave the anchors' bbox + pad."""
    known = ~np.isnan(arrays.ball).any(axis=1)
    known_idx = np.flatnonzero(known)
    pad, pad_frac = CFG["fill_pad"], CFG["fill_pad_frac"]
    for a, b in _kinematic_runs(track):
        i0, i1 = _brackets(known_idx, a, b)
        p0, p1 = arrays.ball[i0], arrays.ball[i1]
        span = np.abs(p1 - p0)
        margin = np.maximum(pad, pad_frac * span)  # absolute or proportional margin
        lo, hi = np.minimum(p0, p1) - margin, np.maximum(p0, p1) + margin
        live = ~np.isnan(track.xy[a:b]).any(axis=1)
        assert (track.xy[a:b][live] >= lo - 1e-12).all()
        assert (track.xy[a:b][live] <= hi + 1e-12).all()


def test_drop_only_nulls_estimates(arrays, track):
    """No estimate survives outside the margin; ATD positions are flagged, never dropped."""
    pos = track.xy
    has = ~np.isnan(pos).any(axis=1)
    kin = (track.source == SOURCE_KINEMATIC) & has
    margin = CFG["pitch_margin"]
    inside = ((pos[:, 0] >= -margin) & (pos[:, 0] <= 1 + margin)
              & (pos[:, 1] >= -margin) & (pos[:, 1] <= 1 + margin))
    assert inside[kin].all(), "kinematic position outside the pitch margin survived"


def test_out_of_pitch_flag_covers_atd(arrays, track):
    """Every out-of-pitch ATD frame carries the out_of_pitch flag."""
    m = CFG["pitch_margin"]
    known = ~np.isnan(arrays.ball).any(axis=1)
    atd_out = (known
               & ((arrays.ball[:, 0] < -m) | (arrays.ball[:, 0] > 1 + m)
                  | (arrays.ball[:, 1] < -m) | (arrays.ball[:, 1] > 1 + m)))
    assert (track.out_of_pitch[atd_out]).all() if atd_out.any() else True
    # flagged ATD frames keep their position
    np.testing.assert_array_equal(track.xy[atd_out], arrays.ball[atd_out])


def test_age_s_provenance(arrays, track):
    """Ball age: 0 on ATD frames, positive on fills, bounded by match length."""
    known = ~np.isnan(arrays.ball).any(axis=1)
    assert np.isfinite(track.age_s).all()
    assert (track.age_s[known] == 0).all(), "ATD frames are age 0 by definition"
    kin = track.source == SOURCE_KINEMATIC
    assert (track.age_s[kin] > 0).all()
    assert track.age_s.max() <= (D.N_FRAMES - 1) / CFG["fps"]


def test_deterministic(arrays):
    """Two builds from the same input produce identical tracks."""
    a = build_ball_track(arrays.ball, arrays.half)
    b = build_ball_track(arrays.ball, arrays.half)
    np.testing.assert_array_equal(a.xy, b.xy)  # NaN == NaN by default here
    np.testing.assert_array_equal(a.source, b.source)
    np.testing.assert_array_equal(a.age_s, b.age_s)
    np.testing.assert_array_equal(a.out_of_pitch, b.out_of_pitch)


def test_inputs_not_mutated(arrays):
    """The build leaves the input ball array unchanged."""
    before = arrays.ball.copy()
    build_ball_track(arrays.ball, arrays.half)
    np.testing.assert_array_equal(arrays.ball, before)
