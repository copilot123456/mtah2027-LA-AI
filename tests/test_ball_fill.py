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
    return build_ball_track(arrays.ball, arrays.half, FILL_ON)  # fill tier enabled via FILL_ON


def test_fill_tier_ships_disabled(arrays):
    """Shipped default: the kinematic tier is off (IDSSE ablation; config provenance)."""
    default = build_ball_track(arrays.ball, arrays.half)  # build with shipped config
    assert (default.source != SOURCE_KINEMATIC).all()  # kinematic tier ships off


def test_atd_never_overwritten(arrays, track):
    """Raw observations ride through bit-exactly; only their provenance label is added."""
    known = ~np.isnan(arrays.ball).any(axis=1)  # frames with real ball
    assert (track.source[known] == SOURCE_ATD).all()  # observations stay ATD
    np.testing.assert_array_equal(track.xy[known], arrays.ball[known])  # positions bit-identical


def test_every_frame_has_exactly_one_source(track):
    """Only allowed source codes appear, and they cover all frames."""
    assert set(np.unique(track.source)) <= {SOURCE_ATD, SOURCE_KINEMATIC, SOURCE_NONE}  # only allowed source codes
    cov = coverage(track)  # summarize source shares
    assert cov["all_atd"] + cov["all_kinematic"] + cov["all_none"] == pytest.approx(1.0)  # shares cover every frame


def test_fill_stays_inside_halves(arrays, track):
    """Fills stay inside a real half, never in ungraded frames."""
    kin = track.source == SOURCE_KINEMATIC  # mask the fills
    assert kin.any(), "expected some fills"  # fills do happen
    assert (arrays.half[kin] > 0).all(), "a fill crossed a half boundary"  # fills never outside halves


def _kinematic_runs(track):
    kin = track.source == SOURCE_KINEMATIC  # locate fill frames
    edges = np.flatnonzero(np.diff(np.concatenate(([0], kin.view(np.int8), [0]))))  # rising/falling fill edges
    return zip(edges[::2], edges[1::2])  # pair rise and fall


def _brackets(known_idx, a, b):
    """Original ATD anchors surrounding a filled run (edge fills may have been dropped)."""
    i0 = known_idx[known_idx < a][-1]  # anchor before run
    i1 = known_idx[known_idx >= b][0]  # anchor after run
    return i0, i1  # both anchors returned


def test_filled_gaps_within_time_limit(arrays, track):
    """No filled gap is longer than the configured time limit."""
    known = ~np.isnan(arrays.ball).any(axis=1)  # observed frames mask
    known_idx = np.flatnonzero(known)  # their indices
    assert known_idx.size > 0  # some observations exist
    limit = FILL_ON["fill_long_gap_s"] * FILL_ON["fps"]  # seconds → frames
    runs = list(_kinematic_runs(track))  # list all fill runs
    assert runs, "expected some fills"  # fills exist to check
    for a, b in runs:  # check each run
        i0, i1 = _brackets(known_idx, a, b)  # anchors around this run
        assert (i1 - i0) <= limit + 1, f"gap {(i1 - i0) / CFG['fps']:.1f}s exceeds limit"  # gap within limit


def test_overshoot_guard(arrays, track):
    """Filled positions never leave the anchors' bbox + pad."""
    known = ~np.isnan(arrays.ball).any(axis=1)  # observed frames mask
    known_idx = np.flatnonzero(known)  # their indices
    pad, pad_frac = CFG["fill_pad"], CFG["fill_pad_frac"]  # pad settings
    for a, b in _kinematic_runs(track):
        i0, i1 = _brackets(known_idx, a, b)  # anchors around this run
        p0, p1 = arrays.ball[i0], arrays.ball[i1]  # anchor positions
        span = np.abs(p1 - p0)  # anchor span
        margin = np.maximum(pad, pad_frac * span)  # absolute or proportional margin
        lo, hi = np.minimum(p0, p1) - margin, np.maximum(p0, p1) + margin  # padded allowed box
        live = ~np.isnan(track.xy[a:b]).any(axis=1)  # filled frames only
        assert (track.xy[a:b][live] >= lo - 1e-12).all()  # not below lower bound
        assert (track.xy[a:b][live] <= hi + 1e-12).all()  # not above upper bound


def test_drop_only_nulls_estimates(arrays, track):
    """No estimate survives outside the margin; ATD positions are flagged, never dropped."""
    pos = track.xy  # current positions
    has = ~np.isnan(pos).any(axis=1)  # frames with a position
    kin = (track.source == SOURCE_KINEMATIC) & has  # live fills only
    margin = CFG["pitch_margin"]  # allowed pitch margin
    inside = ((pos[:, 0] >= -margin) & (pos[:, 0] <= 1 + margin)  # margin test per axis
              & (pos[:, 1] >= -margin) & (pos[:, 1] <= 1 + margin))
    assert inside[kin].all(), "kinematic position outside the pitch margin survived"  # fills stay on pitch


def test_out_of_pitch_flag_covers_atd(arrays, track):
    """Every out-of-pitch ATD frame carries the out_of_pitch flag."""
    m = CFG["pitch_margin"]  # pitch margin reused
    known = ~np.isnan(arrays.ball).any(axis=1)  # observed frames
    atd_out = (known  # observed outside pitch
               & ((arrays.ball[:, 0] < -m) | (arrays.ball[:, 0] > 1 + m)
                  | (arrays.ball[:, 1] < -m) | (arrays.ball[:, 1] > 1 + m)))
    assert (track.out_of_pitch[atd_out]).all() if atd_out.any() else True  # out-of-pitch flag set
    # flagged ATD frames keep their position
    np.testing.assert_array_equal(track.xy[atd_out], arrays.ball[atd_out])


def test_age_s_provenance(arrays, track):
    """Ball age: 0 on ATD frames, positive on fills, bounded by match length."""
    known = ~np.isnan(arrays.ball).any(axis=1)  # observed frames
    assert np.isfinite(track.age_s).all()  # ages always finite
    assert (track.age_s[known] == 0).all(), "ATD frames are age 0 by definition"  # observations age zero
    kin = track.source == SOURCE_KINEMATIC  # filled frames
    assert (track.age_s[kin] > 0).all()  # fills age positive
    assert track.age_s.max() <= (D.N_FRAMES - 1) / CFG["fps"]  # never older than match


def test_deterministic(arrays):
    """Two builds from the same input produce identical tracks."""
    a = build_ball_track(arrays.ball, arrays.half)  # first build
    b = build_ball_track(arrays.ball, arrays.half)  # second identical build
    np.testing.assert_array_equal(a.xy, b.xy)  # NaN == NaN by default here
    np.testing.assert_array_equal(a.source, b.source)  # sources identical
    np.testing.assert_array_equal(a.age_s, b.age_s)  # ages identical
    np.testing.assert_array_equal(a.out_of_pitch, b.out_of_pitch)  # flags identical


def test_inputs_not_mutated(arrays):
    """The build leaves the input ball array unchanged."""
    before = arrays.ball.copy()  # snapshot before build
    build_ball_track(arrays.ball, arrays.half)  # run the build
    np.testing.assert_array_equal(arrays.ball, before)  # input untouched
