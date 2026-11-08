"""P2 acceptance: swap symmetry, flat at conf=0, NaN-free, duel neutrality,
confidence semantics, dead-ball monotonicity, determinism, real-data integration."""

import numpy as np
import pytest

from config import CFG
from src.ball_fill import SOURCE_ATD, SOURCE_KINEMATIC, SOURCE_NONE, BallTrack
from src.emission import CLASS_A, CLASS_D, CLASS_H, build_emission, team_confidence

N = 40


def _slots_on_grid():
    """46 slots spread over the pitch, normalized coords, no NaNs."""
    xs = np.linspace(0.05, 0.95, 46)
    ys = np.linspace(0.05, 0.95, 24)
    xy = np.stack([xs, np.resize(ys, 46)], axis=-1)
    return np.tile(xy, (N, 1, 1))          # (N, 46, 2)


def _track(ball_xy, source=None, age=None, out=None):
    n = len(ball_xy)
    source = np.full(n, SOURCE_ATD, dtype=np.int8) if source is None else source
    age = np.zeros(n) if age is None else age.astype(float)
    out = np.zeros(n, dtype=bool) if out is None else out
    return BallTrack(xy=ball_xy, source=source, age_s=age, out_of_pitch=out)


def test_shape_and_finite():
    ball = np.full((N, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.5, 0.5
    track = _track(ball)
    players = _slots_on_grid()
    em = build_emission(track, players, np.full((N, 46), 3.0), np.arange(46) % 2)
    assert em.log_e.shape == (N, 3)
    assert np.isfinite(em.log_e).all()
    assert np.isfinite(em.p_d).all() and ((em.p_d > 0) & (em.p_d < 1)).all()
    assert (em.conf >= 0).all() and (em.conf <= 1).all()


def test_ha_swap_symmetry():
    """Swapping team labels swaps the H and A columns exactly (R-3.2)."""
    ball = np.full((N, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.5, 0.5
    track = _track(ball)
    players = _slots_on_grid()
    speeds = np.full((N, 46), 5.0)
    teams = (np.arange(46) % 2).astype(np.int8)

    em1 = build_emission(track, players, speeds, teams)
    em2 = build_emission(track, players, speeds, 1 - teams)

    np.testing.assert_allclose(em1.log_e[:, CLASS_H], em2.log_e[:, CLASS_A], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(em1.log_e[:, CLASS_A], em2.log_e[:, CLASS_H], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(em1.log_e[:, CLASS_D], em2.log_e[:, CLASS_D], rtol=1e-12, atol=1e-12)


def test_no_ball_means_no_team_information():
    """conf=0 (source none): H and A columns are exactly equal (R-3.1)."""
    ball = np.full((N, 2), np.nan)
    source = np.full(N, SOURCE_NONE, dtype=np.int8)
    track = BallTrack(xy=ball, source=source, age_s=np.full(N, 12.0), out_of_pitch=np.zeros(N, bool))
    em = build_emission(track, _slots_on_grid(), np.full((N, 46), 1.0), np.arange(46) % 2)
    np.testing.assert_array_equal(em.log_e[:, CLASS_H], em.log_e[:, CLASS_A])
    assert (em.conf == 0).all()


def test_duel_gives_neutral_margin():
    """Ball equidistant from one H and one A player => equal H/A columns at conf=1."""
    ball = np.full((N, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.5, 0.5
    track = _track(ball)
    players = np.full((N, 46, 2), np.nan)
    players[:, 0] = [0.45, 0.5]   # H player 5 m left (approx)
    players[:, 1] = [0.55, 0.5]   # A player 5 m right
    teams = np.zeros(46, dtype=np.int8)
    teams[1] = 1
    em = build_emission(track, players, np.full((N, 46), 5.0), teams)
    np.testing.assert_allclose(em.log_e[:, CLASS_H], em.log_e[:, CLASS_A], atol=1e-9)
    np.testing.assert_allclose(em.margin, 0.0, atol=1e-9)


def test_near_ball_team_wins():
    """Ball on top of an H player, A far away => H column strictly greater."""
    ball = np.full((N, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.3, 0.4
    track = _track(ball)
    players = np.full((N, 46, 2), np.nan)
    players[:, 0] = [0.3, 0.4]          # H player on the ball
    players[:, 1] = [0.9, 0.9]          # A player far
    teams = np.zeros(46, dtype=np.int8)
    teams[1] = 1
    em = build_emission(track, players, np.full((N, 46), 5.0), teams)
    assert (em.log_e[:, CLASS_H] > em.log_e[:, CLASS_A]).all()


def test_invisible_team_hits_the_log_floor_not_minus_inf():
    """Only H players visible: p_team_A is exactly 0 and the A column is clipped at EPS."""
    ball = np.full((N, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.3, 0.4
    track = _track(ball)
    players = np.full((N, 46, 2), np.nan)
    players[:, 0] = [0.3, 0.4]                 # one H player on the ball, no A visible
    teams = np.zeros(46, dtype=np.int8)
    teams[1:] = 1
    em = build_emission(track, players, np.full((N, 46), 5.0), teams)
    assert np.isfinite(em.log_e).all()
    assert (em.ev[:, 1] == 0).all()
    assert (em.log_e[:, CLASS_H] > em.log_e[:, CLASS_A]).all()
    assert em.log_e[0, CLASS_A] == pytest.approx(np.log(1e-12) + np.log(1 - em.p_d[0]))


def test_ball_speed_nan_edges_do_not_count_as_stationary():
    """Frame 0 and frames next to a missing ball have no speed; they must not look at rest."""
    from src.emission import _ball_speed_ms, _dead_features

    ball = np.full((N, 2), 0.5)
    ball[5] = np.nan                           # one missing frame in a still ball
    track = _track(ball)
    speed = _ball_speed_ms(track, CFG)
    assert np.isnan(speed[0]) and np.isnan(speed[5]) and np.isnan(speed[6])
    assert speed[1] == 0.0 and speed[10] == 0.0
    feats = _dead_features(track, np.full((N, 46), 3.0), CFG)
    assert feats["stationary"][0] == 0.0 and feats["stationary"][5] == 0.0 and feats["stationary"][6] == 0.0
    assert feats["stationary"][1] == 1.0


def test_confidence_semantics():
    """atd conf = c0_atd; kinematic decays with age; none conf = 0."""
    ball = np.full((N, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.5, 0.5
    source = np.array([SOURCE_ATD, SOURCE_KINEMATIC, SOURCE_KINEMATIC, SOURCE_NONE] + [SOURCE_ATD] * (N - 4),
                      dtype=np.int8)
    age = np.zeros(N)
    age[1], age[2] = 0.5, 5.0
    track = BallTrack(xy=ball, source=source, age_s=age, out_of_pitch=np.zeros(N, bool))
    conf = team_confidence(track)
    assert conf[0] == CFG["c0_atd"]
    assert conf[1] > conf[2] > 0, "kinematic confidence must decay with age"
    assert conf[1] == pytest.approx(CFG["c0_kinematic"] * np.exp(-0.5 / CFG["tau_kinematic_s"]))
    assert conf[3] == 0.0


def test_dead_ball_monotonicity():
    """Out-of-pitch position => higher p_d than the identical in-pitch frame."""
    ball = np.full((N, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.5, 0.5
    players = _slots_on_grid()
    speeds = np.full((N, 46), 5.0)
    teams = np.arange(46) % 2

    em_in = build_emission(_track(ball), players, speeds, teams)
    em_out = build_emission(_track(ball, out=np.ones(N, bool)), players, speeds, teams)
    assert (em_out.p_d > em_in.p_d).all()


def test_dead_features_raise_pd():
    """Stationary + long-missing + low-activity frame beats a fast active one."""
    ball = np.full((N, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.5, 0.5
    players = _slots_on_grid()
    teams = np.arange(46) % 2

    still = BallTrack(xy=ball,
                      source=np.full(N, SOURCE_NONE, dtype=np.int8),
                      age_s=np.full(N, 30.0),
                      out_of_pitch=np.zeros(N, bool))

    moving = ball.copy()
    moving[:, 0] += np.arange(N) * 0.01          # ball in motion each frame
    fast = BallTrack(xy=moving,
                     source=np.full(N, SOURCE_ATD, dtype=np.int8),
                     age_s=np.zeros(N),
                     out_of_pitch=np.zeros(N, bool))

    em_slow = build_emission(still, players, np.full((N, 46), 0.5), teams)
    em_fast = build_emission(fast, players, np.full((N, 46), 6.0), teams)
    assert (em_slow.p_d > em_fast.p_d).all()


def test_deterministic_and_inputs_untouched():
    ball = np.full((N, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.5, 0.5
    track = _track(ball)
    players = _slots_on_grid()
    speeds = np.full((N, 46), 3.0)
    teams = np.arange(46) % 2
    before_p, before_s = players.copy(), speeds.copy()

    a = build_emission(track, players, speeds, teams)
    b = build_emission(track, players, speeds, teams)
    np.testing.assert_array_equal(a.log_e, b.log_e)
    np.testing.assert_array_equal(a.p_d, b.p_d)
    np.testing.assert_array_equal(players, before_p)
    np.testing.assert_array_equal(speeds, before_s)


def test_real_data_integration(arrays):
    """Full-match run: shapes, finiteness, conf by source."""
    from src.ball_fill import build_ball_track

    track = build_ball_track(arrays.ball, arrays.half)
    em = build_emission(track, arrays.players, arrays.players_s, arrays.slot_team)
    n = arrays.n_frames
    assert em.log_e.shape == (n, 3)
    assert np.isfinite(em.log_e).all(), "emission must be NaN/inf free"
    assert (em.conf[track.source == SOURCE_ATD] == CFG["c0_atd"]).all()
    assert (em.conf[track.source == SOURCE_NONE] == 0).all()
    kin = track.source == SOURCE_KINEMATIC
    if kin.any():
        assert ((em.conf[kin] > 0) & (em.conf[kin] < CFG["c0_atd"])).all()
