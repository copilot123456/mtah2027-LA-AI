"""P2 acceptance: swap symmetry, flat at conf=0, NaN-free, duel neutrality,
confidence semantics, dead-ball monotonicity, determinism, real-data integration."""

import numpy as np
import pytest

from config import CFG
from src.ball_fill import SOURCE_ATD, SOURCE_KINEMATIC, SOURCE_NONE, BallTrack
from src.emission import CLASS_A, CLASS_D, CLASS_H, build_emission, team_confidence

N = 40  # frames per toy batch


def _slots_on_grid():
    """46 slots spread over the pitch, normalized coords, no NaNs."""
    xs = np.linspace(0.05, 0.95, 46)  # slot x across pitch
    ys = np.linspace(0.05, 0.95, 24)  # slot y positions
    xy = np.stack([xs, np.resize(ys, 46)], axis=-1)  # xy pair per slot
    return np.tile(xy, (N, 1, 1))          # (N, 46, 2)


# defaults describe a clean track: all-ATD source, age 0, no flags
def _track(ball_xy, source=None, age=None, out=None):
    n = len(ball_xy)  # one row per frame
    source = np.full(n, SOURCE_ATD, dtype=np.int8) if source is None else source  # default source ATD
    age = np.zeros(n) if age is None else age.astype(float)  # default age zero
    out = np.zeros(n, dtype=bool) if out is None else out  # default nothing flagged
    return BallTrack(xy=ball_xy, source=source, age_s=age, out_of_pitch=out)  # assemble the track


def test_shape_and_finite():
    """Emission is well-formed: right shape, finite, valid probabilities."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    ball[:, 0], ball[:, 1] = 0.5, 0.5  # ball at centre
    track = _track(ball)  # wrap as track
    players = _slots_on_grid()  # fixed player grid
    em = build_emission(track, players, np.full((N, 46), 3.0), np.arange(46) % 2)  # mild speeds, split teams
    assert em.log_e.shape == (N, 3)  # N rows, three classes
    assert np.isfinite(em.log_e).all()  # no NaN or inf
    assert np.isfinite(em.p_d).all() and ((em.p_d > 0) & (em.p_d < 1)).all()  # strict probability
    assert (em.conf >= 0).all() and (em.conf <= 1).all()  # confidence within [0, 1]


def test_ha_swap_symmetry():
    """Swapping team labels swaps the H and A columns exactly (R-3.2)."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    ball[:, 0], ball[:, 1] = 0.5, 0.5  # ball at centre
    track = _track(ball)  # wrap as track
    players = _slots_on_grid()  # fixed player grid
    speeds = np.full((N, 46), 5.0)  # uniform moderate speeds
    teams = (np.arange(46) % 2).astype(np.int8)  # alternate team slots

    em1 = build_emission(track, players, speeds, teams)  # baseline emission
    em2 = build_emission(track, players, speeds, 1 - teams)  # teams flipped

    np.testing.assert_allclose(em1.log_e[:, CLASS_H], em2.log_e[:, CLASS_A], rtol=1e-12, atol=1e-12)  # H becomes A
    np.testing.assert_allclose(em1.log_e[:, CLASS_A], em2.log_e[:, CLASS_H], rtol=1e-12, atol=1e-12)  # A becomes H
    np.testing.assert_allclose(em1.log_e[:, CLASS_D], em2.log_e[:, CLASS_D], rtol=1e-12, atol=1e-12)  # D unchanged


def test_no_ball_means_no_team_information():
    """conf=0 (source none): H and A columns are exactly equal (R-3.1)."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    source = np.full(N, SOURCE_NONE, dtype=np.int8)  # source set to none
    track = BallTrack(xy=ball, source=source, age_s=np.full(N, 12.0), out_of_pitch=np.zeros(N, bool))  # stale no-ball track
    em = build_emission(track, _slots_on_grid(), np.full((N, 46), 1.0), np.arange(46) % 2)  # build emission
    np.testing.assert_array_equal(em.log_e[:, CLASS_H], em.log_e[:, CLASS_A])  # H and A equal
    assert (em.conf == 0).all()  # zero confidence


def test_duel_gives_neutral_margin():
    """Ball equidistant from one H and one A player => equal H/A columns at conf=1."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    ball[:, 0], ball[:, 1] = 0.5, 0.5  # ball at centre
    track = _track(ball)  # wrap as track
    players = np.full((N, 46, 2), np.nan)  # all slots invisible
    players[:, 0] = [0.45, 0.5]   # H player 5 m left (approx)
    players[:, 1] = [0.55, 0.5]   # A player 5 m right
    teams = np.zeros(46, dtype=np.int8)  # everyone home team
    teams[1] = 1  # slot 1 away
    em = build_emission(track, players, np.full((N, 46), 5.0), teams)  # build duel emission
    np.testing.assert_allclose(em.log_e[:, CLASS_H], em.log_e[:, CLASS_A], atol=1e-9)  # duel is neutral
    np.testing.assert_allclose(em.margin, 0.0, atol=1e-9)  # zero margin


def test_near_ball_team_wins():
    """Ball on top of an H player, A far away => H column strictly greater."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    ball[:, 0], ball[:, 1] = 0.3, 0.4  # ball placed
    track = _track(ball)  # wrap as track
    players = np.full((N, 46, 2), np.nan)  # all slots invisible
    players[:, 0] = [0.3, 0.4]          # H player on the ball
    players[:, 1] = [0.9, 0.9]          # A player far
    teams = np.zeros(46, dtype=np.int8)  # everyone home team
    teams[1] = 1  # slot 1 away
    em = build_emission(track, players, np.full((N, 46), 5.0), teams)  # build emission
    assert (em.log_e[:, CLASS_H] > em.log_e[:, CLASS_A]).all()  # home team wins


def test_invisible_team_hits_the_log_floor_not_minus_inf():
    """Only H players visible: p_team_A is exactly 0 and the A column is clipped at EPS."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    ball[:, 0], ball[:, 1] = 0.3, 0.4  # ball placed
    track = _track(ball)  # wrap as track
    players = np.full((N, 46, 2), np.nan)  # all slots invisible
    players[:, 0] = [0.3, 0.4]                 # one H player on the ball, no A visible
    teams = np.zeros(46, dtype=np.int8)  # everyone home team
    teams[1:] = 1  # rest of slots away
    em = build_emission(track, players, np.full((N, 46), 5.0), teams)  # build emission
    assert np.isfinite(em.log_e).all()  # stays finite
    assert (em.ev[:, 1] == 0).all()  # no team-A evidence visible
    assert (em.log_e[:, CLASS_H] > em.log_e[:, CLASS_A]).all()  # home still preferred
    assert em.log_e[0, CLASS_A] == pytest.approx(np.log(1e-12) + np.log(1 - em.p_d[0]))  # EPS floor, not -inf


def test_ball_speed_nan_edges_do_not_count_as_stationary():
    """Frame 0 and frames next to a missing ball have no speed; they must not look at rest."""
    from src.emission import _ball_speed_ms, _dead_features

    ball = np.full((N, 2), 0.5)  # still ball mid-pitch
    ball[5] = np.nan                           # one missing frame in a still ball
    track = _track(ball)  # wrap as track
    speed = _ball_speed_ms(track, CFG)  # per-frame ball speed
    assert np.isnan(speed[0]) and np.isnan(speed[5]) and np.isnan(speed[6])  # undefined at edges and gaps
    assert speed[1] == 0.0 and speed[10] == 0.0  # still frames speed zero
    feats = _dead_features(track, np.full((N, 46), 3.0), CFG)  # extract dead-ball features
    assert feats["stationary"][0] == 0.0 and feats["stationary"][5] == 0.0 and feats["stationary"][6] == 0.0  # NaN speed not at rest
    assert feats["stationary"][1] == 1.0  # genuinely at rest


def test_confidence_semantics():
    """atd conf = c0_atd; kinematic decays with age; none conf = 0."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    ball[:, 0], ball[:, 1] = 0.5, 0.5  # ball at centre
    source = np.array([SOURCE_ATD, SOURCE_KINEMATIC, SOURCE_KINEMATIC, SOURCE_NONE] + [SOURCE_ATD] * (N - 4),  # mixed sources per frame
                      dtype=np.int8)
    age = np.zeros(N)  # ages start zero
    age[1], age[2] = 0.5, 5.0  # two fill ages
    track = BallTrack(xy=ball, source=source, age_s=age, out_of_pitch=np.zeros(N, bool))  # assemble mixed track
    conf = team_confidence(track)  # compute confidences
    assert conf[0] == CFG["c0_atd"]  # ATD confidence constant
    assert conf[1] > conf[2] > 0, "kinematic confidence must decay with age"  # decay order holds
    assert conf[1] == pytest.approx(CFG["c0_kinematic"] * np.exp(-0.5 / CFG["tau_kinematic_s"]))  # exponential decay
    assert conf[3] == 0.0  # no ball, no confidence


def test_dead_ball_monotonicity():
    """Out-of-pitch position => higher p_d than the identical in-pitch frame."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    ball[:, 0], ball[:, 1] = 0.5, 0.5  # ball at centre
    players = _slots_on_grid()  # fixed player grid
    speeds = np.full((N, 46), 5.0)  # moderate speeds
    teams = np.arange(46) % 2  # alternating teams

    em_in = build_emission(_track(ball), players, speeds, teams)  # in-pitch baseline
    em_out = build_emission(_track(ball, out=np.ones(N, bool)), players, speeds, teams)  # same frame flagged out
    assert (em_out.p_d > em_in.p_d).all()  # out flag raises p_d


def test_dead_features_raise_pd():
    """Stationary + long-missing + low-activity frame beats a fast active one."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    ball[:, 0], ball[:, 1] = 0.5, 0.5  # ball at centre
    players = _slots_on_grid()  # fixed player grid
    teams = np.arange(46) % 2  # alternating teams

    still = BallTrack(xy=ball,  # long-dead still track
                      source=np.full(N, SOURCE_NONE, dtype=np.int8),
                      age_s=np.full(N, 30.0),
                      out_of_pitch=np.zeros(N, bool))

    moving = ball.copy()  # copy the ball
    moving[:, 0] += np.arange(N) * 0.01          # ball in motion each frame
    fast = BallTrack(xy=moving,  # live moving track
                     source=np.full(N, SOURCE_ATD, dtype=np.int8),
                     age_s=np.zeros(N),
                     out_of_pitch=np.zeros(N, bool))

    em_slow = build_emission(still, players, np.full((N, 46), 0.5), teams)  # slow dead frame
    em_fast = build_emission(fast, players, np.full((N, 46), 6.0), teams)  # fast live frame
    assert (em_slow.p_d > em_fast.p_d).all()  # deadlier frame wins


def test_deterministic_and_inputs_untouched():
    """Same inputs reproduce identical emissions and leave inputs unmodified."""
    ball = np.full((N, 2), np.nan)  # NaN placeholder ball
    ball[:, 0], ball[:, 1] = 0.5, 0.5  # ball at centre
    track = _track(ball)  # wrap as track
    players = _slots_on_grid()  # fixed player grid
    speeds = np.full((N, 46), 3.0)  # mild speeds
    teams = np.arange(46) % 2  # alternating teams
    before_p, before_s = players.copy(), speeds.copy()  # snapshot the inputs

    a = build_emission(track, players, speeds, teams)  # first run
    b = build_emission(track, players, speeds, teams)  # second run same input
    np.testing.assert_array_equal(a.log_e, b.log_e)  # identical log_e
    np.testing.assert_array_equal(a.p_d, b.p_d)  # identical p_d
    np.testing.assert_array_equal(players, before_p)  # players untouched
    np.testing.assert_array_equal(speeds, before_s)  # speeds untouched


def test_real_data_integration(arrays):
    """Full-match run: shapes, finiteness, conf by source."""
    from src.ball_fill import build_ball_track

    track = build_ball_track(arrays.ball, arrays.half)  # real ball track
    em = build_emission(track, arrays.players, arrays.players_s, arrays.slot_team)  # emission on real data
    n = arrays.n_frames  # real frame count
    assert em.log_e.shape == (n, 3)  # full-match shape
    assert np.isfinite(em.log_e).all(), "emission must be NaN/inf free"  # all values finite
    assert (em.conf[track.source == SOURCE_ATD] == CFG["c0_atd"]).all()  # ATD conf by source
    assert (em.conf[track.source == SOURCE_NONE] == 0).all()  # none conf is zero
    kin = track.source == SOURCE_KINEMATIC  # filled frames
    if kin.any():  # only if fills exist
        assert ((em.conf[kin] > 0) & (em.conf[kin] < CFG["c0_atd"])).all()  # decayed but positive
