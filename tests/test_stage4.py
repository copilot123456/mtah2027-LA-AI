"""Stage-4 unit tests: G6 grid integrity, the three phase cues, weights, alignment maths.

Everything here is pure — no IDSSE files, no network, no scoring run. The mask-weight
test is seeded and statistical by construction (weights 1 vs 500 make the concentration
overwhelming for any seed).
"""

import numpy as np

from config import CFG
from tuning.simulate import (
    crowding_cue,
    event_cue,
    map_event_rows,
    mask_ball,
    phase_weights,
    possession_change_cue,
)
from tuning.stage4 import GRID6, KINDS, grid6_points
from tuning.sweep import PARAMS

FROZEN = {"switch_pen": 2.0, "dead_pen": 5.0, "sigma_m": 1.5,
          "min_dwell_s": 2.0, "switch_support_s": 1.5}


def test_grid6_is_144_and_contains_the_frozen_cell():
    points = grid6_points()
    assert set(GRID6) == set(PARAMS)
    assert len(points) == 144 == 3 * 3 * 2 * 4 * 2
    assert FROZEN in points
    assert KINDS == ("uniform", "phase_a2", "phase_a3", "phase_a2_fill")


def test_possession_change_cue_exact_and_dilated():
    letters = np.array(list("HHDAAXA"), dtype=object)
    raw = possession_change_cue(letters, half_s=0.0, fps=1.0)
    assert raw.tolist() == [True, False, False, True, False, False, True]
    wide = possession_change_cue(letters, half_s=2.0, fps=1.0)   # radius 2
    # changes at 0, 3, 6 with radius 2 overlap: the whole span is covered
    assert wide.all()


def test_event_cue_window():
    cue = event_cue(np.array([10]), 20, half_s=2.0, fps=1.0)
    expected = np.zeros(20, dtype=bool)
    expected[8:13] = True
    assert cue.tolist() == expected.tolist()
    assert not event_cue(np.array([-5, 99]), 20, half_s=2.0, fps=1.0).any()


def test_crowding_cue_ball_anchor_and_count():
    # 3 frames, 4 slots; centre of pitch at (0.5, 0.5)
    players = np.full((3, 4, 2), np.nan)
    ball = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    near_dy = 2.0 / CFG["pitch_width_m"]          # 2 m north of centre
    players[0, :3] = [[0.5, 0.5 + near_dy], [0.5, 0.5 - near_dy], [0.5, 0.5]]
    players[0, 3] = [0.2, 0.2]                    # far slot: irrelevant
    players[1, :2] = [[0.5, 0.5 + near_dy], [0.5, 0.5 - near_dy]]   # only 2 close
    players[2, :3] = players[0, :3]
    ball[2] = np.nan                              # ball missing -> false
    cue = crowding_cue(players, ball,
                       length_m=CFG["pitch_length_m"], width_m=CFG["pitch_width_m"])
    assert cue.tolist() == [True, False, False]


def test_phase_weights_are_one_plus_alpha_times_cue_sum():
    c1 = np.array([True, False, True])
    c2 = np.array([False, False, True])
    c3 = np.array([False, False, False])
    assert phase_weights([c1, c2, c3], 2.0).tolist() == [3.0, 1.0, 5.0]
    assert phase_weights([c1, c2, c3], 3.0).tolist() == [4.0, 1.0, 7.0]


def test_map_event_rows_recovers_known_rows():
    # two periods, 10 frames each, 40000 µs per frame, ts resets each period
    ts = np.tile(np.arange(10, dtype=np.float64) * 40_000.0, 2)
    fid = np.concatenate([np.arange(100, 110), np.arange(200, 210)])
    period = np.array([1] * 10 + [2] * 10)
    # period offsets: p1 = 1000 s, p2 = 2000 s (epoch = offset + ts/1e6)
    anchors = [(1000.0 + 120_000 / 1e6, 103),      # p1 row 3
               (2000.0 + 120_000 / 1e6, 203)]      # p2 row 13
    events = [1000.0 + 320_000 / 1e6,              # p1 -> frame 8 -> row 8
              2000.0 + 0.0]                        # p2 -> frame 0 -> row 10
    rows = map_event_rows(ts, fid, period, anchors, events)
    assert rows.tolist() == [8, 10]
    # no anchors -> everything unmappable
    assert (map_event_rows(ts, fid, period, [], events) == -1).all()


def test_mask_ball_weighted_concentrates_and_uniform_is_untouched():
    ball = np.zeros((200, 2))
    ball[::7] = np.nan
    runs = np.array([2, 3, 4])
    weights = np.ones(200)
    weights[100:120] = 500.0
    # target the region's capacity (~20 frames): with weight 500 vs 1 the first draws
    # land there overwhelmingly; a larger target would exhaust the region and fall back
    # to the background tail by construction.
    target = 0.05

    masked_w, _ = mask_ball(ball, runs, np.random.default_rng(0),
                            target_fraction=target, weights=weights)
    removed_w = np.flatnonzero(np.isnan(masked_w).all(axis=1)
                               & ~np.isnan(ball).all(axis=1))
    inside_w = np.mean((removed_w >= 100) & (removed_w < 120))
    assert inside_w > 0.7

    masked_u, _ = mask_ball(ball, runs, np.random.default_rng(0), target_fraction=target)
    removed_u = np.flatnonzero(np.isnan(masked_u).all(axis=1)
                               & ~np.isnan(ball).all(axis=1))
    inside_u = np.mean((removed_u >= 100) & (removed_u < 120))
    assert inside_u < 0.4
    assert inside_w > inside_u + 0.3

    # weights=None reproduces the default path exactly (P5 masks)
    a, sa = mask_ball(ball, runs, np.random.default_rng(7), target_fraction=0.3)
    b, sb = mask_ball(ball, runs, np.random.default_rng(7), target_fraction=0.3,
                      weights=None)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert sa == sb
