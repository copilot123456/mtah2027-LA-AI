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

FROZEN = {"switch_pen": 2.0, "dead_pen": 5.0, "sigma_m": 1.5,   # frozen config values
          "min_dwell_s": 2.0, "switch_support_s": 1.5}


def test_grid6_is_144_and_contains_the_frozen_cell():
    """G6 grid: 144 cells, frozen cell included."""
    points = grid6_points()                   # enumerate G6 cells
    assert set(GRID6) == set(PARAMS)           # same parameter names
    assert len(points) == 144 == 3 * 3 * 2 * 4 * 2   # expected cell count
    assert FROZEN in points                      # frozen cell is swept
    assert KINDS == ("uniform", "phase_a2", "phase_a3", "phase_a2_fill")   # four mask kinds


def test_possession_change_cue_exact_and_dilated():
    """Cue flags exact changes; wide radius covers them all."""
    letters = np.array(list("HHDAAXA"), dtype=object)   # letter pattern to scan
    raw = possession_change_cue(letters, half_s=0.0, fps=1.0)   # exact, no dilation
    assert raw.tolist() == [True, False, False, True, False, False, True]   # flags at changes
    wide = possession_change_cue(letters, half_s=2.0, fps=1.0)   # radius 2
    # changes at 0, 3, 6 with radius 2 overlap: the whole span is covered
    assert wide.all()


def test_event_cue_window():
    """Window spans half-width; out-of-range events ignored."""
    cue = event_cue(np.array([10]), 20, half_s=2.0, fps=1.0)   # one event at frame 10
    expected = np.zeros(20, dtype=bool)       # all-off baseline
    expected[8:13] = True        # 10 ± 2 frames, inclusive
    assert cue.tolist() == expected.tolist()  # window matches exactly
    assert not event_cue(np.array([-5, 99]), 20, half_s=2.0, fps=1.0).any()   # bad events ignored


def test_crowding_cue_ball_anchor_and_count():
    """Needs three nearby players and a visible ball."""
    # 3 frames, 4 slots; centre of pitch at (0.5, 0.5)
    players = np.full((3, 4, 2), np.nan)
    ball = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])   # ball at pitch centre
    near_dy = 2.0 / CFG["pitch_width_m"]          # 2 m north of centre
    players[0, :3] = [[0.5, 0.5 + near_dy], [0.5, 0.5 - near_dy], [0.5, 0.5]]   # three players near ball
    players[0, 3] = [0.2, 0.2]                    # far slot: irrelevant
    players[1, :2] = [[0.5, 0.5 + near_dy], [0.5, 0.5 - near_dy]]   # only 2 close
    players[2, :3] = players[0, :3]           # repeat crowded frame
    ball[2] = np.nan                              # ball missing -> false
    cue = crowding_cue(players, ball,   # evaluate the cue
                       length_m=CFG["pitch_length_m"], width_m=CFG["pitch_width_m"])
    assert cue.tolist() == [True, False, False]   # dense, sparse, no ball


def test_phase_weights_are_one_plus_alpha_times_cue_sum():
    """Weights are 1 + alpha × active cues."""
    c1 = np.array([True, False, True])        # two cues on
    c2 = np.array([False, False, True])       # one cue on
    c3 = np.array([False, False, False])      # no cues on
    assert phase_weights([c1, c2, c3], 2.0).tolist() == [3.0, 1.0, 5.0]   # one plus alpha sum
    assert phase_weights([c1, c2, c3], 3.0).tolist() == [4.0, 1.0, 7.0]   # scales with alpha


def test_map_event_rows_recovers_known_rows():
    """Anchors map events to rows; no anchors give -1."""
    # two periods, 10 frames each, 40000 µs per frame, ts resets each period
    ts = np.tile(np.arange(10, dtype=np.float64) * 40_000.0, 2)
    fid = np.concatenate([np.arange(100, 110), np.arange(200, 210)])   # frame ids, two periods
    period = np.array([1] * 10 + [2] * 10)    # period labels for frames
    # period offsets: p1 = 1000 s, p2 = 2000 s (epoch = offset + ts/1e6)
    anchors = [(1000.0 + 120_000 / 1e6, 103),      # p1 row 3
               (2000.0 + 120_000 / 1e6, 203)]      # p2 row 13
    events = [1000.0 + 320_000 / 1e6,              # p1 -> frame 8 -> row 8
              2000.0 + 0.0]                        # p2 -> frame 0 -> row 10
    rows = map_event_rows(ts, fid, period, anchors, events)   # map events to rows
    assert rows.tolist() == [8, 10]           # rows found correctly
    # no anchors -> everything unmappable
    assert (map_event_rows(ts, fid, period, [], events) == -1).all()


def test_mask_ball_weighted_concentrates_and_uniform_is_untouched():
    """Weights focus masking; default path stays identical."""
    ball = np.zeros((200, 2))                 # blank track
    ball[::7] = np.nan                        # sprinkle missing frames
    runs = np.array([2, 3, 4])                # allowed run lengths
    weights = np.ones(200)                    # flat weights everywhere
    weights[100:120] = 500.0                  # boost one region
    # target the region's capacity (~20 frames): with weight 500 vs 1 the first draws
    # land there overwhelmingly; a larger target would exhaust the region and fall back
    # to the background tail by construction.
    target = 0.05

    masked_w, _ = mask_ball(ball, runs, np.random.default_rng(0),   # mask with weights
                            target_fraction=target, weights=weights)
    removed_w = np.flatnonzero(np.isnan(masked_w).all(axis=1)   # newly masked frames
                               & ~np.isnan(ball).all(axis=1))
    inside_w = np.mean((removed_w >= 100) & (removed_w < 120))   # share inside boosted region
    assert inside_w > 0.7                     # weights concentrate here

    masked_u, _ = mask_ball(ball, runs, np.random.default_rng(0), target_fraction=target)   # mask without weights
    removed_u = np.flatnonzero(np.isnan(masked_u).all(axis=1)   # unweighted masked frames
                               & ~np.isnan(ball).all(axis=1))
    inside_u = np.mean((removed_u >= 100) & (removed_u < 120))   # unweighted inside share
    assert inside_u < 0.4                     # near background rate
    assert inside_w > inside_u + 0.3          # weighting clearly wins

    # weights=None reproduces the default path exactly (P5 masks)
    a, sa = mask_ball(ball, runs, np.random.default_rng(7), target_fraction=0.3)
    b, sb = mask_ball(ball, runs, np.random.default_rng(7), target_fraction=0.3,   # explicit None weights
                      weights=None)
    assert np.array_equal(np.isnan(a), np.isnan(b))   # identical masks
    assert sa == sb                           # identical stats too


def test_gate1_read_reproduces_the_recorded_verdict():
    """The committed stage-4 matrices reproduce the dossier §7 Gate-1 verdict."""
    from tuning.gate1 import gate1_read

    r = gate1_read()                          # read committed record
    assert r["verdict"] == "REJECT-KEEP"        # recorded verdict, pinned
    assert r["chosen"] == {"switch_pen": 1.0, "dead_pen": 12.0, "sigma_m": 1.5,   # pinned chosen cell
                           "min_dwell_s": 2.0, "switch_support_s": 1.0}
    assert r["delta"] < 0.01                  # under the tie band
    assert r["n_pos"] == 3                    # three paired wins
    assert abs(r["twins"]["fill"][0]) < 1e-12 and abs(r["twins"]["fill"][1]) < 1e-12   # fill twin is identical
    assert r["chosen_a3"] == r["chosen"]      # alpha-3 twin agrees
