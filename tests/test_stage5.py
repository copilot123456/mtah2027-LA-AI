"""Stage 5 (dossier §7 P7, 2026-11-14): grid, dead-cover masks, baseline port, gate logic.

Pure-function tests on synthetic inputs; the committed-matrix test at the bottom pins the
recorded verdicts and is skipped until the matrices exist.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tuning import gate5
from tuning.simulate import mask_ball_dead_cover
from tuning.stage5 import AS_SHIPPED, D_PARAMS, KINDS, g7_cells, nearest_codes, smooth

ROOT = Path(__file__).resolve().parents[1]   # repo root path


def test_g7_has_49_unique_cells_plus_the_shipped_one():
    """G7 grid: 49 swept cells plus the shipped cell."""
    cells = g7_cells()                        # build the G7 grid
    assert len(cells) == 50 and cells[-1] == AS_SHIPPED   # 49 plus shipped
    keys = {tuple(c[p] for p in D_PARAMS) for c in cells}   # one key per cell
    assert len(keys) == 50, "a zero weight must be counted once"   # no duplicate cells
    assert gate5.key(gate5.OFF) in keys, "the U1 OFF cell is part of the grid"   # OFF cell included
    zero_la = [c for c in cells[:-1] if c["d_w_low_activity"] == 0.0]   # cells with weight zero
    assert {c["d_activity_thr_ms"] for c in zero_la} == {1.0}   # only at 1 ms


def _letters(n=40_000, seed=3):
    """Alternating alive / dead blocks with X padding, like an IDSSE match."""
    rng = np.random.default_rng(seed)         # reproducible draws
    out = np.full(n, "X", dtype=object)       # X padding everywhere
    i = 1_000                             # start after padding
    while i < n - 1_000:                      # fill middle region
        alive = int(rng.integers(200, 900))   # random alive length
        dead = int(rng.integers(60, 500))     # random dead length
        out[i:i + alive] = "H" if rng.random() < 0.5 else "A"   # random home or away
        out[i + alive:i + alive + dead] = "D"   # dead block after alive
        i += alive + dead                     # advance to next block
    out[n - 1_000:] = "X"                 # trailing X padding
    return out                                # hand back letters


def test_dead_cover_mask_couples_ball_loss_to_dead_ball():
    """q shifts a fixed masking share onto dead frames."""
    letters = _letters()                      # synthetic match letters
    ball = np.full((len(letters), 2), 0.5)    # ball at centre always
    runs = np.array([40, 120, 260, 400])      # allowed run lengths
    in_match, dead = letters != "X", letters == "D"   # masks for in-play, dead
    shares = {}                               # dead shares by q
    for q in (0.0, 0.5, 0.9):                 # three coupling strengths
        masked, stats = mask_ball_dead_cover(ball, letters, runs, np.random.default_rng(1), q)   # mask at this q
        miss = np.isnan(masked).any(axis=1)   # frames now missing
        assert abs(miss[in_match].mean() - 0.431) < 0.03, "target share reached"   # hits fixed share
        assert stats["masked_fraction"] == pytest.approx(miss[in_match].mean())   # stats agree here
        shares[q] = miss[in_match & dead].mean()   # record dead share
        assert np.isfinite(ball).all(), "input must not be mutated"   # input untouched
    assert shares[0.0] < shares[0.5] < shares[0.9]   # share rises with q
    assert shares[0.9] > 0.8                  # high q dominates

    again, _ = mask_ball_dead_cover(ball, letters, runs, np.random.default_rng(1), 0.9)   # rerun same seed
    np.testing.assert_array_equal(np.isnan(again), np.isnan(masked))   # same seed, same mask


def test_baseline_port_matches_the_starter_rules():
    """Starter nearest-code and smoothing rules port exactly."""
    ball = np.array([[0.10, 0.5], [np.nan, np.nan], [0.90, 0.5], [0.5, 0.5]])   # four frames of ball
    players = np.zeros((4, 2, 2))             # empty player slots
    players[:, 0] = [0.12, 0.5]          # home slot near the left
    players[:, 1] = [0.88, 0.5]          # away slot near the right
    players[3, 0] = np.nan               # only the away slot visible on the last frame
    codes = nearest_codes(ball, players, np.array([0, 1], dtype=np.int8))   # nearest slot per frame
    assert codes.tolist() == [0, -1, 1, 1]    # home, none, away, away
    assert smooth(codes, 0).tolist() == ["H", "D", "A", "A"]   # no smoothing applied
    assert smooth(np.array([0, 1, -1, -1, -1]), 1).tolist() == ["H", "H", "A", "D", "D"], \
        "ties go home; D only where nobody is seen in the window"   # window ties resolved


def _table(scores: dict) -> pd.DataFrame:
    """Synthetic stage-5 table: {cell tuple: aligned spellF1 on every match}."""
    rows = []                              # collect table rows
    for cell, value in scores.items():      # one entry per cell
        for m in range(7):                  # seven matches each
            rows.append({**dict(zip(D_PARAMS, cell)), "match_id": f"M{m}",   # one row per match
                         "spell_f1_orig": value, "spell_f1_aligned": value + 0.001 * m,
                         "macro_f1_orig": 0.5, "macro_f1_aligned": 0.5, "pred_d_share": 0.3})
    return pd.DataFrame(rows).set_index(D_PARAMS + ["match_id"]).sort_index()   # indexed sorted table


def test_u1_and_r1_rules_on_synthetic_tables():
    """U1 adopts wins; R1 tie-breaks and rejects weak cells."""
    ship, off = gate5.key(AS_SHIPPED), gate5.key(gate5.OFF)   # cell keys for lookup
    a, b = (1.0, 3.0, 1.0, 1.0), (2.0, 3.0, 2.0, 1.5)   # two candidate cells
    base = {ship: 0.30, off: 0.35, a: 0.38, b: 0.382}   # scores per cell
    tables = {k: _table(base) for k in KINDS}   # same scores all regimes
    u1 = gate5.u1_verdict(tables)             # run U1 gate
    assert u1["adopt"] and u1["primary"] == pytest.approx(0.05) and u1["wins"] == 7   # clean sweep adopts

    read = gate5.r1(tables, gate5.OFF)        # run R1 gate
    # a and b are within the 0.005 tie band; a changes less weight, so the tie goes to a
    assert gate5.key(read["candidate"]["cell"]) == a
    assert read["adopt"] and read["candidate"]["g"] == pytest.approx(0.03)   # adopts with gain

    # one regime where the candidate loses: maximin drops below the bar -> reject
    tables["uniform"] = _table({**base, a: 0.34, b: 0.34})
    assert not gate5.r1(tables, gate5.OFF)["adopt"]   # weak cell rejected
    # ...but the coupled-regime reading ignores `uniform`
    assert gate5.r2(tables, gate5.OFF)["adopt"]


def test_tie_break_counts_only_effective_parameters():
    """Only weight changes count; thresholds ride along."""
    ref = gate5.OFF                                     # long-missing 0.5/10 s, low-activity off
    on = {**ref, "d_w_low_activity": 0.5, "d_activity_thr_ms": 1.5}   # add weight plus threshold
    assert gate5.n_changed(on, ref) == (1, 0.5), "the threshold rides along with a new weight"   # only weight counts
    moved = {**ref, "d_missing_run_s": 3.0}   # move a timing param
    assert gate5.n_changed(moved, ref) == (1, 0.0)   # no weight change


def test_committed_stage5_record_reproduces_the_verdicts():
    """Pins the dossier §7 P7 read to the committed matrices."""
    if not (ROOT / "tuning" / "results" / "stage5_uniform.csv").exists():   # skip if not generated
        pytest.skip("stage-5 matrices not generated yet")   # wait for matrices
    _, summary = gate5.build_report()         # recompute gate summary
    record = ROOT / "tuning" / "results" / "stage5_verdicts.csv"   # committed verdict file
    expected = pd.read_csv(record).set_index("gate")["verdict"].to_dict()   # recorded verdicts dict
    assert expected["U1"] == ("ADOPT" if summary["u1"]["adopt"] else "KEEP")   # U1 verdict matches
    assert expected["D1_R1"] == ("PASS" if summary["r1"]["adopt"] else "REJECT")   # R1 verdict matches
    assert expected["D1_R2"] == ("PASS" if summary["r2"]["adopt"] else "REJECT")   # R2 verdict matches
