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

ROOT = Path(__file__).resolve().parents[1]


def test_g7_has_49_unique_cells_plus_the_shipped_one():
    """G7 grid: 49 swept cells plus the shipped cell."""
    cells = g7_cells()
    assert len(cells) == 50 and cells[-1] == AS_SHIPPED
    keys = {tuple(c[p] for p in D_PARAMS) for c in cells}
    assert len(keys) == 50, "a zero weight must be counted once"
    assert gate5.key(gate5.OFF) in keys, "the U1 OFF cell is part of the grid"
    zero_la = [c for c in cells[:-1] if c["d_w_low_activity"] == 0.0]
    assert {c["d_activity_thr_ms"] for c in zero_la} == {1.0}


def _letters(n=40_000, seed=3):
    """Alternating alive / dead blocks with X padding, like an IDSSE match."""
    rng = np.random.default_rng(seed)
    out = np.full(n, "X", dtype=object)
    i = 1_000
    while i < n - 1_000:
        alive = int(rng.integers(200, 900))
        dead = int(rng.integers(60, 500))
        out[i:i + alive] = "H" if rng.random() < 0.5 else "A"
        out[i + alive:i + alive + dead] = "D"
        i += alive + dead
    out[n - 1_000:] = "X"
    return out


def test_dead_cover_mask_couples_ball_loss_to_dead_ball():
    """q shifts a fixed masking share onto dead frames."""
    letters = _letters()
    ball = np.full((len(letters), 2), 0.5)
    runs = np.array([40, 120, 260, 400])
    in_match, dead = letters != "X", letters == "D"
    shares = {}
    for q in (0.0, 0.5, 0.9):
        masked, stats = mask_ball_dead_cover(ball, letters, runs, np.random.default_rng(1), q)
        miss = np.isnan(masked).any(axis=1)
        assert abs(miss[in_match].mean() - 0.431) < 0.03, "target share reached"
        assert stats["masked_fraction"] == pytest.approx(miss[in_match].mean())
        shares[q] = miss[in_match & dead].mean()
        assert np.isfinite(ball).all(), "input must not be mutated"
    assert shares[0.0] < shares[0.5] < shares[0.9]
    assert shares[0.9] > 0.8

    again, _ = mask_ball_dead_cover(ball, letters, runs, np.random.default_rng(1), 0.9)
    np.testing.assert_array_equal(np.isnan(again), np.isnan(masked))   # same seed, same mask


def test_baseline_port_matches_the_starter_rules():
    """Starter nearest-code and smoothing rules port exactly."""
    ball = np.array([[0.10, 0.5], [np.nan, np.nan], [0.90, 0.5], [0.5, 0.5]])
    players = np.zeros((4, 2, 2))
    players[:, 0] = [0.12, 0.5]          # home slot near the left
    players[:, 1] = [0.88, 0.5]          # away slot near the right
    players[3, 0] = np.nan               # only the away slot visible on the last frame
    codes = nearest_codes(ball, players, np.array([0, 1], dtype=np.int8))
    assert codes.tolist() == [0, -1, 1, 1]
    assert smooth(codes, 0).tolist() == ["H", "D", "A", "A"]
    assert smooth(np.array([0, 1, -1, -1, -1]), 1).tolist() == ["H", "H", "A", "D", "D"], \
        "ties go home; D only where nobody is seen in the window"


def _table(scores: dict) -> pd.DataFrame:
    """Synthetic stage-5 table: {cell tuple: aligned spellF1 on every match}."""
    rows = []
    for cell, value in scores.items():
        for m in range(7):
            rows.append({**dict(zip(D_PARAMS, cell)), "match_id": f"M{m}",
                         "spell_f1_orig": value, "spell_f1_aligned": value + 0.001 * m,
                         "macro_f1_orig": 0.5, "macro_f1_aligned": 0.5, "pred_d_share": 0.3})
    return pd.DataFrame(rows).set_index(D_PARAMS + ["match_id"]).sort_index()


def test_u1_and_r1_rules_on_synthetic_tables():
    """U1 adopts wins; R1 tie-breaks and rejects weak cells."""
    ship, off = gate5.key(AS_SHIPPED), gate5.key(gate5.OFF)
    a, b = (1.0, 3.0, 1.0, 1.0), (2.0, 3.0, 2.0, 1.5)
    base = {ship: 0.30, off: 0.35, a: 0.38, b: 0.382}
    tables = {k: _table(base) for k in KINDS}
    u1 = gate5.u1_verdict(tables)
    assert u1["adopt"] and u1["primary"] == pytest.approx(0.05) and u1["wins"] == 7

    read = gate5.r1(tables, gate5.OFF)
    # a and b are within the 0.005 tie band; a changes less weight, so the tie goes to a
    assert gate5.key(read["candidate"]["cell"]) == a
    assert read["adopt"] and read["candidate"]["g"] == pytest.approx(0.03)

    # one regime where the candidate loses: maximin drops below the bar -> reject
    tables["uniform"] = _table({**base, a: 0.34, b: 0.34})
    assert not gate5.r1(tables, gate5.OFF)["adopt"]
    # ...but the coupled-regime reading ignores `uniform`
    assert gate5.r2(tables, gate5.OFF)["adopt"]


def test_tie_break_counts_only_effective_parameters():
    """Only weight changes count; thresholds ride along."""
    ref = gate5.OFF                                     # long-missing 0.5/10 s, low-activity off
    on = {**ref, "d_w_low_activity": 0.5, "d_activity_thr_ms": 1.5}
    assert gate5.n_changed(on, ref) == (1, 0.5), "the threshold rides along with a new weight"
    moved = {**ref, "d_missing_run_s": 3.0}
    assert gate5.n_changed(moved, ref) == (1, 0.0)


def test_committed_stage5_record_reproduces_the_verdicts():
    """Pins the dossier §7 P7 read to the committed matrices."""
    if not (ROOT / "tuning" / "results" / "stage5_uniform.csv").exists():
        pytest.skip("stage-5 matrices not generated yet")
    _, summary = gate5.build_report()
    record = ROOT / "tuning" / "results" / "stage5_verdicts.csv"
    expected = pd.read_csv(record).set_index("gate")["verdict"].to_dict()
    assert expected["U1"] == ("ADOPT" if summary["u1"]["adopt"] else "KEEP")
    assert expected["D1_R1"] == ("PASS" if summary["r1"]["adopt"] else "REJECT")
    assert expected["D1_R2"] == ("PASS" if summary["r2"]["adopt"] else "REJECT")
