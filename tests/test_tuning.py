"""P5 unit tests: grid integrity, label derivation, masking, plateau selection, LOMO, CI."""

import numpy as np
import pandas as pd
import pytest

from tuning.simulate import empirical_run_lengths, mask_ball
from tuning.sweep import (
    GRID,
    PARAMS,
    bootstrap_ci,
    flat_select,
    grid_points,
    lomo,
    smoke_grid,
)


def test_grid_has_243_unique_cells():
    points = grid_points()
    assert len(points) == 243 == 3 ** len(PARAMS)
    keys = {tuple(p[p_] for p_ in PARAMS) for p in points}
    assert len(keys) == 243


def test_smoke_grid_is_subset():
    assert len(smoke_grid()) == 6


def test_derive_letters_rules():
    from tuning.idsse import derive_letters

    period = np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.int8)
    state = np.array(["alive", "dead", "alive", "alive", "alive", "dead", "dead"], dtype=object)
    owner = np.array(["HOM", "", "HOM", "", "AWY", "", ""], dtype=object)
    letters, counts = derive_letters(period, state, owner, "HOM", "AWY")
    assert letters[0] == "H"          # alive, known home owner
    assert letters[1] == "D"          # dead beats owner
    assert letters[2] == "H"
    assert letters[3] == "H"          # unknown owner, forward-filled from frame 2
    assert letters[4] == "A"
    assert letters[6] == "X"          # outside periods
    assert counts["X"] == 1
    assert counts["owner_filled"] == 1


def test_derive_letters_dead_priority_and_counts():
    from tuning.idsse import derive_letters

    period = np.ones(6, dtype=np.int8)
    state = np.array(["dead"] * 6, dtype=object)
    owner = np.array(["HOM"] * 6, dtype=object)
    letters, counts = derive_letters(period, state, owner, "HOM", "AWY")
    assert set(letters) == {"D"}
    assert counts["D"] == 6


def test_empirical_runs_extraction():
    mask = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
    np.testing.assert_array_equal(empirical_run_lengths(mask), [2, 1, 3])


def test_mask_ball_deterministic_and_surgical():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    ball = np.stack([np.linspace(0, 1, 5000), np.linspace(1, 0, 5000)], axis=-1)
    runs = np.array([10, 50, 300, 600])
    m1, s1 = mask_ball(ball, runs, rng1, target_fraction=0.3)
    m2, s2 = mask_ball(ball, runs, rng2, target_fraction=0.3)
    np.testing.assert_array_equal(np.isnan(m1), np.isnan(m2))
    assert s1["masked_fraction"] >= 0.29
    untouched = ~np.isnan(m1)
    np.testing.assert_array_equal(m1[untouched], ball[untouched])
    assert s1["n_runs"] > 0


def test_mask_ball_never_touches_full_target_block_randomly():
    """Masked count comes from the empirical distribution, not arbitrary lengths."""
    rng = np.random.default_rng(1)
    ball = np.zeros((10_000, 2))
    runs = np.array([100])
    m, stats = mask_ball(ball, runs, rng, target_fraction=0.2)
    assert stats["masked_frames"] >= int(0.2 * 10_000)
    assert stats["masked_frames"] % 100 == 0   # only multiples of the sole run length


def _toy_results(n_matches: int = 7) -> pd.DataFrame:
    """243-cell matrix: wide plateau around the grid centre, one far-corner spike."""
    center = {p: GRID[p][len(GRID[p]) // 2] for p in PARAMS}
    rows = []
    for point in grid_points():
        l1 = sum(1 for p in PARAMS if point[p] != center[p])
        mean = 0.54 if l1 <= 1 else 0.50
        if point[PARAMS[0]] == GRID[PARAMS[0]][-1] and all(
                point[p] == GRID[p][0] for p in PARAMS[1:]):
            mean = 0.55   # isolated spike in the far corner
        for m in range(n_matches):
            rows.append({**point, "scenario": "kinematic", "match_id": f"M{m}",
                         "spell_f1": mean + 0.001 * m, "macro_f1": 0.5})
        rows.append({**point, "scenario": "none", "match_id": "M0", "spell_f1": mean})
    return pd.DataFrame(rows)


def test_flat_select_prefers_plateau_over_isolated_spike():
    df = _toy_results()
    chosen, detail = flat_select(df, scenario="kinematic")
    center = {p: GRID[p][len(GRID[p]) // 2] for p in PARAMS}
    assert chosen == center, "plateau centre must win over the corner spike"
    assert detail["selected"].sum() == 1


def test_lomo_returns_all_folds():
    df = _toy_results()
    chosen, _ = flat_select(df, scenario="kinematic")
    table = lomo(df, chosen)
    assert len(table) == 7
    assert table["fold_score"].notna().all()
    assert table["global_score"].notna().all()


def test_bootstrap_ci_brackets_the_mean():
    values = np.array([0.30, 0.32, 0.31, 0.33, 0.29, 0.32, 0.31])
    lo, hi = bootstrap_ci(values, n=500, seed=3)
    assert lo <= values.mean() <= hi
    assert 0.0 < hi - lo < 0.05
