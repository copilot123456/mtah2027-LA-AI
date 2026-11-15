"""P5 unit tests: grid integrity, label derivation, masking, plateau selection, LOMO, CI,
and the reproduction of the frozen selection from the committed matrices."""

import numpy as np
import pandas as pd
import pytest

from config import CFG
from tuning.simulate import empirical_run_lengths, mask_ball
from tuning.sweep import (
    GRID,
    PARAMS,
    _neighbour_means,
    _score,
    bootstrap_ci,
    flat_select,
    grid_points,
    load_union,
    lomo,
    smoke_grid,
)


def test_grid_has_243_unique_cells():
    """Grid has 243 unique cells (3 values per param)."""
    points = grid_points()                      # all grid cells
    assert len(points) == 243 == 3 ** len(PARAMS)   # expected cell count
    keys = {tuple(p[p_] for p_ in PARAMS) for p in points}   # unique param tuples
    assert len(keys) == 243                     # no duplicate cells


def test_smoke_grid_is_subset():
    """Smoke grid holds exactly six cells."""
    assert len(smoke_grid()) == 6               # smoke grid size


def test_derive_letters_rules():
    """Period, state and owner map to letters."""
    from tuning.idsse import derive_letters

    period = np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.int8)   # period, then outside
    state = np.array(["alive", "dead", "alive", "alive", "alive", "dead", "dead"], dtype=object)   # alive/dead mix
    owner = np.array(["HOM", "", "HOM", "", "AWY", "", ""], dtype=object)   # home/empty/away owners
    letters, counts = derive_letters(period, state, owner, "HOM", "AWY")   # derive labels here
    assert letters[0] == "H"          # alive, known home owner
    assert letters[1] == "D"          # dead beats owner
    assert letters[2] == "H"                   # alive, known owner
    assert letters[3] == "H"          # unknown owner, forward-filled from frame 2
    assert letters[4] == "A"                   # alive, away owner
    assert letters[6] == "X"          # outside periods
    assert counts["X"] == 1                    # one outside frame
    assert counts["owner_filled"] == 1         # one filled owner


def test_derive_letters_dead_priority_and_counts():
    """Dead state overrides a known owner everywhere."""
    from tuning.idsse import derive_letters

    period = np.ones(6, dtype=np.int8)         # all frames in period
    state = np.array(["dead"] * 6, dtype=object)   # every frame dead
    owner = np.array(["HOM"] * 6, dtype=object)    # home owner everywhere
    letters, counts = derive_letters(period, state, owner, "HOM", "AWY")   # dead must win
    assert set(letters) == {"D"}               # all labels dead
    assert counts["D"] == 6                    # six dead labels


def test_derive_letters_edge_cases():
    """NaN owners, a period with no owner information, labels outside periods."""
    from tuning.idsse import derive_letters

    period = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2, 0], dtype=np.int8)   # two periods, edges outside
    state = np.array(["alive", "alive", "alive", "dead", "alive", "alive",
                      "alive", "alive", "dead", "alive"], dtype=object)   # mostly alive, some dead
    owner = pd.Series(["HOM", np.nan, "AWY", "AWY", "AWY", np.nan,   # NaN owners included
                       np.nan, np.nan, np.nan, "HOM"])
    letters, counts = derive_letters(period, state, owner, "HOM", "AWY")   # derive edge labels
    assert letters[0] == "X" and letters[9] == "X", "outside the periods is X whatever the owner"   # X outside periods
    assert letters[1] == "A", "NaN owner at the period head is back-filled from the next frame"   # head back-fill
    assert letters[2] == "A"                   # alive, away owner
    assert letters[3] == "D", "dead beats the owner field"   # dead wins
    assert letters[4] == "A" and letters[5] == "A", "forward fill inside the period"   # fill inside period
    assert letters[6] == "D" and letters[7] == "D", \
        "fills copy the nearest labelled frame in the period, dead included; no owner at all => D"   # fill copies nearest
    assert letters[8] == "D"                   # dead state wins
    assert sum(counts[k] for k in ("H", "A", "D", "X")) == len(period)   # labels cover all frames
    assert counts["owner_filled"] == 4         # four filled owners


def test_empirical_runs_extraction():
    """True-blocks in a mask become run lengths."""
    mask = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)   # three true runs
    np.testing.assert_array_equal(empirical_run_lengths(mask), [2, 1, 3])   # run lengths extracted


def test_mask_ball_deterministic_and_surgical():
    """Same seed masks identically; other frames untouched."""
    rng1 = np.random.default_rng(42)           # seed for first mask
    rng2 = np.random.default_rng(42)           # same seed again
    ball = np.stack([np.linspace(0, 1, 5000), np.linspace(1, 0, 5000)], axis=-1)   # synthetic ball track
    runs = np.array([10, 50, 300, 600])        # allowed mask lengths
    m1, s1 = mask_ball(ball, runs, rng1, target_fraction=0.3)   # mask with first seed
    m2, s2 = mask_ball(ball, runs, rng2, target_fraction=0.3)   # mask with same seed
    np.testing.assert_array_equal(np.isnan(m1), np.isnan(m2))   # identical mask patterns
    assert s1["masked_fraction"] >= 0.29        # reaches the 0.30 target
    untouched = ~np.isnan(m1)                  # frames that kept data
    np.testing.assert_array_equal(m1[untouched], ball[untouched])   # unmasked data unchanged
    assert s1["n_runs"] > 0                     # some runs drawn


def test_mask_ball_never_touches_full_target_block_randomly():
    """Masked count comes from the empirical distribution, not arbitrary lengths."""
    rng = np.random.default_rng(1)             # fixed seed
    ball = np.zeros((10_000, 2))               # large blank track
    runs = np.array([100])                     # one run length only
    m, stats = mask_ball(ball, runs, rng, target_fraction=0.2)   # mask to 20 percent
    assert stats["masked_frames"] >= int(0.2 * 10_000)   # reaches target count
    assert stats["masked_frames"] % 100 == 0   # only multiples of the sole run length


def _toy_results(n_matches: int = 7) -> pd.DataFrame:
    """243-cell matrix: wide plateau around the grid centre, one far-corner spike."""
    center = {p: GRID[p][len(GRID[p]) // 2] for p in PARAMS}   # middle value per param
    rows = []                           # collect matrix rows
    for point in grid_points():
        l1 = sum(1 for p in PARAMS if point[p] != center[p])   # distance from centre
        mean = 0.54 if l1 <= 1 else 0.50      # plateau score nearby
        if point[PARAMS[0]] == GRID[PARAMS[0]][-1] and all(
                point[p] == GRID[p][0] for p in PARAMS[1:]):
            mean = 0.55   # isolated spike in the far corner
        for m in range(n_matches):             # seven match rows
            rows.append({**point, "scenario": "kinematic", "match_id": f"M{m}",   # kinematic row each
                         "spell_f1": mean + 0.001 * m, "macro_f1": 0.5})
        rows.append({**point, "scenario": "none", "match_id": "M0", "spell_f1": mean})   # one none row
    return pd.DataFrame(rows)                 # build the table


def test_flat_select_prefers_plateau_over_isolated_spike():
    """Plateau centre beats an isolated corner spike."""
    df = _toy_results()                       # toy matrix ready
    chosen, detail = flat_select(df, scenario="kinematic")   # run the selector
    center = {p: GRID[p][len(GRID[p]) // 2] for p in PARAMS}   # expected plateau centre
    assert chosen == center, "plateau centre must win over the corner spike"   # centre wins
    assert detail["selected"].sum() == 1      # exactly one pick


def test_lomo_returns_all_folds():
    """LOMO scores each leave-one-match-out fold."""
    df = _toy_results()                       # same toy matrix
    chosen, _ = flat_select(df, scenario="kinematic")   # cell under test
    table = lomo(df, chosen)                  # run leave-one-out
    assert len(table) == 7                      # one fold per match
    assert table["fold_score"].notna().all()  # every fold scored
    assert table["global_score"].notna().all()   # global scores too


def test_harness_scorer_matches_the_official_evaluate():
    """The sweep/post_checks scorer must agree with evaluate(): X inside a graded run is
    forward-filled for spellF1 (H X X H is one possession) and charged only in macroF1."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))   # reach official scorer
    import possession_metrics as pm

    n = 3_000                               # frames in toy match
    gt_letters = np.full(n, "X", dtype=object)   # start all abstained
    blocks = [(100, 700, "H"), (700, 1200, "A"), (1200, 1300, "D"), (1300, 2000, "H"),   # ground-truth blocks
              (2100, 2600, "A"), (2600, 2900, "D")]
    for start, stop, lab in blocks:           # paint each block
        gt_letters[start:stop] = lab          # label that span
    pred_letters = gt_letters.copy()          # prediction starts from it
    pred_letters[400:450] = "X"                       # abstention inside a possession
    pred_letters[1250:1280] = "X"                     # abstention inside a dead-ball run
    pred_letters[2100:2130] = "X"                     # abstention at a run head (back-filled)
    pred_letters[1500:1560] = "A"                     # one wrong stretch: macroF1 < 1, extra spell
    gt, pred = pm.label_codes(gt_letters), pm.label_codes(pred_letters)   # numeric score codes

    ours = _score(gt, pred)                   # harness scorer result
    official = pm.evaluate(gt, pred, bootstrap_n=0)   # official scorer result
    assert ours["spell_f1"] == pytest.approx(official["spell_f1_mean"], abs=1e-12)   # spellF1 must match
    assert ours["macro_f1"] == pytest.approx(official["macro_f1"], abs=1e-12)   # macroF1 must match
    # and the fill really matters here: scoring the raw X would differ
    runs = pm.valid_runs(gt)
    raw = pm.spell_scores(gt, pred, runs)["mean_f1"]   # score without fills
    assert raw != pytest.approx(ours["spell_f1"], abs=1e-6)   # fills change the score


def test_neighbour_means_use_the_swept_values():
    """Two cells that differ only in a notch outside the stage-1 GRID are neighbours."""
    base = {p: GRID[p][0] for p in PARAMS}   # first value each param
    cells = [{**base, "sigma_m": 1.0}, {**base, "sigma_m": 1.5}]   # neither value is in GRID
    means = pd.DataFrame([{**c, "mean": m} for c, m in zip(cells, (0.30, 0.20))])   # one score per cell
    means = means.set_index(PARAMS)["mean"]   # index by parameters
    worst = _neighbour_means(means)           # worst neighbour means
    assert worst.loc[tuple(cells[0][p] for p in PARAMS)] == pytest.approx(0.20)   # sees the lower neighbour
    assert worst.loc[tuple(cells[1][p] for p in PARAMS)] == pytest.approx(0.30)   # sees the higher neighbour


def test_final_selection_reproduces_frozen_config():
    """The committed stage-2+3 matrices + the pre-declared rule give exactly config.py's
    cell and the numbers quoted in dossier.md / report_final.md."""
    union = load_union()                      # load committed matrices
    assert union[PARAMS].drop_duplicates().shape[0] == 930   # expected union size
    chosen, detail = flat_select(union, scenario="none")   # re-apply the pinned plateau rule
    assert chosen == {p: float(CFG[p]) for p in PARAMS}   # exactly the shipped cell
    rows = union[union["scenario"] == "none"]   # keep none scenario
    for p, v in chosen.items():               # narrow to chosen cell
        rows = rows[rows[p] == v]             # filter one parameter
    assert len(rows) == 7                       # one row per match
    assert rows["spell_f1"].mean() == pytest.approx(0.2759, abs=5e-4)   # pinned mean score
    lo, hi = bootstrap_ci(rows["spell_f1"].to_numpy())   # bootstrap the interval
    assert (lo, hi) == pytest.approx((0.2578, 0.2932), abs=5e-4)   # pinned CI bounds
    gap = lomo(union, chosen, scenario="none")   # LOMO on selection
    assert (gap["fold_score"] - gap["global_score"]).mean() == pytest.approx(-0.0026, abs=5e-4)   # pinned fold gap


def test_bootstrap_ci_brackets_the_mean():
    """Bootstrap interval brackets the sample mean."""
    values = np.array([0.30, 0.32, 0.31, 0.33, 0.29, 0.32, 0.31])   # small sample values
    lo, hi = bootstrap_ci(values, n=500, seed=3)   # seeded bootstrap
    assert lo <= values.mean() <= hi          # mean inside interval
    assert 0.0 < hi - lo < 0.05                 # tight interval
