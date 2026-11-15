"""Speed units (dossier §7 U0, 2026-11-14).

The IDSSE provider column is km/h, the pipeline's `d_activity_thr_ms` is m/s and the
provided match reports m/s. Until the fix the external bed ran the low-activity feature
on km/h, i.e. effectively off. These tests pin the self-check and both loaders.
"""

import numpy as np
import pytest

from tuning.idsse import CACHE_DIR, MATCH_IDS, speed_ratio, speed_unit_factor


def _walkers(n=4000, k=22, speed_ms=2.0, seed=0):
    """k players moving in straight lines at `speed_ms`, normalized coordinates."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, k)
    step = speed_ms / 25.0
    t = np.arange(n)[:, None]
    x = 0.5 + (t * step * np.cos(angle)[None, :]) / 105.3 * 0.01
    y = 0.5 + (t * step * np.sin(angle)[None, :]) / 68.0 * 0.01
    players = np.stack([x, y], axis=-1)
    true_ms = np.full((n, k), speed_ms * 0.01)
    return players, true_ms


def test_factor_is_one_for_metres_per_second():
    """Metre-per-second data yields factor 1."""
    players, true_ms = _walkers()
    assert speed_ratio(players, true_ms) == pytest.approx(1.0, rel=1e-6)
    assert speed_unit_factor(players, true_ms) == 1.0


def test_factor_is_3_6_for_kilometres_per_hour():
    """Kilometre-per-hour data is detected as 3.6x."""
    players, true_ms = _walkers()
    assert speed_unit_factor(players, true_ms * 3.6) == pytest.approx(3.6)


def test_ambiguous_units_raise():
    """Neither m/s nor km/h matches: the check raises."""
    players, true_ms = _walkers()
    with pytest.raises(ValueError, match="neither m/s nor km/h"):
        speed_unit_factor(players, true_ms * 2.0)


def test_missing_slots_do_not_break_the_ratio():
    """Tracker NaNs do not break unit detection."""
    players, true_ms = _walkers()
    players = players.copy()
    players[::7, :5] = np.nan                       # tracker dropouts
    assert speed_unit_factor(players, true_ms) == 1.0


def test_provided_match_reports_metres_per_second(arrays):
    """Assertion only — no value from the provided match is printed or stored."""
    graded = arrays.graded
    assert speed_unit_factor(arrays.players[graded], arrays.players_s[graded]) == 1.0


def test_idsse_loader_returns_metres_per_second():
    """The external IDSSE loader returns m/s."""
    from tuning.idsse import load_match

    match_id = MATCH_IDS[0]
    if not (CACHE_DIR / f"idsse_{match_id}.pkl").exists():
        pytest.skip("IDSSE cache not present (external data is optional for the suite)")
    case = load_match(match_id)
    assert case.info["speed_units"] == "m/s"
    assert speed_unit_factor(case.players, case.players_s) == 1.0
    # live-play players only
    alive = np.isin(case.letters, ("H", "A"))
    counts = np.sum(np.isfinite(case.players_s), axis=1)
    # mean speed across visible players
    mean_speed = np.nansum(case.players_s, axis=1) / np.maximum(counts, 1)
    # mean player speed in live play is a couple of m/s, never the ~8 of a km/h column
    assert 1.5 < float(np.median(mean_speed[alive])) < 3.5
