"""E7 unit tests: config pin, f_sp window maths, tagging-noise simulator, emission OFF
identity — all pure or synthetic; the byte-identity of the frozen submission with the
new wiring is enforced end-to-end by tests/test_submission.py."""

import numpy as np

from config import CFG, PROVENANCE
from src.emission import _set_piece_feature, build_emission


def test_config_e7_defaults_are_off_and_pinned():
    assert CFG["d_w_set_piece"] == 0.0          # OFF until the gate passes
    assert CFG["sp_pre_s"] == 2.0
    assert CFG["sp_post_s"] == 1.0
    for key in ("d_w_set_piece", "sp_pre_s", "sp_post_s"):
        assert key in PROVENANCE


def test_set_piece_feature_window_and_clipping():
    n = 300
    cfg = {"sp_pre_s": 2.0, "sp_post_s": 1.0, "fps": 25}
    f = _set_piece_feature(np.array([100]), n, cfg)
    assert f[49] == 0.0 and f[50] == 1.0        # start - 2 s = 100 - 50
    assert f[125] == 1.0 and f[126] == 0.0      # start + 1 s = 100 + 25 (inclusive)
    assert f.sum() == 76.0
    # clipping at both edges
    edge = _set_piece_feature(np.array([5, n - 3]), n, cfg)
    assert edge[0] == 1.0 and edge[n - 1] == 1.0
    # absent input -> all off
    assert _set_piece_feature(None, n, cfg).sum() == 0.0
    assert _set_piece_feature(np.array([], dtype=np.int64), n, cfg).sum() == 0.0


def test_noisy_setpiece_is_seeded_deterministic_and_counted():
    from tuning.e7 import noisy_setpiece

    raw = np.arange(0, 1000, 10)               # 100 events
    a = noisy_setpiece(raw, 5000, "J03WPY")
    b = noisy_setpiece(raw, 5000, "J03WPY")
    assert np.array_equal(a, b)                # same seed -> identical
    assert np.array_equal(a, np.sort(a))       # sorted
    assert a.min() >= 0 and a.max() < 5000     # clipped to the match
    # stream per §7: jitter draw, then the drop draw, then spurious
    rng = np.random.default_rng(
        20261107 + 700 + sum(ord(c) for c in "J03WPY"))
    _ = rng.integers(-25, 26, size=100)
    kept = int((rng.random(100) >= 0.25).sum())
    assert len(a) == kept + int(round(0.10 * 100))
    # a different match id gives a different stream (documented per-match seeding)
    c = noisy_setpiece(raw, 5000, "J03WN1")
    assert not np.array_equal(a[:5], c[:5])


def test_emission_off_is_identity_and_on_raises_p_d_in_window():
    from test_emission import _slots_on_grid, _track

    n = 120
    ball = np.full((n, 2), np.nan)
    ball[:, 0], ball[:, 1] = 0.5, 0.5
    track = _track(ball)
    players = np.tile(_slots_on_grid()[:1], (n, 1, 1))   # one frame tiled to n
    ps = np.full((n, 46), 3.0)
    starts = np.array([60])
    cfg_off = {**CFG, "d_w_set_piece": 0.0}
    cfg_on = {**CFG, "d_w_set_piece": 1.0}

    em_none = build_emission(track, players, ps, np.arange(46) % 2, cfg_off)
    em_off = build_emission(track, players, ps, np.arange(46) % 2, cfg_off,
                            setpiece_starts=starts)
    assert np.array_equal(em_none.p_d, em_off.p_d)     # OFF: bit-identical

    em_on = build_emission(track, players, ps, np.arange(46) % 2, cfg_on,
                           setpiece_starts=starts)
    idx = np.arange(n)
    inside = (idx >= 60 - 50) & (idx <= 60 + 25)       # window [10, 85]
    assert (em_on.p_d[inside] > em_off.p_d[inside]).all()
    assert np.array_equal(em_on.p_d[~inside], em_off.p_d[~inside])


def test_arrays_carry_setpiece_starts():
    from src.data import N_FRAMES, load_arrays

    arrays = load_arrays()
    starts = arrays.setpiece_starts
    assert starts.dtype.kind == "i"
    assert len(starts) == 59                      # dossier §7: SET PIECES instances
    assert np.all(np.diff(starts) >= 0)
    assert starts.min() >= 0 and starts.max() < N_FRAMES


def test_e7_gate_record_reproduces_the_reject():
    """The committed gate record pins the dossier §7 verdict (primary below the bar)."""
    import pandas as pd

    df = pd.read_csv("tuning/results/e7_gate.csv")
    prim = df[(df["kind"] == "phase_a2") & (df["label"] == "aligned")]
    assert len(prim) == 7
    delta = (prim["on"] - prim["off_committed"]).mean()
    assert delta < 0.02                       # the pinned adoption bar
    assert (df["off_drift"].abs() < 1e-12).all()   # OFF side reproduces stage-4 exactly
