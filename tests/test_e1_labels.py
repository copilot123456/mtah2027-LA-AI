"""E1 challenge-definition labels (pre-declared: dossier §7 P6, 2026-11-09).

Unit tests of `derive_letters_controlled` on synthetic sequences: absorption,
restart protection, chaining, the strict T boundary, and dead/X invariance.
"""

import numpy as np

from tuning.idsse import derive_letters_controlled

FPS = 25


def _seq(*spec):
    """Build a frame array from (label, seconds) pairs."""
    out = []
    for lab, secs in spec:
        out.extend([lab] * int(secs * FPS))
    return np.array(out, dtype=object)


def test_short_adjacent_run_is_absorbed_by_the_preceding_team():
    original = _seq(("H", 10), ("A", 2), ("D", 5))
    out = derive_letters_controlled(original, FPS, hold_s=3.0)
    assert (out[: 10 * FPS] == "H").all()
    assert (out[10 * FPS:12 * FPS] == "H").all(), "2 s control folds into the last team"
    assert (out[12 * FPS:] == "D").all()
    assert (original[10 * FPS:12 * FPS] == "A").all(), "input must not be mutated"


def test_run_at_or_above_t_is_a_genuine_possession_change():
    original = _seq(("H", 10), ("A", 3), ("H", 10))   # exactly T = 3 s
    out = derive_letters_controlled(original, FPS, hold_s=3.0)
    assert (out[10 * FPS:13 * FPS] == "A").all(), "shorter than T is strict: ==T is kept"


def test_first_alive_run_after_a_stoppage_is_never_absorbed():
    original = _seq(("H", 10), ("D", 6), ("A", 1), ("H", 10))
    out = derive_letters_controlled(original, FPS, hold_s=3.0)
    assert (out[10 * FPS:16 * FPS] == "D").all(), "stoppage untouched"
    assert (out[16 * FPS:17 * FPS] == "A").all(), "1 s run after restart keeps its team"
    assert (out[17 * FPS:] == "H").all()


def test_chaining_folds_consecutive_short_runs_into_one_possession():
    original = _seq(("H", 10), ("A", 1), ("H", 1), ("A", 1), ("D", 5))
    out = derive_letters_controlled(original, FPS, hold_s=3.0)
    assert (out[: 13 * FPS] == "H").all(), "all three short flips chain into H"
    assert (out[13 * FPS:] == "D").all()


def test_long_change_then_short_flip_folds_into_the_new_team():
    original = _seq(("H", 10), ("A", 10), ("H", 2))
    out = derive_letters_controlled(original, FPS, hold_s=3.0)
    assert (out[10 * FPS:20 * FPS] == "A").all(), "genuine change stands"
    assert (out[20 * FPS:] == "A").all(), "loose 2 s ball stays with the last team"


def test_dead_and_outside_frames_never_change():
    original = _seq(("D", 4), ("X", 3), ("D", 4))
    out = derive_letters_controlled(original, FPS, hold_s=3.0)
    np.testing.assert_array_equal(out, original)
    # dead-share invariance on a mixed sequence: only H/A frames may differ
    mixed = _seq(("H", 10), ("A", 1), ("D", 5))
    out2 = derive_letters_controlled(mixed, FPS, hold_s=3.0)
    dead = mixed == "D"
    np.testing.assert_array_equal(out2[dead], mixed[dead])
    assert (mixed != out2).sum() == 1 * FPS, "only the adjacent 1 s alive run may change"
