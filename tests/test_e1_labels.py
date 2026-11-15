"""E1 challenge-definition labels (pre-declared: dossier §7 P6, 2026-11-09).

Unit tests of `derive_letters_controlled` on synthetic sequences: absorption,
restart protection, chaining, the strict T boundary, and dead/X invariance.
"""

import numpy as np

from tuning.idsse import derive_letters_controlled

FPS = 25                                          # 25 frames per second


def _seq(*spec):
    """Build a frame array from (label, seconds) pairs."""
    out = []                                      # accumulate frames
    for lab, secs in spec:                        # one run per pair
        out.extend([lab] * int(secs * FPS))       # repeat label for duration
    return np.array(out, dtype=object)            # frame array of labels


def test_short_adjacent_run_is_absorbed_by_the_preceding_team():
    """A sub-T control run folds into the preceding team."""
    original = _seq(("H", 10), ("A", 2), ("D", 5)) # H, A, D input runs
    out = derive_letters_controlled(original, FPS, hold_s=3.0) # apply rule at T=3
    assert (out[: 10 * FPS] == "H").all()         # leading H untouched
    assert (out[10 * FPS:12 * FPS] == "H").all(), "2 s control folds into the last team" # two-second A absorbed
    assert (out[12 * FPS:] == "D").all()          # dead run untouched
    assert (original[10 * FPS:12 * FPS] == "A").all(), "input must not be mutated" # input left unchanged


def test_run_at_or_above_t_is_a_genuine_possession_change():
    """A run at exactly T counts as a real change."""
    original = _seq(("H", 10), ("A", 3), ("H", 10))   # exactly T = 3 s
    out = derive_letters_controlled(original, FPS, hold_s=3.0) # apply rule at T=3
    assert (out[10 * FPS:13 * FPS] == "A").all(), "shorter than T is strict: ==T is kept" # exactly T is kept


def test_first_alive_run_after_a_stoppage_is_never_absorbed():
    """The first run after a stoppage keeps its team."""
    original = _seq(("H", 10), ("D", 6), ("A", 1), ("H", 10)) # H, D, short A, H
    out = derive_letters_controlled(original, FPS, hold_s=3.0) # apply rule at T=3
    assert (out[10 * FPS:16 * FPS] == "D").all(), "stoppage untouched" # stoppage untouched
    assert (out[16 * FPS:17 * FPS] == "A").all(), "1 s run after restart keeps its team" # first run keeps team
    assert (out[17 * FPS:] == "H").all()          # rest returns to H


def test_chaining_folds_consecutive_short_runs_into_one_possession():
    """Consecutive short flips chain into one possession."""
    original = _seq(("H", 10), ("A", 1), ("H", 1), ("A", 1), ("D", 5)) # three one-second flips
    out = derive_letters_controlled(original, FPS, hold_s=3.0) # apply rule at T=3
    assert (out[: 13 * FPS] == "H").all(), "all three short flips chain into H" # flips chain into H
    assert (out[13 * FPS:] == "D").all()          # dead run untouched


def test_long_change_then_short_flip_folds_into_the_new_team():
    """A short flip after a long change folds into the new team."""
    original = _seq(("H", 10), ("A", 10), ("H", 2)) # long A, short H
    out = derive_letters_controlled(original, FPS, hold_s=3.0) # apply rule at T=3
    assert (out[10 * FPS:20 * FPS] == "A").all(), "genuine change stands" # long change stands
    assert (out[20 * FPS:] == "A").all(), "loose 2 s ball stays with the last team" # short flip absorbed


def test_dead_and_outside_frames_never_change():
    """Dead/X frames never change; only alive runs may."""
    original = _seq(("D", 4), ("X", 3), ("D", 4)) # only dead and X
    out = derive_letters_controlled(original, FPS, hold_s=3.0) # apply rule at T=3
    np.testing.assert_array_equal(out, original)  # nothing changed
    # dead-share invariance on a mixed sequence: only H/A frames may differ
    mixed = _seq(("H", 10), ("A", 1), ("D", 5))
    out2 = derive_letters_controlled(mixed, FPS, hold_s=3.0) # apply rule at T=3
    dead = mixed == "D"                           # dead frames mask
    np.testing.assert_array_equal(out2[dead], mixed[dead]) # dead frames unchanged
    assert (mixed != out2).sum() == 1 * FPS, "only the adjacent 1 s alive run may change" # only one run changed
