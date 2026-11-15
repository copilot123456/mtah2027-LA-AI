"""P8 decoder components (dossier §7 P8): gate move mode, gap placement, D minimum, delay."""

import numpy as np

from config import CFG
from src.ball_fill import SOURCE_ATD, SOURCE_NONE
from src.decoder import (CODE_A, CODE_D, CODE_H, apply_gating, delay_team_switches,
                         enforce_d_min_dwell, place_gap_switches)

H, A, D = CODE_H, CODE_A, CODE_D                  # short label aliases


def _log_e(winners):
    """(N, 3) emission whose argmax per frame is `winners`."""
    out = np.full((len(winners), 3), -5.0)        # fill low everywhere
    out[np.arange(len(winners)), winners] = 0.0   # chosen label scores top
    return out                                    # fake emissions


def _gap_case():
    """200 frames: H evidence 0-59, ball gap 60-139, A evidence 140-199; Viterbi put the
    switch at the gap start (frame 60), 80 frames before any A evidence."""
    labels = np.array([H] * 60 + [A] * 140, dtype=np.int8) # H then A labels
    source = np.full(200, SOURCE_ATD, dtype=np.int8) # ATD everywhere first
    source[60:140] = SOURCE_NONE                  # gap in the middle
    log_e = _log_e(np.array([H] * 60 + [H] * 80 + [A] * 60)) # emissions follow labels
    log_e[60:140] = -1.0                      # flat inside the gap
    return labels, source, log_e                  # hand back the case


def test_revert_mode_swallows_a_supported_segment():
    """Revert mode drops the segment when its boundary sits in a gap."""
    labels, source, log_e = _gap_case()           # reuse the gap case
    out, reverted = apply_gating(labels, source, log_e, {**CFG, "gate_mode": "revert"}) # run gate in revert mode
    assert reverted == 1                          # one switch reverted
    assert (out == H).all()                   # 60 frames of A ball evidence are lost


def test_move_mode_keeps_the_segment_from_its_first_support():
    """Move mode slides the boundary to the first supporting frame."""
    labels, source, log_e = _gap_case()           # reuse the gap case
    out, reverted = apply_gating(labels, source, log_e, {**CFG, "gate_mode": "move"}) # run gate in move mode
    assert reverted == 0                          # switch kept
    # boundary moved from gap start (60) to first A evidence (140)
    assert (out[:140] == H).all() and (out[140:] == A).all()


def test_move_mode_still_merges_a_segment_without_any_support():
    """Move mode still merges a segment with no support at all."""
    labels, source, log_e = _gap_case()           # reuse the gap case
    log_e[140:] = _log_e(np.full(60, H))      # the ball never favours A
    out, reverted = apply_gating(labels, source, log_e, {**CFG, "gate_mode": "move"}) # run gate in move mode
    assert reverted == 1 and (out == H).all()     # no support: merged away


def test_move_mode_leaves_supported_boundaries_alone():
    """Both modes leave an already-supported boundary alone."""
    labels = np.array([H] * 100 + [A] * 100, dtype=np.int8) # H then A labels
    source = np.full(200, SOURCE_ATD, dtype=np.int8) # ATD everywhere
    log_e = _log_e(labels)                        # emissions follow labels
    for mode in ("revert", "move"):               # test both modes
        out, reverted = apply_gating(labels, source, log_e, {**CFG, "gate_mode": mode}) # run gate in this mode
        assert reverted == 0 and (out == labels).all() # supported boundary untouched


def test_gap_placement_middle_and_noop():
    """Gap placement centres the switch in the gap; frac 0 is a no-op."""
    labels = np.array([H] * 140 + [A] * 60, dtype=np.int8)     # boundary at the gap end
    source = np.full(200, SOURCE_ATD, dtype=np.int8) # ATD everywhere first
    source[60:140] = SOURCE_NONE                  # gap in the middle
    assert place_gap_switches(labels, source, 0.0) is labels # zero fraction is no-op
    out = place_gap_switches(labels, source, 0.5) # move switch halfway
    assert (out[:100] == H).all() and (out[100:] == A).all() # switch sits mid-gap
    early = np.array([H] * 60 + [A] * 140, dtype=np.int8)      # boundary at the gap start
    assert (place_gap_switches(early, source, 0.5) == out).all()   # canonical: same result


def test_gap_placement_ignores_visible_and_dead_boundaries():
    """Only ball-gap boundaries move; visible and dead ones stay."""
    source = np.full(200, SOURCE_ATD, dtype=np.int8) # ATD everywhere first
    source[60:140] = SOURCE_NONE                  # gap in the middle
    visible = np.array([H] * 30 + [A] * 170, dtype=np.int8) # visible-boundary labels
    assert (place_gap_switches(visible, source, 0.5) == visible).all() # visible boundary untouched
    dead = np.array([D] * 60 + [A] * 140, dtype=np.int8) # dead-boundary labels
    assert (place_gap_switches(dead, source, 0.5) == dead).all() # dead boundary untouched


def test_gap_placement_never_erases_a_segment():
    """Gap placement never erases a whole segment."""
    labels = np.array([H] * 5 + [A] * 3 + [H] * 192, dtype=np.int8) # three spells, tiny A
    source = np.full(200, SOURCE_NONE, dtype=np.int8)           # one gap over everything
    out = place_gap_switches(labels, source, 0.5) # run gap placement
    assert set(np.unique(out)) == {H, A}          # both labels survive
    assert (np.diff(np.flatnonzero(out == A)) == 1).all() and (out == A).sum() >= 1 # A run stays contiguous


def test_d_minimum_merges_short_dead_runs_only():
    """D min-dwell merges short stoppages and keeps long ones."""
    labels = np.array([H] * 100 + [D] * 60 + [H] * 100 + [D] * 200 + [A] * 100, dtype=np.int8) # H D H D A spells
    out = enforce_d_min_dwell(labels, 75)         # minimum D dwell 75
    assert (out[:260] == H).all()             # H D H -> one possession
    assert (out[260:460] == D).all()          # the long stoppage stays
    assert (out[460:] == A).all()                 # final A untouched


def test_d_minimum_between_two_teams_goes_to_the_longer_neighbour():
    """A stoppage shorter than the minimum joins the longer side."""
    labels = np.array([H] * 100 + [D] * 60 + [A] * 300, dtype=np.int8) # H, D, A spells
    out = enforce_d_min_dwell(labels, 75)         # minimum D dwell 75
    assert (out[:100] == H).all() and (out[100:] == A).all() # short D joins longer A
    # 60-frame stoppage meets a 50 minimum: untouched
    untouched = enforce_d_min_dwell(labels, 50)
    assert (untouched == labels).all()            # longer minimum: no merge


def test_delay_moves_ball_placed_team_switches_only():
    """Switch delay shifts ball-placed team switches, not D edges."""
    labels = np.array([H] * 100 + [A] * 100 + [D] * 100 + [H] * 100, dtype=np.int8) # H A D H spells
    source = np.full(400, SOURCE_ATD, dtype=np.int8) # ATD everywhere
    out = delay_team_switches(labels, source, 12) # delay by 12 frames
    # H->A boundary delayed by 12 frames
    assert (out[:112] == H).all() and (out[112:200] == A).all()
    assert (out[200:] == labels[200:]).all()   # into and out of D: untouched
    assert delay_team_switches(labels, source, 0) is labels # zero delay is no-op


def test_delay_is_capped_and_skips_gap_boundaries():
    """Delay is capped by segment size and skipped next to gaps."""
    labels = np.array([H] * 100 + [A] * 10 + [H] * 100, dtype=np.int8) # short A run mid-match
    source = np.full(210, SOURCE_ATD, dtype=np.int8) # ATD everywhere
    out = delay_team_switches(labels, source, 12) # delay by 12 frames
    # start: capped at half the 10-frame segment (+5); end: +12 (half of 100 allows it)
    assert (out[:105] == H).all() and (out[105:122] == A).all() and (out[122:] == H).all()
    gap = source.copy()                           # copy source for gap
    gap[99] = SOURCE_NONE                      # the first boundary touches a ball gap
    out = delay_team_switches(labels, gap, 12)    # run delay with gap
    assert (out[:100] == H).all() and (out[100:122] == A).all() and (out[122:] == H).all() # gap boundary not delayed


# ---------------- stage 8: grid, rule, committed verdicts ----------------

def test_stage8_grid_is_the_pinned_factorial():
    """Stage-8 grid is the pinned 72-cell factorial."""
    from tuning.stage8 import REFERENCE, p8_cells

    cells = p8_cells()                            # build the grid
    assert len(cells) == 72 and cells[0] == REFERENCE # 72 cells, reference first
    assert len({tuple(c.values()) for c in cells}) == 72 # all cells unique


def test_default_flags_are_the_p7_decoder():
    """With the P8 keys at their OFF values the decoder is the pre-P8 one."""
    labels, source, log_e = _gap_case()           # reuse the gap case
    off = {**CFG, "gate_mode": "revert", "gap_switch_frac": 0.0, # every P8 key off
           "switch_delay_s": 0.0, "d_min_dwell_s": 0.0}
    out, _ = apply_gating(labels, source, log_e, off) # run gate with OFF flags
    assert (out == H).all()                       # behaves like P7


def test_committed_stage8_verdicts_and_shipped_config():
    """The recorded read: G keep (move), S keep (0.64 s), M drop, bundle adopted — and the
    shipped config is exactly that bundle."""
    from pathlib import Path

    import pytest

    from tuning import gate8

    if not (Path(gate8.RES) / "stage8_uniform.csv").exists(): # matrices are optional
        pytest.skip("stage-8 matrices not present") # skip without matrices
    tables = gate8.load_tables()                  # load recorded tables
    # tables reproduce the recorded harness pin
    assert gate8.harness_pin(tables) <= gate8.PIN_TOL
    rule = gate8.apply_rule(tables)               # apply the rule
    assert rule["G"]["keep"] and rule["S"]["keep"] and not rule["M"]["keep"] # G and S kept, M dropped
    assert rule["adopt"] and rule["bundle_primary"] > 0.02 and rule["bundle_wins"] >= 5 # bundle adopted and wins
    assert all(v[0] > 0 for v in rule["bundle_deltas"].values()) # every delta positive
    for key in ("gate_mode", "gap_switch_frac", "switch_delay_s", "d_min_dwell_s"): # compare each pinned key
        assert CFG[key] == rule["bundle"][key], key # shipped config matches bundle
