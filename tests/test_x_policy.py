"""X-policy boundary tests: distances to switches/edges and the >= / <= edges of the rule.

The shipped config has x_use=False (test_submission covers the no-op); these tests pin
the mechanics so that the P5 ablation result stays reproducible.
"""

import numpy as np

from config import CFG
from src.decoder import CODE_A, CODE_H, CODE_X
from src.x_policy import _distance_to_nearest_boundary, apply_x_policy


def test_distance_without_switches_is_distance_to_span_edges():
    """No switches: distance shrinks to zero at both span edges."""
    dist = _distance_to_nearest_boundary(7, np.array([], dtype=int))  # span with no switches
    np.testing.assert_array_equal(dist, [0, 1, 2, 3, 2, 1, 0])  # falls toward both edges


def test_distance_to_nearest_switch_or_edge():
    """Distance falls to zero at switches and at span edges."""
    # switches at 10 and 30 in a 50-frame span (index = first frame of the new segment)
    dist = _distance_to_nearest_boundary(50, np.array([10, 30]))
    assert dist[0] == 0 and dist[49] == 0            # span edges
    assert dist[5] == 5                              # min(edge 5, switch 10-5)
    assert dist[10] == 0 and dist[30] == 0           # on the switch frame
    assert dist[12] == 2 and dist[29] == 1           # offsets from the nearest switch
    assert dist[20] == 10                            # midway between the two switches
    assert dist[45] == 4                             # closer to the end edge than to 30
    assert (dist >= 0).all() and np.isfinite(dist).all()  # distances all sane


def _one_switch_half(arrays, local_switch):
    """Half 1 = H then A from `local_switch` (local index); everything else X."""
    idx = np.flatnonzero(arrays.half == 1)  # half-1 frame indices
    labels = np.full(arrays.n_frames, CODE_X, dtype=np.int8)  # X everywhere first
    labels[idx] = CODE_H  # half 1 starts H
    labels[idx[local_switch:]] = CODE_A  # then flips to A
    return idx, labels  # indices and toy labels


def test_stable_distance_edge_is_inclusive(arrays):
    """At exactly stable distance the frame abstains; one closer keeps its label."""
    cfg = {**CFG, "x_use": True}  # switch the policy on
    stable = int(round(cfg["x_stable_s"] * cfg["fps"]))  # stable window in frames
    idx, labels = _one_switch_half(arrays, 40_000)  # switch placed at 40k
    conf = np.zeros(arrays.n_frames)            # every graded frame is low-confidence
    out = apply_x_policy(labels, conf, arrays.half, cfg)  # apply the policy

    assert out[idx[40_000 + stable]] == CODE_X, ">= stable frames from the switch abstains"  # exactly at window edge
    assert out[idx[40_000 + stable - 1]] == CODE_A, "one frame closer keeps the label"  # one inside keeps A
    assert out[idx[40_000 - stable]] == CODE_X  # symmetric far side
    assert out[idx[40_000 - stable + 1]] == CODE_H  # one inside keeps H
    assert out[idx[stable]] == CODE_X and out[idx[stable - 1]] == CODE_H, "half start"  # half-start edge
    assert out[idx[-1 - stable]] == CODE_X and out[idx[-stable]] == CODE_A, "half end"  # half-end edge
    assert (out[arrays.half == 0] == CODE_X).all()   # ungraded frames untouched
    assert (out[arrays.half == 2] == CODE_X).all()   # half 2 input was X, stays X


def test_confidence_edge_is_inclusive(arrays):
    """conf exactly at x_conf_max counts as low; just above it is kept."""
    cfg = {**CFG, "x_use": True}  # switch the policy on
    idx, labels = _one_switch_half(arrays, 40_000)  # same toy labels
    conf = np.ones(arrays.n_frames)             # confident everywhere ...
    conf[idx[20_000]] = cfg["x_conf_max"]       # ... except two interior frames
    conf[idx[20_001]] = cfg["x_conf_max"] + 1e-9  # just above threshold
    out = apply_x_policy(labels, conf, arrays.half, cfg)  # apply the policy
    assert out[idx[20_000]] == CODE_X, "conf == x_conf_max counts as low"  # at-threshold abstains
    assert out[idx[20_001]] == CODE_H, "just above the threshold is kept"  # above-threshold kept
    assert (out[idx[:20_000]] == CODE_H).all()  # confident head kept


def test_policy_does_not_touch_inputs(arrays):
    """The policy returns a copy; inputs are never mutated."""
    cfg = {**CFG, "x_use": True}  # switch the policy on
    _, labels = _one_switch_half(arrays, 40_000)  # same toy labels
    conf = np.zeros(arrays.n_frames)  # zero confidence
    before = labels.copy()  # snapshot labels
    apply_x_policy(labels, conf, arrays.half, cfg)  # run the policy
    np.testing.assert_array_equal(labels, before)  # labels unchanged
