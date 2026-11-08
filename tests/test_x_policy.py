"""X-policy boundary tests: distances to switches/edges and the >= / <= edges of the rule.

The shipped config has x_use=False (test_submission covers the no-op); these tests pin
the mechanics so that the P5 ablation result stays reproducible.
"""

import numpy as np

from config import CFG
from src.decoder import CODE_A, CODE_H, CODE_X
from src.x_policy import _distance_to_nearest_boundary, apply_x_policy


def test_distance_without_switches_is_distance_to_span_edges():
    dist = _distance_to_nearest_boundary(7, np.array([], dtype=int))
    np.testing.assert_array_equal(dist, [0, 1, 2, 3, 2, 1, 0])


def test_distance_to_nearest_switch_or_edge():
    # switches at 10 and 30 in a 50-frame span (index = first frame of the new segment)
    dist = _distance_to_nearest_boundary(50, np.array([10, 30]))
    assert dist[0] == 0 and dist[49] == 0            # span edges
    assert dist[5] == 5                              # min(edge 5, switch 10-5)
    assert dist[10] == 0 and dist[30] == 0           # on the switch frame
    assert dist[12] == 2 and dist[29] == 1
    assert dist[20] == 10                            # midway between the two switches
    assert dist[45] == 4                             # closer to the end edge than to 30
    assert (dist >= 0).all() and np.isfinite(dist).all()


def _one_switch_half(arrays, local_switch):
    """Half 1 = H then A from `local_switch` (local index); everything else X."""
    idx = np.flatnonzero(arrays.half == 1)
    labels = np.full(arrays.n_frames, CODE_X, dtype=np.int8)
    labels[idx] = CODE_H
    labels[idx[local_switch:]] = CODE_A
    return idx, labels


def test_stable_distance_edge_is_inclusive(arrays):
    cfg = {**CFG, "x_use": True}
    stable = int(round(cfg["x_stable_s"] * cfg["fps"]))
    idx, labels = _one_switch_half(arrays, 40_000)
    conf = np.zeros(arrays.n_frames)            # every graded frame is low-confidence
    out = apply_x_policy(labels, conf, arrays.half, cfg)

    assert out[idx[40_000 + stable]] == CODE_X, ">= stable frames from the switch abstains"
    assert out[idx[40_000 + stable - 1]] == CODE_A, "one frame closer keeps the label"
    assert out[idx[40_000 - stable]] == CODE_X
    assert out[idx[40_000 - stable + 1]] == CODE_H
    assert out[idx[stable]] == CODE_X and out[idx[stable - 1]] == CODE_H, "half start"
    assert out[idx[-1 - stable]] == CODE_X and out[idx[-stable]] == CODE_A, "half end"
    assert (out[arrays.half == 0] == CODE_X).all()
    assert (out[arrays.half == 2] == CODE_X).all()


def test_confidence_edge_is_inclusive(arrays):
    cfg = {**CFG, "x_use": True}
    idx, labels = _one_switch_half(arrays, 40_000)
    conf = np.ones(arrays.n_frames)             # confident everywhere ...
    conf[idx[20_000]] = cfg["x_conf_max"]       # ... except two interior frames
    conf[idx[20_001]] = cfg["x_conf_max"] + 1e-9
    out = apply_x_policy(labels, conf, arrays.half, cfg)
    assert out[idx[20_000]] == CODE_X, "conf == x_conf_max counts as low"
    assert out[idx[20_001]] == CODE_H, "just above the threshold is kept"
    assert (out[idx[:20_000]] == CODE_H).all()


def test_policy_does_not_touch_inputs(arrays):
    cfg = {**CFG, "x_use": True}
    _, labels = _one_switch_half(arrays, 40_000)
    conf = np.zeros(arrays.n_frames)
    before = labels.copy()
    apply_x_policy(labels, conf, arrays.half, cfg)
    np.testing.assert_array_equal(labels, before)
