"""P4 acceptance: file format, official validator PASS on the full file, X-policy rules."""

import numpy as np
import pytest

from config import CFG
from src.decoder import CODE_A, CODE_D, CODE_H, CODE_X
from src.submit import labels_to_frame_table, write_submission


def _fake_labels(n=199_319):
    """Deterministic toy path: X outside halves, alternating blocks inside."""
    labels = np.full(n, CODE_X, dtype=np.int8)
    labels[11_185:50_000] = CODE_H
    labels[50_000:90_652] = CODE_A
    labels[110_808:150_000] = CODE_D
    labels[150_000:190_198] = CODE_H
    return labels


def test_frame_table_contract():
    labels = _fake_labels()
    table = labels_to_frame_table(labels)
    assert list(table.columns) == ["frame", "label"]
    assert len(table) == 199_319
    assert table["frame"].iloc[0] == 0 and table["frame"].iloc[-1] == 199_318
    assert (np.diff(table["frame"].to_numpy()) == 1).all()
    assert set(table["label"].unique()) <= {"H", "A", "D", "X"}
    assert table["label"].notna().all()


def test_official_reader_roundtrip(tmp_path):
    labels = _fake_labels()
    path = tmp_path / "toy.csv"
    write_submission(labels, str(path))

    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "evaluation"))
    import possession_metrics as pm

    table = pm.read_label_csv(str(path))
    codes = pm.label_codes(table)          # scorer codes: H, A, D, X = 0, 1, 2, 3
    expected = np.where(labels == CODE_X, 3, labels).astype(np.int8)
    np.testing.assert_array_equal(codes, expected)


def test_x_policy_default_off_is_noop(arrays):
    from src.x_policy import apply_x_policy

    labels = _fake_labels()
    conf = np.zeros(arrays.n_frames)  # zero confidence everywhere
    out = apply_x_policy(labels, conf, arrays.half, CFG)
    np.testing.assert_array_equal(out, labels)


def test_x_policy_rule_when_enabled(arrays):
    from src.x_policy import apply_x_policy

    cfg = {**CFG, "x_use": True}
    labels = np.full(arrays.n_frames, CODE_X, dtype=np.int8)
    # half 1: one long H segment, conf 0 deep inside, conf 1 near the start
    labels[11_185:90_652] = CODE_H
    conf = np.zeros(arrays.n_frames)
    conf[11_185:11_185 + 100] = 1.0          # evidence at the head
    out = apply_x_policy(labels, conf, arrays.half, cfg)

    head = 11_185 + 10                       # near half start (< 5 s) -> kept
    assert out[head] == CODE_H
    evidence = 11_185 + 50                   # conf=1 -> kept
    assert out[evidence] == CODE_H
    interior = 50_000                         # conf=0, far from switch/edges -> X
    assert out[interior] == CODE_X
    assert (out[arrays.half == 0] == CODE_X).all()


def test_x_policy_never_abstains_whole_half(arrays):
    from src.x_policy import apply_x_policy

    cfg = {**CFG, "x_use": True}
    labels = np.full(arrays.n_frames, CODE_X, dtype=np.int8)
    labels[11_185:90_652] = CODE_H
    conf = np.zeros(arrays.n_frames)          # every frame a candidate
    out = apply_x_policy(labels, conf, arrays.half, cfg)
    assert (out[arrays.half == 1] == CODE_H).all(), "whole-half abstention must be blocked"


def test_full_pipeline_validates_officially(tmp_path):
    """THE P4 acceptance: official validator PASS on the real generated file."""
    from src.pipeline import run
    from src.submit import validate

    labels, summary = run(CFG)
    assert len(labels) == 199_319
    path = tmp_path / "LA+AI.csv"
    write_submission(labels, str(path))
    assert validate(str(path)) == 0, "official validate_submission.py must PASS"
    # planned bands (own-output counts only)
    assert 10 <= summary["n_segments"] <= 5000
    assert summary["subsecond_share"] <= 0.02
    assert 0.05 <= summary["d_share"] <= 0.75
