"""P4 acceptance: file format, official validator PASS on the full file, X-policy rules,
and the byte-identity of the committed LA+AI.csv with a fresh run."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from config import CFG
from src.decoder import CODE_A, CODE_D, CODE_H, CODE_X
from src.submit import labels_to_frame_table, validate, write_submission

ROOT = Path(__file__).resolve().parents[1]  # repo root locator


def _fake_labels(n=199_319):
    """Deterministic toy path: X outside halves, alternating blocks inside."""
    labels = np.full(n, CODE_X, dtype=np.int8)  # X outside halves
    labels[11_185:50_000] = CODE_H  # half-1 home block
    labels[50_000:90_652] = CODE_A  # half-1 away block
    labels[110_808:150_000] = CODE_D  # half-2 dead-ball block
    labels[150_000:190_198] = CODE_H  # half-2 home block
    return labels  # toy path ready


def test_frame_table_contract():
    """Frame table: exact columns, row count, 0-based range, valid labels."""
    labels = _fake_labels()  # toy labels in
    table = labels_to_frame_table(labels)  # convert to table
    assert list(table.columns) == ["frame", "label"]  # exact column names
    assert len(table) == 199_319  # one row per frame
    assert table["frame"].iloc[0] == 0 and table["frame"].iloc[-1] == 199_318  # 0-based ids
    assert (np.diff(table["frame"].to_numpy()) == 1).all()  # frame ids contiguous
    assert set(table["label"].unique()) <= {"H", "A", "D", "X"}  # only valid labels
    assert table["label"].notna().all()  # no missing labels


def test_frame_table_rejects_bad_input():
    """Wrong length or unknown label codes raise clear errors."""
    with pytest.raises(ValueError, match="expected 199319 labels"):  # wrong length rejected
        labels_to_frame_table(np.zeros(10, dtype=np.int8))  # only ten labels
    bad = _fake_labels()  # valid, then broken
    bad[12_000] = 5                            # not one of CODE_H/A/D/X
    with pytest.raises(ValueError, match="unknown label codes"):  # bad code rejected
        labels_to_frame_table(bad)  # run the check


def test_validate_runs_from_another_cwd(tmp_path, monkeypatch):
    """The validator is found relative to the package, not the current directory."""
    path = tmp_path / "toy.csv"  # scratch output path
    write_submission(_fake_labels(), str(path))  # write toy submission
    monkeypatch.chdir(tmp_path)  # leave the repo cwd
    assert validate("toy.csv") == 0  # validator still passes


def test_official_reader_roundtrip(tmp_path):
    """The official scorer reads back exactly the labels we wrote."""
    labels = _fake_labels()  # toy labels in
    path = tmp_path / "toy.csv"  # scratch output path
    write_submission(labels, str(path))  # write the file

    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]  # repo root again
    sys.path.insert(0, str(root / "evaluation"))  # make scorer importable
    import possession_metrics as pm

    table = pm.read_label_csv(str(path))  # official reader opens it
    codes = pm.label_codes(table)          # scorer codes: H, A, D, X = 0, 1, 2, 3
    expected = np.where(labels == CODE_X, 3, labels).astype(np.int8)  # our codes, scorer numbering
    np.testing.assert_array_equal(codes, expected)  # round-trip is exact


def test_x_policy_default_off_is_noop(arrays):
    """Shipped config (x_use=False): the policy changes nothing."""
    from src.x_policy import apply_x_policy

    labels = _fake_labels()  # toy labels in
    conf = np.zeros(arrays.n_frames)  # zero confidence everywhere
    out = apply_x_policy(labels, conf, arrays.half, CFG)  # shipped config: no-op
    np.testing.assert_array_equal(out, labels)  # output equals input


def test_x_policy_rule_when_enabled(arrays):
    """Enabled: evidenced and near-edge labels survive; deep interior abstains."""
    from src.x_policy import apply_x_policy

    cfg = {**CFG, "x_use": True}  # switch the policy on
    labels = np.full(arrays.n_frames, CODE_X, dtype=np.int8)  # X everywhere first
    # half 1: one long H segment, conf 0 deep inside, conf 1 near the start
    labels[11_185:90_652] = CODE_H
    conf = np.zeros(arrays.n_frames)  # zero confidence everywhere
    conf[11_185:11_185 + 100] = 1.0          # evidence at the head
    out = apply_x_policy(labels, conf, arrays.half, cfg)  # apply the rule

    head = 11_185 + 10                       # near half start (< 5 s) -> kept
    assert out[head] == CODE_H  # head frame kept
    evidence = 11_185 + 50                   # conf=1 -> kept
    assert out[evidence] == CODE_H  # evidenced frame kept
    interior = 50_000                         # conf=0, far from switch/edges -> X
    assert out[interior] == CODE_X  # deep interior abstains
    assert (out[arrays.half == 0] == CODE_X).all()  # ungraded frames untouched


def test_x_policy_never_abstains_whole_half(arrays):
    """A switch-free, zero-confidence half keeps its label near both edges (the rule's
    half-start / half-end clause) and abstains only in the interior; with x_stable_s=0
    every frame qualifies and the explicit whole-half guard keeps the label."""
    from src.x_policy import apply_x_policy

    cfg = {**CFG, "x_use": True}  # switch the policy on
    labels = np.full(arrays.n_frames, CODE_X, dtype=np.int8)  # X everywhere first
    labels[11_185:90_652] = CODE_H  # one long home segment
    conf = np.zeros(arrays.n_frames)          # every frame low-confidence
    out = apply_x_policy(labels, conf, arrays.half, cfg)  # apply the rule
    half1 = out[arrays.half == 1]  # restrict to half 1
    stable = int(round(cfg["x_stable_s"] * cfg["fps"]))  # stable window in frames
    assert (half1[:stable] == CODE_H).all() and (half1[-stable:] == CODE_H).all()  # edges keep labels
    assert (half1[stable:-stable] == CODE_X).all()  # interior abstains
    assert (half1 == CODE_H).any(), "whole-half abstention must be blocked"  # not the whole half

    out0 = apply_x_policy(labels, conf, arrays.half, {**cfg, "x_stable_s": 0.0})  # zero stable window
    assert (out0[arrays.half == 1] == CODE_H).all(), "explicit guard when every frame qualifies"  # guard keeps the half


@pytest.fixture(scope="module")
def pipeline_output():
    """One full run shared by the acceptance tests below (a few seconds with a warm cache)."""
    from src.pipeline import run

    return run(CFG)  # one shared run


def test_full_pipeline_validates_officially(tmp_path, pipeline_output):
    """THE P4 acceptance: official validator PASS on the real generated file."""
    labels, summary = pipeline_output  # unpack shared run
    assert len(labels) == 199_319  # full frame count
    path = tmp_path / "LA+AI.csv"  # scratch LA+AI.csv
    write_submission(labels, str(path))  # write fresh file
    assert validate(str(path)) == 0, "official validate_submission.py must PASS"  # official validator passes
    # planned bands (own-output counts only)
    assert 10 <= summary["n_segments"] <= 5000
    assert summary["subsecond_share"] <= 0.02  # planned subsecond band
    assert 0.05 <= summary["d_share"] <= 0.75  # planned D-share band


def test_committed_submission_is_what_the_code_produces(tmp_path, pipeline_output):
    """Byte-identity guard on OUR output (not a ground-truth snapshot): the LA+AI.csv in
    the repository must be exactly what `python -m src.submit` writes from this code and
    config. A config or decoder change must regenerate the file in the same commit."""
    labels, _ = pipeline_output  # reuse shared run
    fresh = tmp_path / "fresh.csv"  # fresh output path
    write_submission(labels, str(fresh))  # write fresh copy
    committed = ROOT / "LA+AI.csv"  # committed submission file
    assert committed.exists(), "LA+AI.csv is part of the repository"  # must be in repo
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()  # noqa: E731
    assert md5(fresh) == md5(committed), (  # byte-identity must hold
        "committed LA+AI.csv differs from a fresh run — regenerate it with "
        "`python -m src.submit` and commit it together with the change")
