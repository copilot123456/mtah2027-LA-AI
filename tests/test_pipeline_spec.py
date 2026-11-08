"""Framework tests: spec validation (requires/provides, docstrings, duplicates),
runner behaviour on custom specs, and the default declaration's shape.

The default spec's end-to-end equivalence with the frozen CSV is enforced by
tests/test_submission.py (byte-identity + official validator).
"""

import numpy as np
import pytest

from config import CFG
from src.pipeline import Step, run, validate_spec


def test_default_spec_validates_clean():
    from pipeline_spec import SPEC

    assert validate_spec(SPEC) == []


def test_default_spec_stage_names():
    from pipeline_spec import SPEC

    assert [s.name for s in SPEC] == [
        "load", "ball_fill", "emission", "decode", "x_policy", "summary",
    ]


def test_requires_must_be_provided_by_an_earlier_step():
    def needs_track(ctx, cfg):
        """Toy step that wants a track nobody built."""
        pass

    problems = validate_spec([Step("bad", needs_track, requires=("track",))])
    assert problems, "an unsatisfiable requires must be reported"
    assert "track" in problems[0] and "bad" in problems[0]


def test_undocumented_step_is_rejected():
    def naked(ctx, cfg):
        pass

    problems = validate_spec([Step("naked", naked)])
    assert any("docstring" in p for p in problems), "steps must document themselves"
    with pytest.raises(ValueError, match="docstring"):
        run(CFG, spec=[Step("naked", naked)])


def test_duplicate_step_names_rejected():
    def handler(ctx, cfg):
        """Toy handler."""
        pass

    problems = validate_spec([Step("dup", handler), Step("dup", handler)])
    assert any("duplicate" in p for p in problems)


def test_runner_executes_in_order_and_returns_labels_and_summary():
    order = []

    def first(ctx, cfg):
        """Provide toy labels and record execution order."""
        ctx.labels = np.zeros(4, dtype=np.int8)
        order.append("first")

    def second(ctx, cfg):
        """Record execution order and provide the summary."""
        order.append("second")
        ctx.summary = {"n_segments": 1}

    spec = [
        Step("first", first, provides=("labels",)),
        Step("second", second, requires=("labels",), provides=("summary",)),
    ]
    labels, summary = run(CFG, spec=spec)
    assert summary == {"n_segments": 1}
    assert len(labels) == 4
    assert order == ["first", "second"], "steps must run in declaration order"


def test_swapping_the_handler_is_the_only_change_needed():
    def base(ctx, cfg):
        """Baseline handler."""
        ctx.labels = np.zeros(4, dtype=np.int8)
        ctx.summary = {"handler": "base"}

    def alternative(ctx, cfg):
        """Replacement handler (e.g. ViT instead of YOLO)."""
        ctx.labels = np.ones(4, dtype=np.int8)
        ctx.summary = {"handler": "alternative"}

    def spec_with(fn):
        return [Step("stage", fn, provides=("labels", "summary"))]

    _, s_base = run(CFG, spec=spec_with(base))
    _, s_alt = run(CFG, spec=spec_with(alternative))
    assert s_base["handler"] == "base"
    assert s_alt["handler"] == "alternative"


def test_run_requires_labels_and_summary():
    def noop(ctx, cfg):
        """Does nothing."""
        pass

    with pytest.raises(ValueError, match="labels"):
        run(CFG, spec=[Step("noop", noop)])
