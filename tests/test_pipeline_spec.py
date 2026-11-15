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
    """The shipped default spec passes validation with no problems."""
    from pipeline_spec import SPEC

    assert validate_spec(SPEC) == []  # default spec is clean


def test_default_spec_stage_names():
    """Default spec names the six pipeline stages in order."""
    from pipeline_spec import SPEC

    assert [s.name for s in SPEC] == [  # six stages, in order
        "load", "ball_fill", "emission", "decode", "x_policy", "summary",
    ]


def test_requires_must_be_provided_by_an_earlier_step():
    """A requires no earlier step provides must be reported."""
    def needs_track(ctx, cfg):
        """Toy step that wants a track nobody built."""
        pass  # nothing to do

    problems = validate_spec([Step("bad", needs_track, requires=("track",))])  # spec with unmet requires
    assert problems, "an unsatisfiable requires must be reported"  # must report something
    assert "track" in problems[0] and "bad" in problems[0]  # names step and requirement


def test_undocumented_step_is_rejected():
    """A step without a docstring fails validation and run."""
    def naked(ctx, cfg):
        pass  # no docstring by design

    problems = validate_spec([Step("naked", naked)])  # validate bare step
    assert any("docstring" in p for p in problems), "steps must document themselves"  # missing docstring flagged
    with pytest.raises(ValueError, match="docstring"):  # rejection expected
        run(CFG, spec=[Step("naked", naked)])  # run must also reject


def test_duplicate_step_names_rejected():
    """Duplicate step names are reported by validation."""
    def handler(ctx, cfg):
        """Toy handler."""
        pass  # no-op handler

    problems = validate_spec([Step("dup", handler), Step("dup", handler)])  # same name twice
    assert any("duplicate" in p for p in problems)  # duplicate reported


def test_runner_executes_in_order_and_returns_labels_and_summary():
    """Runner executes steps in declaration order and returns labels plus summary."""
    order = []  # execution log

    def first(ctx, cfg):
        """Provide toy labels and record execution order."""
        ctx.labels = np.zeros(4, dtype=np.int8)  # four fake labels
        order.append("first")  # log first ran

    def second(ctx, cfg):
        """Record execution order and provide the summary."""
        order.append("second")  # log second ran
        ctx.summary = {"n_segments": 1}  # provide a summary

    spec = [  # two-step declaration
        Step("first", first, provides=("labels",)),
        Step("second", second, requires=("labels",), provides=("summary",)),
    ]
    labels, summary = run(CFG, spec=spec)  # execute the spec
    assert summary == {"n_segments": 1}  # summary from second step
    assert len(labels) == 4  # labels from first step
    assert order == ["first", "second"], "steps must run in declaration order"  # declaration order kept


def test_swapping_the_handler_is_the_only_change_needed():
    """Swapping the handler is enough; the spec shape stays identical."""
    def base(ctx, cfg):
        """Baseline handler."""
        ctx.labels = np.zeros(4, dtype=np.int8)  # baseline labels
        ctx.summary = {"handler": "base"}  # baseline summary

    def alternative(ctx, cfg):
        """Replacement handler (e.g. ViT instead of YOLO)."""
        ctx.labels = np.ones(4, dtype=np.int8)  # alternative labels
        ctx.summary = {"handler": "alternative"}  # alternative summary

    def spec_with(fn):
        return [Step("stage", fn, provides=("labels", "summary"))]  # one-step spec wrapper

    _, s_base = run(CFG, spec=spec_with(base))  # run with baseline
    _, s_alt = run(CFG, spec=spec_with(alternative))  # run with replacement
    assert s_base["handler"] == "base"  # baseline honored
    assert s_alt["handler"] == "alternative"  # replacement honored


def test_run_requires_labels_and_summary():
    """A run that never gets labels fails fast."""
    def noop(ctx, cfg):
        """Does nothing."""
        pass  # never provides labels

    with pytest.raises(ValueError, match="labels"):  # missing labels rejected
        run(CFG, spec=[Step("noop", noop)])  # run must fail
