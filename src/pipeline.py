"""Spec-driven pipeline runner and step contract.

The pipeline itself is declared in `pipeline_spec.py` at the repository root: an
ordered list of `Step`s. Swap a stage (e.g. YOLO -> ViT gap filling) or add a new
one by editing that file; this module only validates and executes the declaration.

Contract (full team standard: PIPELINE_STANDARD.html / PIPELINE_STANDARD_AI.md):
  * A step is `Step(name, fn, requires, provides)` with `fn(ctx, cfg) -> None`:
    it reads what it needs from the `Ctx` and stores its results back.
  * `validate_spec` runs BEFORE any step executes: `requires` must be satisfied
    by earlier steps, names must be unique, and every `fn` must carry a
    docstring — the framework enforces that each step documents itself.
  * `run` returns `(labels, summary)`; a spec that never produces either fails.
  * The default spec reproduces the frozen submission byte-for-byte
    (md5 c121715836ed603fc5792b85daf6f279); tests/test_submission.py enforces it.

The only inputs are the provided match files and cfg; nothing here reads ground
truth or the video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import CFG


@dataclass
class Ctx:
    """State handed from step to step; `extra` is free space for new steps."""

    arrays: object = None
    track: object = None
    emission: object = None
    decoded: object = None
    labels: np.ndarray | None = None
    summary: dict | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    """One pipeline stage: `fn(ctx, cfg) -> None` plus its data-flow declaration."""

    name: str
    fn: Callable[..., None]
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()


def validate_spec(spec: list[Step]) -> list[str]:
    """Problems with the spec; an empty list means it is valid.

    Runs before any step executes so a mis-declared spec fails fast with the
    offending step named, not with an AttributeError in the middle of a run.
    """
    problems: list[str] = []
    provided: set[str] = set()
    seen: set[str] = set()
    for step in spec:
        if step.name in seen:
            problems.append(f"step {step.name!r}: duplicate step name")
        seen.add(step.name)
        if not (getattr(step.fn, "__doc__", "") or "").strip():
            problems.append(
                f"step {step.name!r}: fn {getattr(step.fn, '__name__', step.fn)!r} has no "
                "docstring — every step must document itself"
            )
        missing = [r for r in step.requires if r not in provided]
        if missing:
            problems.append(
                f"step {step.name!r}: requires {missing} but no earlier step provides it"
            )
        provided.update(step.provides)
    return problems


def run(cfg: dict = CFG, spec: list[Step] | None = None) -> tuple[np.ndarray, dict]:
    """Labels for all 199,319 frames (X outside graded spans) + a summary dict.

    `spec` defaults to the declaration in pipeline_spec.py; validation happens
    before the first step runs.
    """
    if spec is None:
        from pipeline_spec import SPEC
        spec = SPEC
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid pipeline spec:\n  - " + "\n  - ".join(problems))

    ctx = Ctx()
    for step in spec:
        step.fn(ctx, cfg)

    if ctx.labels is None or ctx.summary is None:
        raise ValueError(
            "pipeline produced no labels/summary — steps must provide both "
            f"(labels={ctx.labels is not None}, summary={ctx.summary is not None})"
        )
    return ctx.labels, ctx.summary
