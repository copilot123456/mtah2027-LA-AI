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

from __future__ import annotations    # postponed evaluation of annotations

from dataclasses import dataclass, field  # state containers below
from typing import Callable           # step function type hint

import numpy as np                   # labels are numpy arrays

from config import CFG                # default configuration


@dataclass
class Ctx:
    """State handed from step to step; `extra` is free space for new steps."""

    arrays: object = None            # aligned match arrays (ATD, players, spans)
    track: object = None             # reconstructed ball track
    emission: object = None          # per-frame H/A/D log-scores
    decoded: object = None           # Viterbi output (labels + switch stats)
    labels: np.ndarray | None = None # final per-frame labels
    summary: dict | None = None      # run statistics for reporting
    extra: dict = field(default_factory=dict)  # scratch space for new steps


@dataclass(frozen=True)
class Step:
    """One pipeline stage: `fn(ctx, cfg) -> None` plus its data-flow declaration."""

    name: str                        # unique step id
    fn: Callable[..., None]          # handler: fn(ctx, cfg) -> None
    requires: tuple[str, ...] = ()   # Ctx fields that must already exist
    provides: tuple[str, ...] = ()   # Ctx fields this step writes


def validate_spec(spec: list[Step]) -> list[str]:
    """Problems with the spec; an empty list means it is valid.

    Runs before any step executes so a mis-declared spec fails fast with the
    offending step named, not with an AttributeError in the middle of a run.
    """
    problems: list[str] = []         # error messages; empty list = valid
    provided: set[str] = set()       # fields available from earlier steps
    seen: set[str] = set()           # step names used so far
    for step in spec:
        if step.name in seen:        # names must be unique
            problems.append(f"step {step.name!r}: duplicate step name")  # record the duplicate
        seen.add(step.name)          # mark name seen
        if not (getattr(step.fn, "__doc__", "") or "").strip():  # fn must self-document
            problems.append(           # record missing docstring
                f"step {step.name!r}: fn {getattr(step.fn, '__name__', step.fn)!r} has no "
                "docstring — every step must document itself"
            )
        missing = [r for r in step.requires if r not in provided]  # unmet requirements
        if missing:                  # required field never provided earlier
            problems.append(           # record unmet requires
                f"step {step.name!r}: requires {missing} but no earlier step provides it"
            )
        provided.update(step.provides)  # publish this step's outputs
    return problems                  # empty list means valid


def run(cfg: dict = CFG, spec: list[Step] | None = None) -> tuple[np.ndarray, dict]:
    """Labels for all 199,319 frames (X outside graded spans) + a summary dict.

    `spec` defaults to the declaration in pipeline_spec.py; validation happens
    before the first step runs.
    """
    if spec is None:
        from pipeline_spec import SPEC  # default spec; imported lazily (it imports this module)
        spec = SPEC                  # use the default spec
    problems = validate_spec(spec)   # check spec before running
    if problems:                     # fail fast, before any step runs
        raise ValueError("invalid pipeline spec:\n  - " + "\n  - ".join(problems))  # refuse bad spec

    ctx = Ctx()                      # fresh state for this run
    for step in spec:                # execute stages in declaration order
        step.fn(ctx, cfg)            # run this stage

    if ctx.labels is None or ctx.summary is None:  # both outputs are mandatory
        raise ValueError(            # outputs missing entirely
            "pipeline produced no labels/summary — steps must provide both "
            f"(labels={ctx.labels is not None}, summary={ctx.summary is not None})"
        )
    return ctx.labels, ctx.summary   # final labels plus run statistics
