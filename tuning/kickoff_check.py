"""K1 verification (dossier §7 P7, 2026-11-14) — categorical output only.

    python -m tuning.kickoff_check

Checks the provided match's `Kick Off (Start)` tags against the two public documents the
rule rests on, and reports the rule's own-output effect as a bucket. It prints PASS / FAIL
and buckets — never a frame number, a count of labels or any other match value.
"""

from __future__ import annotations   # postponed annotation evaluation

import sys                           # sys.path setup below
from pathlib import Path             # repo-root path objects

import numpy as np                   # array math over frame labels

ROOT = Path(__file__).resolve().parents[1]   # repo root (one level above tuning/)
if str(ROOT) not in sys.path:                # make the repo importable as a package
    sys.path.insert(0, str(ROOT))   # repo root enters the import path

FAQ_SECOND_HALF_PREKICKOFF = 2_934   # reference-docs/FAQ.md ("those 2,934 frames ...")
TOLERANCE = 250                      # 10 s at 25 fps


def bucket(share: float) -> str:
    # map a share to the five report buckets, smallest ceiling first
    for edge, name in ((0.02, "< 2 %"), (0.05, "2-5 %"), (0.10, "5-10 %"), (0.20, "10-20 %")):
        if share < edge:   # first ceiling the share stays under
            return name   # first bucket that fits
    return "> 20 %"   # larger than every listed ceiling


def spans(half: np.ndarray, starts: np.ndarray) -> dict[int, list[int]]:
    """Per half: distance in frames from the half's first frame to each tag inside it."""
    out: dict[int, list[int]] = {}   # half number -> offsets of its tags
    for h in (1, 2):                 # handle each half in turn
        idx = np.flatnonzero(half == h)   # frames labelled as half h
        first, last = int(idx[0]), int(idx[-1])   # graded span of that half
        out[h] = [int(s - first) for s in starts if first <= s <= last]   # in-span tags, measured from span start
    return out   # collected offsets per half


def verdicts(per_half: dict[int, list[int]]) -> dict[str, bool]:
    # pinned checks (dossier §7 P7 K1) — reported, no longer deciding
    one_each = all(len(per_half[h]) == 1 for h in (1, 2))   # exactly one tag per half
    h1_ok = len(per_half[1]) == 1 and per_half[1][0] <= TOLERANCE   # first tag near half start
    h2_ok = (len(per_half[2]) == 1
             and abs(per_half[2][0] - FAQ_SECOND_HALF_PREKICKOFF) <= TOLERANCE)   # second tag near the FAQ figure
    return {"one tag inside each graded half": one_each,   # check name -> pass/fail
            "first-half span <= 250 frames": h1_ok,
            "second-half span within +/-250 frames of the FAQ's 2,934": h2_ok}


def amended_verdicts(per_half: dict[int, list[int]]) -> dict[str, bool]:
    """Operator-approved amendment (dossier §7 P7 K1, 2026-11-14): the first half may carry
    no tag inside its graded span (it is graded from the kick-off itself, FAQ) — the rule is
    then a no-op there. The second-half clause is the pinned one, unchanged."""
    h1_ok = len(per_half[1]) == 0 or (len(per_half[1]) == 1 and per_half[1][0] <= TOLERANCE)   # empty first half now passes
    h2_ok = (len(per_half[2]) == 1
             and abs(per_half[2][0] - FAQ_SECOND_HALF_PREKICKOFF) <= TOLERANCE)   # pinned clause, unchanged
    return {"first half: no tag inside, or span <= 250 frames": h1_ok,   # these verdicts decide
            "second-half span within +/-250 frames of the FAQ's 2,934": h2_ok}


def diagnosis(half: np.ndarray, starts: np.ndarray) -> dict[str, bool]:
    """Why a pinned check failed, still categorical: where the first-half tag sits."""
    idx = np.flatnonzero(half == 1)   # frames of the first half
    first, last = int(idx[0]), int(idx[-1])   # its graded span
    inside = [s for s in starts if first <= s <= last]   # tags within the span
    before = [s for s in starts if s < first]   # tags ahead of the span
    return {"two `Kick Off (Start)` tags overall": len(starts) == 2,
            "no tag inside the first graded half": len(inside) == 0,
            "a tag lies within 250 frames before the first graded frame":
                any(first - s <= TOLERANCE for s in before)}   # early-tag clue


def main() -> int:
    from config import CFG        # frozen pipeline config
    from src import data as D     # array loading
    from src.pipeline import run  # full pipeline, rule toggled via CFG

    arrays = D.load_arrays()   # halves and kick-off tag positions
    checks = verdicts(spans(arrays.half, arrays.kickoff_starts))   # pinned verdicts
    print("pinned checks (dossier §7 P7 K1):")   # pinned section header
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")   # one line per check
    amended = amended_verdicts(spans(arrays.half, arrays.kickoff_starts))   # deciding verdicts
    print("amended checks (operator decision 2026-11-14 — these decide):")   # deciding section header
    for name, ok in amended.items():   # loop over deciding checks
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")   # print each deciding verdict
    all_ok = all(amended.values())   # overall = amended checks only
    print("diagnosis (not part of the pinned verdict):")   # diagnostic section header
    for name, ok in diagnosis(arrays.half, arrays.kickoff_starts).items():
        print(f"  {'yes' if ok else 'no '}  {name}")   # yes/no clues, not verdicts

    off, _ = run({**CFG, "kickoff_rule": False})   # pipeline with the rule disabled
    on, _ = run({**CFG, "kickoff_rule": True})     # same config, rule enabled
    changed = float((off != on)[arrays.graded].mean())   # share of graded labels that flip
    print(f"own-output effect: {bucket(changed)} of graded frames change label")   # bucket only
    print(f"K1 verification: {'PASS' if all_ok else 'FAIL'}")   # final line mirrors verdict
    return 0 if all_ok else 1   # exit code = PASS/FAIL


if __name__ == "__main__":   # run only as a script
    raise SystemExit(main())   # exit with main's code
