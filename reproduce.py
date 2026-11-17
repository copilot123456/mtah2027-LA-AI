"""Regenerate the Viterbi submission from scratch, in one command.

    python reproduce.py              # decode the notebook -> CSV -> validate (no GT opened)
    python reproduce.py --evaluate   # + one ground-truth report run (report only, never selection)
    python reproduce.py --check-deps # print the interpreter and pinned versions, then exit

What runs, in order:
  1. evaluation/run_viterbi_vs_gt.py --submission-only   (execs notebooks/viterbi.ipynb --
     the single source of truth for the decoder -- decodes the features parquet, writes
     evaluation/results/viterbi.csv in the evaluation alphabet L->H);
  2. evaluation/validate_submission.py                   (the official format validator:
     header, 199319 rows, frame 0..199318, labels in {H,A,D,X}, no blanks);
  3. with --evaluate: evaluation/run_viterbi_vs_gt.py    (full run: frame-level report,
     official metrics, timeline figure). Its numbers are a report of a frozen config --
     per the repo convention they never choose the next objective.

Determinism: the decoder path (emissions -> Viterbi -> morphological filters) contains no
random-number generation -- no seed to fix. The only seeded steps in the repo are the
missingness simulations in tuning/ (seeds recorded in their scripts) and the tactical
video exports (seed 42), which are not part of the scored output.

Dependencies are pinned in requirements-lock.txt (the scorer's requirements.txt stays
deliberately unpinned on purpose -- see its header).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = [sys.executable, str(ROOT / "evaluation" / "run_viterbi_vs_gt.py")]
VALIDATE = [sys.executable, str(ROOT / "evaluation" / "validate_submission.py")]
CSV = ROOT / "evaluation" / "results" / "viterbi.csv"
LOCK = ROOT / "requirements-lock.txt"


def check_deps() -> int:
    import importlib.metadata as md
    import platform

    print("python   {}".format(platform.python_version()))
    if not LOCK.exists():
        print("requirements-lock.txt missing")
        return 1
    ok = True
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("python=="):
            continue
        name, want = line.split("==")
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            print("  MISSING  {}".format(name))
            ok = False
            continue
        flag = "ok" if got == want else "DRIFT (locked {}, found {})".format(want, got)
        ok &= got == want
        print("  {:<20} {}".format(name, flag))
    print("dependencies {}".format("match the lock" if ok else "DO NOT match the lock"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evaluate", action="store_true",
                    help="after validating, run the ground-truth report (frozen config)")
    ap.add_argument("--check-deps", action="store_true",
                    help="only print interpreter + locked versions")
    args = ap.parse_args()

    if args.check_deps:
        return check_deps()

    steps = [
        ("regenerate submission from the notebook", RUN + ["--submission-only"]),
        ("validate format (official validator)", VALIDATE + [str(CSV)]),
    ]
    if args.evaluate:
        steps.append(("ground-truth report (report only)", RUN + ["--bootstrap", "1000"]))

    for title, cmd in steps:
        print("\n=== {} ===".format(title))
        print("  $ {}".format(" ".join(cmd[1:])))
        code = subprocess.call(cmd, cwd=ROOT)
        if code != 0:
            print("FAILED (exit {})".format(code))
            return code
    print("\nOK: submission regenerated and validated (deterministic, no RNG in the decoder).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
