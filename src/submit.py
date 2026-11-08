"""Write the submission file and validate it exactly like the official scorer.

    python -m src.submit            # runs the pipeline, writes LA+AI.csv, validates
    python -m src.submit --out x.csv --skip-run --labels cached.npz

Format contract (SUBMISSION.md):
  * header exactly `frame,label`
  * exactly 199,319 data rows, frame 0..199318 in order
  * label in {H, A, D, X}, no blank cells, UTF-8, no index column
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from config import CFG
from src.data import N_FRAMES
from src.decoder import CODE_A, CODE_D, CODE_H, CODE_X

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATOR = os.path.join(ROOT, "evaluation", "validate_submission.py")

DEFAULT_OUT = "LA+AI.csv"
_LETTER = {CODE_H: "H", CODE_A: "A", CODE_D: "D", CODE_X: "X"}


def labels_to_frame_table(labels: np.ndarray) -> pd.DataFrame:
    if len(labels) != N_FRAMES:
        raise ValueError(f"expected {N_FRAMES} labels, got {len(labels)}")
    codes = np.asarray(labels, dtype=np.int8)
    unknown = set(np.unique(codes)) - set(_LETTER)
    if unknown:
        raise ValueError(f"unknown label codes {sorted(unknown)}; expected {sorted(_LETTER)}")
    letters = np.array(["X", "H", "A", "D"], dtype="<U1")  # index by code+1 (CODE_X = -1)
    return pd.DataFrame({"frame": np.arange(len(codes), dtype=np.int64),
                         "label": letters[codes + 1]})


def write_submission(labels: np.ndarray, path: str) -> None:
    table = labels_to_frame_table(labels)
    table.to_csv(path, index=False, lineterminator="\n")


def validate(path: str) -> int:
    """Run the official validator as a subprocess; returns its exit code.

    The validator is located relative to this file, not the current directory, so the
    call works from any cwd (pytest invoked from elsewhere, or an absolute --out path).
    """
    result = subprocess.run(
        [sys.executable, VALIDATOR, os.path.abspath(path)],
        capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    from src.pipeline import run

    labels, summary = run(CFG)
    write_submission(labels, args.out)
    print(f"wrote {args.out}")
    print(f"  segments {summary['n_segments']} (possessions {summary['n_possessions']})"
          f"  median {summary['median_len_s']:.1f}s  sub-second {100 * summary['subsecond_share']:.1f}%"
          f"  D share {100 * summary['d_share']:.1f}%  gated switches {summary['reverted_switches']}")

    if args.no_validate:
        return 0
    return validate(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
