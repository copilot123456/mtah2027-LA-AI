"""Own-output change between two submission files — a bucket, never a count.

    python -m tuning.output_change OLD.csv NEW.csv

Prints the share of graded frames whose label differs as one of five buckets
(dossier §7 P7: < 2 % · 2-5 % · 5-10 % · 10-20 % · > 20 %).
"""

from __future__ import annotations   # postponed annotation evaluation

import sys                # argv handling and exit codes
from pathlib import Path  # repo-root path objects

import pandas as pd   # reading submission CSVs

ROOT = Path(__file__).resolve().parents[1]   # repo root
if str(ROOT) not in sys.path:                # make the repo importable
    sys.path.insert(0, str(ROOT))   # repo root enters the import path

from tuning.kickoff_check import bucket  # noqa: E402   # shared bucket formatting


def changed_share(old_path: str, new_path: str) -> float:
    from src.data import graded_mask   # which frames are graded

    old = pd.read_csv(old_path)["label"].to_numpy()   # labels of the old file
    new = pd.read_csv(new_path)["label"].to_numpy()   # labels of the new file
    if len(old) != len(new):
        raise ValueError("files differ in length")   # cannot align unequal files
    graded = graded_mask(len(old))   # restrict to graded frames
    return float((old != new)[graded].mean())   # share of graded labels that differ


def main() -> int:
    if len(sys.argv) != 3:   # need exactly OLD and NEW paths
        print(__doc__)       # show usage
        return 2             # usage-error exit code
    print(f"graded frames changing label: {bucket(changed_share(sys.argv[1], sys.argv[2]))}")   # bucket only
    return 0   # report clean completion


if __name__ == "__main__":   # run only as a script
    raise SystemExit(main())   # exit with main's code
