"""Own-output change between two submission files — a bucket, never a count.

    python -m tuning.output_change OLD.csv NEW.csv

Prints the share of graded frames whose label differs as one of five buckets
(dossier §7 P7: < 2 % · 2-5 % · 5-10 % · 10-20 % · > 20 %).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.kickoff_check import bucket  # noqa: E402


def changed_share(old_path: str, new_path: str) -> float:
    from src.data import graded_mask

    old = pd.read_csv(old_path)["label"].to_numpy()
    new = pd.read_csv(new_path)["label"].to_numpy()
    if len(old) != len(new):
        raise ValueError("files differ in length")
    graded = graded_mask(len(old))
    return float((old != new)[graded].mean())


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    print(f"graded frames changing label: {bucket(changed_share(sys.argv[1], sys.argv[2]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
