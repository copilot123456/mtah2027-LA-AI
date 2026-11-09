"""E1 label statistics (pre-declared: dossier §7 P6, 2026-11-09).

    python -m tuning.e1_label_stats

Original vs challenge-definition labels across all 7 IDSSE matches at the pinned
primary T = 3 s and the reported-only sensitivities T = {2, 5} s. Statistics only —
no scoring, no selection (T is never picked by matching the public window).

Fidelity probe (pre-registered in the dossier): at T = 3 the means must land near the
plan §0 audited figures — 2.49 possessions/min, 13.4 s mean length, 0.03 share under 2 s.
Output: tuning/results/e1_label_stats.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.idsse import MATCH_IDS, load_match, derive_letters_controlled  # noqa: E402

OUT = ROOT / "tuning" / "results" / "e1_label_stats.csv"
FPS = 25
T_GRID = (2.0, 3.0, 5.0)
PROBE_T3 = {"poss_per_min": 2.49, "mean_len_s": 13.4, "under2_share": 0.03}


def alive_runs(letters: np.ndarray) -> np.ndarray:
    """Lengths in seconds of maximal single-team runs — EVALUATION's possession
    definition (a maximal run of H OR of A; team flips split runs; D/X are stops)."""
    runs, i, n = [], 0, len(letters)
    while i < n:
        lab = letters[i]
        if lab not in ("H", "A"):
            i += 1
            continue
        j = i
        while j < n and letters[j] == lab:
            j += 1
        runs.append((j - i) / FPS)
        i = j
    return np.asarray(runs)


def stats_for(letters: np.ndarray, tag: str, match_id: str) -> dict:
    runs = alive_runs(letters)
    in_match = float((letters != "X").sum()) / FPS          # seconds in match
    dead = float((letters == "D").sum())
    alive_frames = float(((letters == "H") | (letters == "A")).sum())
    return {
        "match_id": match_id, "variant": tag,
        "poss_per_min": len(runs) / (in_match / 60.0),
        "mean_len_s": float(runs.mean()) if len(runs) else 0.0,
        "median_len_s": float(np.median(runs)) if len(runs) else 0.0,
        "under2_share": float((runs < 2.0).mean()) if len(runs) else 0.0,
        "dead_share": dead / (dead + alive_frames),
        "n_possessions": len(runs),
    }


def main() -> int:
    rows = []
    for mid in MATCH_IDS:
        case = load_match(mid)
        rows.append(stats_for(case.letters, "original", mid))
        for t in T_GRID:
            ctl = derive_letters_controlled(case.letters, FPS, hold_s=t)
            rows.append(stats_for(ctl, f"T={t:g}s", mid))
        print(f"  {mid} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    means = df.groupby("variant")[["poss_per_min", "mean_len_s", "under2_share", "dead_share"]].mean()
    order = ["original"] + [f"T={t:g}s" for t in T_GRID]
    print("\nmeans over 7 matches:")
    print(means.loc[order].round(4).to_string())

    t3 = means.loc["T=3s"]
    print("\nfidelity probe vs plan §0 (T=3):")
    for k, target in PROBE_T3.items():
        print(f"  {k}: measured {t3[k]:.3f}  | probe {target:.3f}")
    orig = means.loc["original"]
    print(f"dead share invariance: original {orig['dead_share']:.4f} -> "
          + ", ".join(f"{v}: {means.loc[v, 'dead_share']:.4f}" for v in order[1:]))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
