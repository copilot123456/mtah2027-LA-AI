"""Stage 5 — the pre-declared P7 pass (dossier §7, 2026-11-14): units-corrected bed,
dead-ball term grid G7, the as-shipped cell, and the organisers' baselines (B1).

    python -m tuning.stage5 --smoke     # plumbing: 1 match, 4 cells, all four mask kinds
    python -m tuning.stage5             # full pass: 50 cells x 7 matches x 4 mask kinds

The decoder cell is the frozen one; only the dead-ball term varies. Each (match, mask
kind) task scores every cell on BOTH label sets and the four baseline variants. Reading
the matrices and applying the U1 / D1 rules is `tuning.gate5` — never here.
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import possession_metrics as pm  # noqa: E402
from config import CFG  # noqa: E402
from tuning.idsse import MATCH_IDS, derive_letters_controlled, load_match  # noqa: E402
from tuning.simulate import (empirical_run_lengths, mask_ball, mask_ball_dead_cover,  # noqa: E402
                             phase_weights)
from tuning.stage4 import cues_for, load_tackle_rows  # noqa: E402
from tuning.sweep import MASK_SEED, _LETTERS, _score  # noqa: E402

D_PARAMS = ["d_w_long_missing", "d_missing_run_s", "d_w_low_activity", "d_activity_thr_ms"]
G7 = {
    "d_w_long_missing": [0.0, 0.5, 1.0, 2.0],
    "d_missing_run_s": [3.0, 10.0],
    "d_w_low_activity": [0.0, 0.5, 1.0, 2.0],
    "d_activity_thr_ms": [1.0, 1.5],
}
# The configuration that produced the committed submission before P7.
AS_SHIPPED = {"d_w_long_missing": 0.5, "d_missing_run_s": 10.0,
              "d_w_low_activity": 0.5, "d_activity_thr_ms": 2.0}
# Canonical value of a parameter whose weight is zero (the parameter is then inert).
_INERT = {"d_missing_run_s": 10.0, "d_activity_thr_ms": 1.0}
FROZEN_DECODER = {"switch_pen": 2.0, "dead_pen": 5.0, "sigma_m": 1.5,
                  "min_dwell_s": 2.0, "switch_support_s": 1.5}
KINDS = ("uniform", "phase_a2", "cover_q50", "cover_q90")
COVER_Q = {"cover_q50": 0.5, "cover_q90": 0.9}
BASELINES = (("baseline_raw", 0), ("baseline_smooth25", 25),
             ("baseline_smooth50", 50), ("baseline_smooth125", 125))
RESULTS = ROOT / "tuning" / "results"
HOME, AWAY, NOBODY = 0, 1, -1


def canonical(cell: dict) -> dict:
    """Inert parameters collapse to one value, so a zero weight is counted once."""
    out = dict(cell)
    if out["d_w_long_missing"] == 0.0:
        out["d_missing_run_s"] = _INERT["d_missing_run_s"]
    if out["d_w_low_activity"] == 0.0:
        out["d_activity_thr_ms"] = _INERT["d_activity_thr_ms"]
    return out


def g7_cells() -> list[dict]:
    """The 49 unique G7 cells followed by the as-shipped cell."""
    seen, cells = set(), []
    for combo in product(*(G7[p] for p in D_PARAMS)):
        cell = canonical(dict(zip(D_PARAMS, combo)))
        key = tuple(cell[p] for p in D_PARAMS)
        if key not in seen:
            seen.add(key)
            cells.append(cell)
    cells.append(dict(AS_SHIPPED))
    return cells


# ---------------- organisers' baseline (starter/baseline.py, on external arrays) ----------

def nearest_codes(ball: np.ndarray, players: np.ndarray, slot_team: np.ndarray) -> np.ndarray:
    """Team of the nearest visible player in metres; NOBODY where the ball is missing."""
    out = np.full(len(ball), NOBODY, dtype=np.int8)
    idx = np.flatnonzero(~np.isnan(ball).any(axis=1))
    length, width = CFG["pitch_length_m"], CFG["pitch_width_m"]
    for a in range(0, len(idx), 20_000):
        sl = idx[a:a + 20_000]
        d = (((players[sl, :, 0] - ball[sl, None, 0]) * length) ** 2
             + ((players[sl, :, 1] - ball[sl, None, 1]) * width) ** 2)
        d = np.where(np.isnan(d), np.inf, d)
        j = np.argmin(d, axis=1)
        ok = np.isfinite(d[np.arange(len(sl)), j])
        out[sl[ok]] = slot_team[j[ok]]
    return out


def smooth(codes: np.ndarray, window: int) -> np.ndarray:
    """Centred majority vote over +/- `window` frames; D where nobody is seen; ties home."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D")).astype(object)
    home_c = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_c = np.concatenate(([0], np.cumsum(codes == AWAY)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    home, away = home_c[hi] - home_c[lo], away_c[hi] - away_c[lo]
    labels = np.full(len(codes), "D", dtype=object)
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


# ---------------- one task ----------------

def build_mask(case, kind: str, run_lengths: np.ndarray, tackle_rows: np.ndarray):
    rng = np.random.default_rng(MASK_SEED + sum(ord(c) for c in case.match_id))
    if kind == "uniform":
        return mask_ball(case.ball, run_lengths, rng)
    if kind == "phase_a2":
        return mask_ball(case.ball, run_lengths, rng,
                         weights=phase_weights(cues_for(case, tackle_rows), 2.0))
    if kind in COVER_Q:
        return mask_ball_dead_cover(case.ball, case.letters, run_lengths, rng, COVER_Q[kind])
    raise ValueError(kind)


def run_task5(args) -> tuple[list[dict], list[dict], dict]:
    """All cells + baselines for one (match, mask kind). Runs in a worker process."""
    match_id, kind, cells, run_lengths = args
    t0 = time.time()
    case = load_match(match_id)
    if case.info.get("speed_units") != "m/s":
        raise ValueError(f"{match_id}: cache is not in m/s — run the loader once first")
    masked, mask_stats = build_mask(case, kind, run_lengths, load_tackle_rows(match_id))

    from src.ball_fill import build_ball_track
    from src.decoder import decode
    from src.emission import build_emission

    base = {**CFG, **FROZEN_DECODER, "fill_long_gap_s": 0.0}
    track = build_ball_track(masked, case.half, base)
    gt = {"orig": pm.label_codes(case.letters),
          "aligned": pm.label_codes(derive_letters_controlled(case.letters, hold_s=3.0))}

    in_match = case.letters != "X"
    dead = case.letters == "D"
    miss = np.isnan(masked).any(axis=1)
    mask_row = {"match_id": match_id, "kind": kind,
                "masked_fraction": float(miss[in_match].mean()),
                "miss_given_dead": float(miss[in_match & dead].mean()),
                "miss_given_alive": float(miss[in_match & ~dead].mean()),
                "dead_given_miss": float(dead[in_match & miss].mean())}

    def scored(pred_letters) -> dict:
        codes = pm.label_codes(pred_letters)
        row = {}
        for lab, gt_codes in gt.items():
            s = _score(gt_codes, codes)
            row[f"spell_f1_{lab}"] = s["spell_f1"]
            row[f"macro_f1_{lab}"] = s["macro_f1"]
        return row

    rows = []
    for cell in cells:
        cfg = {**base, **cell}
        em = build_emission(track, case.players, case.players_s, case.slot_team, cfg)
        decoded = decode(em, track.source, case.half, cfg)
        pred = _LETTERS[np.asarray(decoded.labels, dtype=np.int8) + 1]
        rows.append({"match_id": match_id, "kind": kind, **cell, **scored(pred),
                     "pred_d_share": float((pred[in_match] == "D").mean()),
                     "runtime_s": round(time.time() - t0, 1)})

    codes = nearest_codes(masked, case.players, case.slot_team)
    base_rows = [{"match_id": match_id, "kind": kind, "system": name,
                  **scored(smooth(codes, window))} for name, window in BASELINES]
    return rows, base_rows, mask_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="1 match, 4 cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from src import data as D

    arrays = D.load_arrays()
    run_lengths = empirical_run_lengths(np.isnan(arrays.ball).any(axis=1) & arrays.graded)
    del arrays

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS
    cells = g7_cells()
    if args.smoke:
        cells = [cells[0], cells[24], cells[48], cells[-1]]
    for mid in matches:                               # warm caches BEFORE the pool
        load_match(mid)
        load_tackle_rows(mid)
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(cells)} cells each", flush=True)

    tasks = [(m, k, cells, run_lengths) for m in matches for k in KINDS]
    from multiprocessing import Pool

    t0 = time.time()
    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}
    base_rows: list[dict] = []
    mask_rows: list[dict] = []
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        for i, (rows, brows, mrow) in enumerate(pool.imap_unordered(run_task5, tasks), 1):
            by_kind[mrow["kind"]].extend(rows)
            base_rows.extend(brows)
            mask_rows.append(mrow)
            print(f"[{i}/{len(tasks)}] {mrow['match_id']} {mrow['kind']} "
                  f"done at {time.time() - t0:.0f}s", flush=True)

    out_dir = Path("/tmp/stage5_smoke") if args.smoke else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {m: i for i, m in enumerate(MATCH_IDS)}
    for kind, rows in by_kind.items():
        df = pd.DataFrame(rows)
        df = df.sort_values(D_PARAMS + ["match_id"], kind="stable",
                            key=lambda s: s.map(order) if s.name == "match_id" else s)
        df.to_csv(out_dir / f"stage5_{kind}.csv", index=False)
        print(f"wrote stage5_{kind}.csv ({len(df)} rows)", flush=True)
    pd.DataFrame(base_rows).sort_values(["kind", "system", "match_id"]).to_csv(
        out_dir / "stage5_baselines.csv", index=False)
    pd.DataFrame(mask_rows).sort_values(["kind", "match_id"]).to_csv(
        out_dir / "stage5_masks.csv", index=False)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
