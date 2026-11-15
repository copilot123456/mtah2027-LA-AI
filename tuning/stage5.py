"""Stage 5 — the pre-declared P7 pass (dossier §7, 2026-11-14): units-corrected bed,
dead-ball term grid G7, the as-shipped cell, and the organisers' baselines (B1).

    python -m tuning.stage5 --smoke     # plumbing: 1 match, 4 cells, all four mask kinds
    python -m tuning.stage5             # full pass: 50 cells x 7 matches x 4 mask kinds

The decoder cell is the frozen one; only the dead-ball term varies. Each (match, mask
kind) task scores every cell on BOTH label sets and the four baseline variants. Reading
the matrices and applying the U1 / D1 rules is `tuning.gate5` — never here.
"""

from __future__ import annotations

import argparse  # CLI flags (--smoke, --workers)
import sys  # sys.path bootstrap below
import time  # runtime stamps in progress logs
from itertools import product  # grid cross-product
from pathlib import Path  # path objects

import numpy as np  # array maths
import pandas as pd  # CSV read/write

ROOT = Path(__file__).resolve().parents[1]  # repo root
for p in (str(ROOT), str(ROOT / "evaluation")):  # expose repo + evaluation to imports
    if p not in sys.path:  # avoid duplicate entries
        sys.path.insert(0, p)  # make path importable

import possession_metrics as pm  # noqa: E402  # official scoring helpers
from config import CFG  # noqa: E402  # pitch and decoder config
from tuning.idsse import MATCH_IDS, derive_letters_controlled, load_match  # noqa: E402  # IDSSE loader + label sets
from tuning.simulate import (empirical_run_lengths, mask_ball, mask_ball_dead_cover,  # noqa: E402  # masks used here
                             phase_weights)
from tuning.stage4 import cues_for, load_tackle_rows  # noqa: E402  # shared cue helpers
from tuning.sweep import MASK_SEED, _LETTERS, _score  # noqa: E402  # seed, letters, scoring

D_PARAMS = ["d_w_long_missing", "d_missing_run_s", "d_w_low_activity", "d_activity_thr_ms"]  # G7's four params
G7 = {  # grid values for each dead-ball term parameter
    "d_w_long_missing": [0.0, 0.5, 1.0, 2.0],  # weight: long no-ball run => dead
    "d_missing_run_s": [3.0, 10.0],  # no-ball seconds that saturate that feature
    "d_w_low_activity": [0.0, 0.5, 1.0, 2.0],  # weight: low player activity => dead
    "d_activity_thr_ms": [1.0, 1.5],  # mean speed (m/s) counting as low activity
}
# The configuration that produced the committed submission before P7.
AS_SHIPPED = {"d_w_long_missing": 0.5, "d_missing_run_s": 10.0,
              "d_w_low_activity": 0.5, "d_activity_thr_ms": 2.0}  # kept as its own cell
# Canonical value of a parameter whose weight is zero (the parameter is then inert).
_INERT = {"d_missing_run_s": 10.0, "d_activity_thr_ms": 1.0}  # stand-in for off params
FROZEN_DECODER = {"switch_pen": 2.0, "dead_pen": 5.0, "sigma_m": 1.5,  # decoder cell held fixed
                  "min_dwell_s": 2.0, "switch_support_s": 1.5}
KINDS = ("uniform", "phase_a2", "cover_q50", "cover_q90")  # the four mask kinds here
COVER_Q = {"cover_q50": 0.5, "cover_q90": 0.9}  # dead-cover probability per kind
BASELINES = (("baseline_raw", 0), ("baseline_smooth25", 25),  # organisers' B1 baselines
             ("baseline_smooth50", 50), ("baseline_smooth125", 125))  # name -> window
RESULTS = ROOT / "tuning" / "results"  # stage matrices are written here
HOME, AWAY, NOBODY = 0, 1, -1  # team codes used by the baseline


def canonical(cell: dict) -> dict:
    """Inert parameters collapse to one value, so a zero weight is counted once."""
    out = dict(cell)  # copy so the input cell stays clean
    if out["d_w_long_missing"] == 0.0:  # weight off -> feature inert
        out["d_missing_run_s"] = _INERT["d_missing_run_s"]  # collapse to one canonical value
    if out["d_w_low_activity"] == 0.0:  # weight off -> feature inert
        out["d_activity_thr_ms"] = _INERT["d_activity_thr_ms"]  # collapse to one canonical value
    return out  # cell with inert values collapsed


def g7_cells() -> list[dict]:
    """The 49 unique G7 cells followed by the as-shipped cell."""
    seen, cells = set(), []  # dedupe keys + ordered result
    for combo in product(*(G7[p] for p in D_PARAMS)):  # full G7 cross-product
        cell = canonical(dict(zip(D_PARAMS, combo)))  # name it, then collapse inert params
        key = tuple(cell[p] for p in D_PARAMS)  # dedupe key after collapsing
        if key not in seen:  # collapsing can duplicate cells
            seen.add(key)  # remember this key
            cells.append(cell)  # store the unique cell
    cells.append(dict(AS_SHIPPED))  # as-shipped cell always included
    return cells  # unique cells plus as-shipped


# ---------------- organisers' baseline (starter/baseline.py, on external arrays) ----------

def nearest_codes(ball: np.ndarray, players: np.ndarray, slot_team: np.ndarray) -> np.ndarray:
    """Team of the nearest visible player in metres; NOBODY where the ball is missing."""
    out = np.full(len(ball), NOBODY, dtype=np.int8)  # default: nobody seen
    idx = np.flatnonzero(~np.isnan(ball).any(axis=1))  # frames with a visible ball
    length, width = CFG["pitch_length_m"], CFG["pitch_width_m"]  # metres scaling
    for a in range(0, len(idx), 20_000):  # chunked to bound memory
        sl = idx[a:a + 20_000]  # frames in this chunk
        d = (((players[sl, :, 0] - ball[sl, None, 0]) * length) ** 2  # squared distance in metres
             + ((players[sl, :, 1] - ball[sl, None, 1]) * width) ** 2)
        d = np.where(np.isnan(d), np.inf, d)  # missing players can never be nearest
        j = np.argmin(d, axis=1)  # nearest player index per frame
        ok = np.isfinite(d[np.arange(len(sl)), j])  # frames that have any player
        out[sl[ok]] = slot_team[j[ok]]  # assign that player's team
    return out  # team code per frame


def smooth(codes: np.ndarray, window: int) -> np.ndarray:
    """Centred majority vote over +/- `window` frames; D where nobody is seen; ties home."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D")).astype(object)  # raw mapping
    home_c = np.concatenate(([0], np.cumsum(codes == HOME)))  # prefix count of home votes
    away_c = np.concatenate(([0], np.cumsum(codes == AWAY)))  # prefix count of away votes
    i = np.arange(len(codes))  # one index per frame
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)  # window bounds
    home, away = home_c[hi] - home_c[lo], away_c[hi] - away_c[lo]  # votes inside each window
    labels = np.full(len(codes), "D", dtype=object)  # default: nobody seen
    seen = (home + away) > 0  # frames with at least one vote
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")  # majority, ties go home
    return labels  # smoothed H/A/D labels


# ---------------- one task ----------------

def build_mask(case, kind: str, run_lengths: np.ndarray, tackle_rows: np.ndarray):
    rng = np.random.default_rng(MASK_SEED + sum(ord(c) for c in case.match_id))  # seeded per match
    if kind == "uniform":
        return mask_ball(case.ball, run_lengths, rng)  # uniform placement
    if kind == "phase_a2":
        return mask_ball(case.ball, run_lengths, rng,  # phase-weighted placement
                         weights=phase_weights(cues_for(case, tackle_rows), 2.0))  # α=2
    if kind in COVER_Q:
        return mask_ball_dead_cover(case.ball, case.letters, run_lengths, rng, COVER_Q[kind])  # dead cover
    raise ValueError(kind)  # unknown mask kind


def run_task5(args) -> tuple[list[dict], list[dict], dict]:
    """All cells + baselines for one (match, mask kind). Runs in a worker process."""
    match_id, kind, cells, run_lengths = args  # unpack the worker task
    t0 = time.time()  # start the runtime stamp
    case = load_match(match_id)  # cached match arrays
    if case.info.get("speed_units") != "m/s":
        raise ValueError(f"{match_id}: cache is not in m/s — run the loader once first")  # units check
    masked, mask_stats = build_mask(case, kind, run_lengths, load_tackle_rows(match_id))  # apply mask kind

    from src.ball_fill import build_ball_track  # gap-fill step on the masked ball
    from src.decoder import decode  # the possession decoder
    from src.emission import build_emission  # emission model builder

    base = {**CFG, **FROZEN_DECODER, "fill_long_gap_s": 0.0}  # frozen cell, fill off
    track = build_ball_track(masked, case.half, base)  # ball track from masked input
    gt = {"orig": pm.label_codes(case.letters),  # orig label set (per-touch)
          "aligned": pm.label_codes(derive_letters_controlled(case.letters, hold_s=3.0))}  # aligned (E1)

    in_match = case.letters != "X"  # frames counted in the mask stats
    dead = case.letters == "D"  # dead-ball frames
    miss = np.isnan(masked).any(axis=1)  # masked frames
    mask_row = {"match_id": match_id, "kind": kind,  # dead-vs-miss contingency stats
                "masked_fraction": float(miss[in_match].mean()),
                "miss_given_dead": float(miss[in_match & dead].mean()),
                "miss_given_alive": float(miss[in_match & ~dead].mean()),
                "dead_given_miss": float(dead[in_match & miss].mean())}

    def scored(pred_letters) -> dict:
        codes = pm.label_codes(pred_letters)  # letters -> label codes
        row = {}  # scores accumulate here
        for lab, gt_codes in gt.items():  # score against both label sets
            s = _score(gt_codes, codes)  # score one label set
            row[f"spell_f1_{lab}"] = s["spell_f1"]  # store spell F1
            row[f"macro_f1_{lab}"] = s["macro_f1"]  # store macro F1
        return row  # four F1 scores

    rows = []  # cell rows accumulate here
    for cell in cells:
        cfg = {**base, **cell}  # frozen base plus this dead-ball cell
        em = build_emission(track, case.players, case.players_s, case.slot_team, cfg)  # emission for this config
        decoded = decode(em, track.source, case.half, cfg)  # decode with this cell
        pred = _LETTERS[np.asarray(decoded.labels, dtype=np.int8) + 1]  # codes -> letters
        rows.append({"match_id": match_id, "kind": kind, **cell, **scored(pred),
                     "pred_d_share": float((pred[in_match] == "D").mean()),  # decoded D share
                     "runtime_s": round(time.time() - t0, 1)})

    codes = nearest_codes(masked, case.players, case.slot_team)  # baseline team labels
    base_rows = [{"match_id": match_id, "kind": kind, "system": name,  # one row per baseline
                  **scored(smooth(codes, window))} for name, window in BASELINES]
    return rows, base_rows, mask_row  # three result groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # help = docstring line 1
    parser.add_argument("--smoke", action="store_true",  # small plumbing run
                        help="1 match, 4 cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=6)  # pool size
    args = parser.parse_args()  # parsed command-line flags

    from src import data as D  # deferred: heavy import

    arrays = D.load_arrays()  # provided arrays, only for run lengths here
    run_lengths = empirical_run_lengths(np.isnan(arrays.ball).any(axis=1) & arrays.graded)  # empirical runs
    del arrays  # free memory before starting the pool

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS  # 1 match in smoke, all 7 otherwise
    cells = g7_cells()  # build the full cell list
    if args.smoke:
        cells = [cells[0], cells[24], cells[48], cells[-1]]  # 4 strided smoke cells
    for mid in matches:                               # warm caches BEFORE the pool
        load_match(mid)  # preload match arrays
        load_tackle_rows(mid)  # preload tackle rows
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(cells)} cells each", flush=True)  # show workload

    tasks = [(m, k, cells, run_lengths) for m in matches for k in KINDS]  # one task per pair
    from multiprocessing import Pool  # deferred: process pool

    t0 = time.time()  # total runtime starts here
    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}  # cell rows grouped by kind
    base_rows: list[dict] = []  # baseline rows across all tasks
    mask_rows: list[dict] = []  # one mask-stats row per task
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        for i, (rows, brows, mrow) in enumerate(pool.imap_unordered(run_task5, tasks), 1):
            by_kind[mrow["kind"]].extend(rows)  # collect this task's cell rows
            base_rows.extend(brows)  # collect this task's baseline rows
            mask_rows.append(mrow)  # collect this task's mask stats
            print(f"[{i}/{len(tasks)}] {mrow['match_id']} {mrow['kind']} "  # progress counter
                  f"done at {time.time() - t0:.0f}s", flush=True)

    out_dir = Path("/tmp/stage5_smoke") if args.smoke else RESULTS  # smoke stays out of results/
    out_dir.mkdir(parents=True, exist_ok=True)  # create the output dir
    order = {m: i for i, m in enumerate(MATCH_IDS)}  # match order for sorting
    for kind, rows in by_kind.items():
        df = pd.DataFrame(rows)  # rows for this kind
        df = df.sort_values(D_PARAMS + ["match_id"], kind="stable",  # params first,
                            key=lambda s: s.map(order) if s.name == "match_id" else s)  # then match order
        df.to_csv(out_dir / f"stage5_{kind}.csv", index=False)  # write this kind's matrix
        print(f"wrote stage5_{kind}.csv ({len(df)} rows)", flush=True)  # confirm file written
    pd.DataFrame(base_rows).sort_values(["kind", "system", "match_id"]).to_csv(
        out_dir / "stage5_baselines.csv", index=False)  # baseline matrix
    pd.DataFrame(mask_rows).sort_values(["kind", "match_id"]).to_csv(
        out_dir / "stage5_masks.csv", index=False)  # mask statistics
    print(f"total {time.time() - t0:.0f}s", flush=True)  # total runtime
    return 0  # zero means success


if __name__ == "__main__":  # module entry point
    sys.exit(main())  # run as a script
