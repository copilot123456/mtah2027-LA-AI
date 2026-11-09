"""Stage 4 — the pre-declared E2+E3+E11 combined pass (dossier §7, 2026-11-09).

    python -m tuning.stage4 --smoke     # plumbing: 1 match, 6 cells, all four mask kinds
    python -m tuning.stage4             # full pass: 144 cells x 7 matches x 4 runs

Four decode runs (uniform / phase α=2 / phase α=3 / phase α=2 with fill on), each
scored on BOTH label sets (original IDSSE letters and E1-aligned T=3 s) — four decision
tables plus the fill (E11) and α=3 sensitivity twins, exactly as pinned. Selection and
the Gate-1 read happen on the committed matrices in a separate step, never here.

Mask statistics (pre-registered §7) land in `stage4_masks.csv`: overall masked share,
plus inside/outside splits for each cue and any-cue — both the raw NaN share and the
share of *newly removed* frames (finite before, NaN after).
"""

from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
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
from tuning.idsse import (IDSSE_DIR, MATCH_IDS, derive_letters_controlled,
                          load_match)  # noqa: E402
from tuning.simulate import (  # noqa: E402
    crowding_cue,
    empirical_run_lengths,
    event_cue,
    mask_ball,
    map_event_rows,
    phase_weights,
    possession_change_cue,
)
from tuning.sweep import MASK_SEED, PARAMS, _LETTERS, _score  # noqa: E402

# G6 (dossier §7): 3 x 3 x 2 x 4 x 2 = 144; frozen cell (2 / 5 / 1.5 / 2.0 / 1.5) inside.
GRID6 = {
    "switch_pen": [1.0, 2.0, 4.0],
    "dead_pen": [5.0, 8.0, 12.0],
    "sigma_m": [1.5, 2.0],
    "min_dwell_s": [1.0, 1.5, 2.0, 3.0],
    "switch_support_s": [1.0, 1.5],
}
KINDS = ("uniform", "phase_a2", "phase_a3", "phase_a2_fill")
RESULTS = ROOT / "tuning" / "results"
CACHE = ROOT / "data" / ".cache"


def grid6_points() -> list[dict]:
    return [dict(zip(PARAMS, combo))
            for combo in product(*(GRID6[p] for p in PARAMS))]


def _tackle_events(xml_path: Path) -> tuple[list[tuple[float, int]], list[float]]:
    """(anchors: epoch_s from StartFrame events, epochs: all TacklingGame parent times)."""
    tree = ET.parse(xml_path)
    anchors: list[tuple[float, int]] = []
    epochs: list[float] = []
    for ev in tree.getroot().iter("Event"):
        attrs = ev.attrib
        if any(c.tag == "TacklingGame" for c in ev):
            epochs.append(pd.Timestamp(attrs["EventTime"]).timestamp())
        if "StartFrame" in attrs:
            anchors.append((pd.Timestamp(attrs["EventTime"]).timestamp(),
                            int(attrs["StartFrame"])))
    return anchors, epochs


def load_tackle_rows(match_id: str) -> np.ndarray:
    """TacklingGame rows for one IDSSE match; kloppy+XML once, then a local npy cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"tackles_{match_id}.npy"
    if cache.exists():
        return np.load(cache)
    xmls = sorted(IDSSE_DIR.glob(f"*events_raw*{match_id}.xml"))
    if not xmls:
        raise FileNotFoundError(f"events XML for {match_id} under {IDSSE_DIR}")
    anchors, epochs = _tackle_events(xmls[0])

    from kloppy import sportec

    from tuning.idsse import _files
    meta_path, raw_path = _files(match_id)
    ds = sportec.load_tracking(meta_data=meta_path, raw_data=raw_path, only_alive=False)
    df = ds.to_df()
    rows = map_event_rows(
        df["timestamp"].to_numpy(dtype=np.float64),
        df["frame_id"].to_numpy(),
        df["period_id"].to_numpy(),
        anchors, epochs)
    kept = rows[rows >= 0]
    print(f"{match_id}: tackles {len(kept)}/{len(epochs)} mapped, "
          f"{len(anchors)} anchors", flush=True)
    np.save(cache, kept)
    return kept


def cues_for(case, tackle_rows: np.ndarray) -> list[np.ndarray]:
    return [
        possession_change_cue(case.letters),
        crowding_cue(case.players, case.ball,
                     length_m=CFG["pitch_length_m"], width_m=CFG["pitch_width_m"]),
        event_cue(tackle_rows, len(case.letters)),
    ]


def run_task4(args) -> tuple[list[dict], dict]:
    """All grid cells for one (match, mask kind). Runs in a worker process."""
    match_id, kind, grid, run_lengths = args
    t0 = time.time()
    case = load_match(match_id)
    tackle_rows = load_tackle_rows(match_id)          # pre-warmed by main()
    cues = cues_for(case, tackle_rows)

    rng = np.random.default_rng(MASK_SEED + sum(ord(c) for c in match_id))
    if kind == "uniform":
        masked, mask_stats = mask_ball(case.ball, run_lengths, rng)
    else:
        alpha = 3.0 if kind == "phase_a3" else 2.0
        masked, mask_stats = mask_ball(case.ball, run_lengths, rng,
                                       weights=phase_weights(cues, alpha))

    base = dict(CFG)
    if kind != "phase_a2_fill":                       # fill on only for the E11 twin
        base["fill_long_gap_s"] = 0.0

    from src.ball_fill import build_ball_track
    from src.decoder import decode
    from src.emission import build_emission

    track = build_ball_track(masked, case.half, base)
    gt_orig = pm.label_codes(case.letters)
    gt_aligned = pm.label_codes(derive_letters_controlled(case.letters, hold_s=3.0))
    runs_orig, runs_aligned = pm.valid_runs(gt_orig), pm.valid_runs(gt_aligned)
    n_gt_orig = len(pm.possession_spells(gt_orig, runs_orig))
    n_gt_aligned = len(pm.possession_spells(gt_aligned, runs_aligned))

    # pre-registered mask statistics (§7): NaN share and newly-removed share, in/out cues
    was_finite = ~np.isnan(case.ball).any(axis=1)
    now_nan = np.isnan(masked).any(axis=1)
    removed = was_finite & now_nan
    any_cue = cues[0] | cues[1] | cues[2]
    mask_row = {"match_id": match_id, "kind": kind, **mask_stats,
                "nan_share": float(now_nan.mean()),
                "removed_share": float(removed.mean())}
    for i, cue in enumerate(cues, start=1):
        mask_row[f"cue{i}_nan_in"] = float(now_nan[cue].mean()) if cue.any() else np.nan
        mask_row[f"cue{i}_nan_out"] = float(now_nan[~cue].mean()) if (~cue).any() else np.nan
        mask_row[f"cue{i}_removed_in"] = float(removed[cue].mean()) if cue.any() else np.nan
        mask_row[f"cue{i}_removed_out"] = float(removed[~cue].mean()) if (~cue).any() else np.nan
        mask_row[f"cue{i}_frame_share"] = float(cue.mean())
    mask_row["anycue_removed_in"] = float(removed[any_cue].mean()) if any_cue.any() else np.nan
    mask_row["anycue_removed_out"] = float(removed[~any_cue].mean()) if (~any_cue).any() else np.nan

    rows: list[dict] = []
    sigmas = sorted({float(c["sigma_m"]) for c in grid})
    for sigma in sigmas:
        em = build_emission(track, case.players, case.players_s, case.slot_team,
                            {**base, "sigma_m": sigma})
        for cell in grid:
            if float(cell["sigma_m"]) != sigma:
                continue
            cfg = {**base, **cell}
            decoded = decode(em, track.source, case.half, cfg)
            pred = _LETTERS[np.asarray(decoded.labels, dtype=np.int8) + 1]
            pred_codes = pm.label_codes(pred)
            s_orig = _score(gt_orig, pred_codes)
            s_aligned = _score(gt_aligned, pred_codes)
            rows.append({
                "match_id": match_id, "kind": kind, **cell,
                "spell_f1_orig": s_orig["spell_f1"], "macro_f1_orig": s_orig["macro_f1"],
                "spell_f1_aligned": s_aligned["spell_f1"],
                "macro_f1_aligned": s_aligned["macro_f1"],
                "n_gt_orig": n_gt_orig, "n_gt_aligned": n_gt_aligned,
                "masked_fraction": round(mask_stats["masked_fraction"], 4),
                "runtime_s": round(time.time() - t0, 1),
            })
    return rows, mask_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="1 match, 6 strided cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    from src import data as D

    arrays = D.load_arrays()
    missing_graded = np.isnan(arrays.ball).any(axis=1) & arrays.graded
    run_lengths = empirical_run_lengths(missing_graded)
    print(f"empirical missing runs: n={len(run_lengths)} "
          f"median={int(np.median(run_lengths))} max={int(run_lengths.max())}", flush=True)

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS
    grid = grid6_points()[::24][:6] if args.smoke else grid6_points()
    for mid in matches:                               # warm caches BEFORE the pool
        load_match(mid)
        load_tackle_rows(mid)
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(grid)} cells each",
          flush=True)

    tasks = [(m, k, grid, run_lengths) for m in matches for k in KINDS]
    from multiprocessing import Pool

    t0 = time.time()
    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}
    mask_rows: list[dict] = []
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        for i, (rows, mask_row) in enumerate(pool.imap_unordered(run_task4, tasks), 1):
            by_kind[rows[0]["kind"]].extend(rows)
            mask_rows.append(mask_row)
            print(f"[{i}/{len(tasks)}] {mask_row['match_id']} {mask_row['kind']} "
                  f"done at {time.time() - t0:.0f}s", flush=True)

    out_dir = Path("/tmp/stage4_smoke") if args.smoke else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, rows in by_kind.items():
        path = out_dir / f"stage4_{kind}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"wrote {path} ({len(rows)} rows)", flush=True)
    masks_path = out_dir / "stage4_masks.csv"
    pd.DataFrame(mask_rows).to_csv(masks_path, index=False)
    print(f"wrote {masks_path} ({len(mask_rows)} rows)", flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
