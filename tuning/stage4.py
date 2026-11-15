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

import argparse  # CLI flags (--smoke, --workers)
import sys  # sys.path bootstrap below
import time  # runtime stamps in progress logs
import xml.etree.ElementTree as ET  # parse the IDSSE events XML
from itertools import product  # grid cross-product
from pathlib import Path  # path objects

import numpy as np  # array maths
import pandas as pd  # CSV read/write, timestamps

ROOT = Path(__file__).resolve().parents[1]  # repo root
for p in (str(ROOT), str(ROOT / "evaluation")):  # expose repo + evaluation to imports
    if p not in sys.path:  # avoid duplicate entries
        sys.path.insert(0, p)  # make path importable

import possession_metrics as pm  # noqa: E402  # official scoring helpers
from config import CFG  # noqa: E402  # pitch and decoder config
# IDSSE paths, match ids and the aligned label set
from tuning.idsse import (IDSSE_DIR, MATCH_IDS, derive_letters_controlled,
                          load_match)  # noqa: E402
# masks, phase cues and event-row mapping used by this stage
from tuning.simulate import (  # noqa: E402
    crowding_cue,
    empirical_run_lengths,
    event_cue,
    mask_ball,
    map_event_rows,
    phase_weights,
    possession_change_cue,
)
from tuning.sweep import MASK_SEED, PARAMS, _LETTERS, _score  # noqa: E402  # seed, grid keys, scoring

# G6 (dossier §7): 3 x 3 x 2 x 4 x 2 = 144; frozen cell (2 / 5 / 1.5 / 2.0 / 1.5) inside.
GRID6 = {
    "switch_pen": [1.0, 2.0, 4.0],  # penalty on H↔A switches
    "dead_pen": [5.0, 8.0, 12.0],  # penalty on team↔D switches
    "sigma_m": [1.5, 2.0],  # emission width in metres
    "min_dwell_s": [1.0, 1.5, 2.0, 3.0],  # shortest segment kept (seconds)
    "switch_support_s": [1.0, 1.5],  # brief-switch revert window (seconds)
}
KINDS = ("uniform", "phase_a2", "phase_a3", "phase_a2_fill")  # the four mask kinds here
RESULTS = ROOT / "tuning" / "results"  # stage matrices are written here
CACHE = ROOT / "data" / ".cache"  # npy cache for mapped event rows


def grid6_points() -> list[dict]:
    return [dict(zip(PARAMS, combo))  # name each parameter combination
            for combo in product(*(GRID6[p] for p in PARAMS))]  # full G6 cross-product


def _event_anchors_and_epochs(xml_path: Path, tags: tuple[str, ...]
                              ) -> tuple[list[tuple[float, int]], list[float]]:
    """(anchors: epoch_s from any StartFrame event, epochs: parents carrying `tags`)."""
    tree = ET.parse(xml_path)  # parse the events XML once
    anchors: list[tuple[float, int]] = []  # (epoch_s, provider frame) pairs
    epochs: list[float] = []  # event times carrying one of our tags
    for ev in tree.getroot().iter("Event"):
        attrs = ev.attrib  # this event's attributes
        if any(c.tag in tags for c in ev):  # child tag matches what we want
            epochs.append(pd.Timestamp(attrs["EventTime"]).timestamp())  # epoch seconds
        if "StartFrame" in attrs:  # event pins a provider frame
            anchors.append((pd.Timestamp(attrs["EventTime"]).timestamp(),
                            int(attrs["StartFrame"])))  # alignment anchor pair
    return anchors, epochs  # both lists for mapping


def load_event_rows(match_id: str, tags: tuple[str, ...], cache_name: str) -> np.ndarray:
    """Rows for one IDSSE event type; kloppy+XML once, then a local npy cache."""
    CACHE.mkdir(parents=True, exist_ok=True)  # make sure the cache dir exists
    cache = CACHE / f"{cache_name}_{match_id}.npy"  # cache file for this match
    if cache.exists():
        return np.load(cache)  # reuse a previous mapping
    xmls = sorted(IDSSE_DIR.glob(f"*events_raw*{match_id}.xml"))  # find this match's events XML
    if not xmls:
        raise FileNotFoundError(f"events XML for {match_id} under {IDSSE_DIR}")  # fail loudly
    anchors, epochs = _event_anchors_and_epochs(xmls[0], tags)  # anchors + event epochs

    from kloppy import sportec  # deferred: heavy import

    from tuning.idsse import _files
    meta_path, raw_path = _files(match_id)  # tracking file paths
    ds = sportec.load_tracking(meta_data=meta_path, raw_data=raw_path, only_alive=False)  # full tracking
    df = ds.to_df()  # flat frame table
    rows = map_event_rows(  # event epochs -> tracking rows
        df["timestamp"].to_numpy(dtype=np.float64),
        df["frame_id"].to_numpy(),
        df["period_id"].to_numpy(),
        anchors, epochs)
    kept = rows[rows >= 0]  # drop unmapped events
    print(f"{match_id}: {cache_name} {len(kept)}/{len(epochs)} mapped, "  # mapping quality
          f"{len(anchors)} anchors", flush=True)
    np.save(cache, kept)  # cache for the worker processes
    return kept  # mapped rows only


def load_tackle_rows(match_id: str) -> np.ndarray:
    """TacklingGame rows for one IDSSE match (E2 cue 3)."""
    return load_event_rows(match_id, ("TacklingGame",), "tackles")  # cue-3 rows, cached


def cues_for(case, tackle_rows: np.ndarray) -> list[np.ndarray]:
    return [
        possession_change_cue(case.letters),  # cue 1: possession changes
        crowding_cue(case.players, case.ball,  # cue 2: crowded ball
                     length_m=CFG["pitch_length_m"], width_m=CFG["pitch_width_m"]),
        event_cue(tackle_rows, len(case.letters)),  # cue 3: TacklingGame events
    ]


def run_task4(args) -> tuple[list[dict], dict]:
    """All grid cells for one (match, mask kind). Runs in a worker process."""
    match_id, kind, grid, run_lengths = args  # unpack the worker task
    t0 = time.time()  # start the runtime stamp
    case = load_match(match_id)  # cached match arrays
    tackle_rows = load_tackle_rows(match_id)          # pre-warmed by main()
    cues = cues_for(case, tackle_rows)  # the three E2 phase cues

    rng = np.random.default_rng(MASK_SEED + sum(ord(c) for c in match_id))  # seeded per match
    if kind == "uniform":
        masked, mask_stats = mask_ball(case.ball, run_lengths, rng)  # uniform placement
    else:
        alpha = 3.0 if kind == "phase_a3" else 2.0  # α=3 sensitivity, else α=2
        masked, mask_stats = mask_ball(case.ball, run_lengths, rng,
                                       weights=phase_weights(cues, alpha))  # phase-weighted

    base = dict(CFG)  # copy of the config
    if kind != "phase_a2_fill":                       # fill on only for the E11 twin
        base["fill_long_gap_s"] = 0.0  # keep fill off for the other kinds

    from src.ball_fill import build_ball_track  # gap-fill step on the masked ball
    from src.decoder import decode  # the possession decoder
    from src.emission import build_emission  # emission model builder

    track = build_ball_track(masked, case.half, base)  # ball track from masked input
    gt_orig = pm.label_codes(case.letters)  # orig label set (per-touch)
    gt_aligned = pm.label_codes(derive_letters_controlled(case.letters, hold_s=3.0))  # aligned (E1, T=3 s)
    runs_orig, runs_aligned = pm.valid_runs(gt_orig), pm.valid_runs(gt_aligned)  # valid runs per set
    n_gt_orig = len(pm.possession_spells(gt_orig, runs_orig))  # GT spell counts for the row
    n_gt_aligned = len(pm.possession_spells(gt_aligned, runs_aligned))  # same, aligned set

    # pre-registered mask statistics (§7): NaN share and newly-removed share, in/out cues
    was_finite = ~np.isnan(case.ball).any(axis=1)  # ball present before masking
    now_nan = np.isnan(masked).any(axis=1)  # ball missing after masking
    removed = was_finite & now_nan  # frames this mask newly removed
    any_cue = cues[0] | cues[1] | cues[2]  # union of the three cues
    mask_row = {"match_id": match_id, "kind": kind, **mask_stats,  # mask-stats row for this task
                "nan_share": float(now_nan.mean()),
                "removed_share": float(removed.mean())}
    # per-cue NaN / removed shares inside and outside the cue, plus the cue's frame share
    for i, cue in enumerate(cues, start=1):
        mask_row[f"cue{i}_nan_in"] = float(now_nan[cue].mean()) if cue.any() else np.nan  # NaN share inside cue
        mask_row[f"cue{i}_nan_out"] = float(now_nan[~cue].mean()) if (~cue).any() else np.nan  # NaN share outside
        mask_row[f"cue{i}_removed_in"] = float(removed[cue].mean()) if cue.any() else np.nan  # removed share inside
        mask_row[f"cue{i}_removed_out"] = float(removed[~cue].mean()) if (~cue).any() else np.nan  # outside share
        mask_row[f"cue{i}_frame_share"] = float(cue.mean())  # how often the cue fires
    mask_row["anycue_removed_in"] = float(removed[any_cue].mean()) if any_cue.any() else np.nan  # inside any cue
    mask_row["anycue_removed_out"] = float(removed[~any_cue].mean()) if (~any_cue).any() else np.nan  # outside all cues

    rows: list[dict] = []  # cell rows accumulate here
    sigmas = sorted({float(c["sigma_m"]) for c in grid})  # unique sigma values to build for
    for sigma in sigmas:
        em = build_emission(track, case.players, case.players_s, case.slot_team,
                            {**base, "sigma_m": sigma})  # emission reused across cells
        for cell in grid:
            if float(cell["sigma_m"]) != sigma:
                continue  # this cell belongs to another sigma group
            cfg = {**base, **cell}  # base config plus this grid cell
            decoded = decode(em, track.source, case.half, cfg)  # decode with this cell
            pred = _LETTERS[np.asarray(decoded.labels, dtype=np.int8) + 1]  # codes -> letters
            pred_codes = pm.label_codes(pred)  # letters -> label codes
            s_orig = _score(gt_orig, pred_codes)  # score against orig labels
            s_aligned = _score(gt_aligned, pred_codes)  # score against aligned labels
            rows.append({  # one matrix row per cell
                "match_id": match_id, "kind": kind, **cell,
                "spell_f1_orig": s_orig["spell_f1"], "macro_f1_orig": s_orig["macro_f1"],
                "spell_f1_aligned": s_aligned["spell_f1"],
                "macro_f1_aligned": s_aligned["macro_f1"],
                "n_gt_orig": n_gt_orig, "n_gt_aligned": n_gt_aligned,
                "masked_fraction": round(mask_stats["masked_fraction"], 4),
                "runtime_s": round(time.time() - t0, 1),
            })
    return rows, mask_row  # cell rows plus mask stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # help = docstring line 1
    parser.add_argument("--smoke", action="store_true",  # small plumbing run
                        help="1 match, 6 strided cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=7)  # pool size
    args = parser.parse_args()  # parsed command-line flags

    from src import data as D  # deferred: heavy import

    arrays = D.load_arrays()  # provided arrays, only for run lengths here
    missing_graded = np.isnan(arrays.ball).any(axis=1) & arrays.graded  # graded missing frames
    run_lengths = empirical_run_lengths(missing_graded)  # empirical missing-run lengths
    print(f"empirical missing runs: n={len(run_lengths)} "  # show run-length summary
          f"median={int(np.median(run_lengths))} max={int(run_lengths.max())}", flush=True)

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS  # 1 match in smoke, all 7 otherwise
    grid = grid6_points()[::24][:6] if args.smoke else grid6_points()  # strided smoke cells
    for mid in matches:                               # warm caches BEFORE the pool
        load_match(mid)  # preload match arrays
        load_tackle_rows(mid)  # preload tackle rows
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(grid)} cells each",  # show workload
          flush=True)

    tasks = [(m, k, grid, run_lengths) for m in matches for k in KINDS]  # one task per pair
    from multiprocessing import Pool  # deferred: process pool

    t0 = time.time()  # total runtime starts here
    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}  # cell rows grouped by kind
    mask_rows: list[dict] = []  # one mask-stats row per task
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        for i, (rows, mask_row) in enumerate(pool.imap_unordered(run_task4, tasks), 1):
            by_kind[rows[0]["kind"]].extend(rows)  # collect this task's cell rows
            mask_rows.append(mask_row)  # collect this task's mask stats
            print(f"[{i}/{len(tasks)}] {mask_row['match_id']} {mask_row['kind']} "  # progress counter
                  f"done at {time.time() - t0:.0f}s", flush=True)

    out_dir = Path("/tmp/stage4_smoke") if args.smoke else RESULTS  # smoke stays out of results/
    out_dir.mkdir(parents=True, exist_ok=True)  # create the output dir
    for kind, rows in by_kind.items():
        path = out_dir / f"stage4_{kind}.csv"  # this kind's output file
        pd.DataFrame(rows).to_csv(path, index=False)  # write this kind's matrix
        print(f"wrote {path} ({len(rows)} rows)", flush=True)  # confirm file written
    masks_path = out_dir / "stage4_masks.csv"  # mask-stats output file
    pd.DataFrame(mask_rows).to_csv(masks_path, index=False)  # write mask statistics
    print(f"wrote {masks_path} ({len(mask_rows)} rows)", flush=True)  # confirm file written
    print(f"total {time.time() - t0:.0f}s", flush=True)  # total runtime
    return 0  # zero means success


if __name__ == "__main__":  # module entry point
    sys.exit(main())  # run as a script
