"""Stage 10 — the pre-declared P10 pass (dossier §7, 2026-11-15): `kin_w_dead` past the
grid edge, everything else at the shipped bundle cell.

    python -m tuning.stage10 --smoke     # plumbing: 1 match, a few cells, all four kinds
    python -m tuning.stage10             # full pass: 6 cells x 7 matches x 4 mask kinds

The harness is stage 9's, one emission + one Viterbi per K weight. `changed_share` is
measured against the 0.3 cell (the shipped bundle), not the pre-P9 reference. Reading the
matrices and applying the pinned rule is `tuning.gate10` — never here.
Full runs write `stage10_<kind>.csv` and `stage10_record.csv` into `tuning/results/`.
"""

from __future__ import annotations  # postponed type-hint evaluation

import argparse  # CLI flags
import sys  # sys.path setup below
import time  # wall-clock timings
from pathlib import Path  # path objects

import numpy as np  # array maths
import pandas as pd  # DataFrame + CSV writing

ROOT = Path(__file__).resolve().parents[1]  # repo root
for p in (str(ROOT), str(ROOT / "evaluation")):  # import roots for the modules below
    if p not in sys.path:
        sys.path.insert(0, p)  # prepend only if missing

# project modules: metrics, config, match/mask loaders, latency, kinematics, cell scoring
import possession_metrics as pm  # noqa: E402
from config import CFG  # noqa: E402
from tuning.idsse import MATCH_IDS, derive_letters_controlled, load_match  # noqa: E402
from tuning.simulate import empirical_run_lengths  # noqa: E402
from tuning.stage4 import load_tackle_rows  # noqa: E402
from tuning.stage5 import FROZEN_DECODER, KINDS, build_mask  # noqa: E402
from tuning.stage8 import team_switch_latency  # noqa: E402
from tuning.stage9 import kin_for_matches  # noqa: E402
from tuning.sweep import MASK_SEED, _LETTERS, _score  # noqa: E402

P_PARAMS = ["kin_w_dead"]  # the only knob varied; CSV sort order
K_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]  # K weight grid past the stage-9 edge (P10)
SHIPPED = {"dwell_coalesce": True, "kin_w_dead": 0.3, "kin_w_team": 0.03,  # shipped bundle cell;
           "gate_kin_support": True, "switch_delay_s": 0.64, "switch_delay_slope": 0.0}  # P9 verdict values
RECORD = {  # record-only rows, never adopted (dossier §7 P10)
    "R_K05_all": ({"kin_w_dead": 0.5}, {"kin_dead_all_frames": True}),  # (config override, emit kwargs)
    "R_K10_T0": ({"kin_w_dead": 1.0, "kin_w_team": 0.0}, {}),
}
RESULTS = ROOT / "tuning" / "results"  # full-pass output dir (smoke uses /tmp)


def run_task10(args) -> tuple[list[dict], list[dict]]:
    """All K weights + the record-only rows for one (match, mask kind)."""
    match_id, kind, kds, run_lengths, kin = args  # unpack worker task
    t0 = time.time()  # task clock, reported as runtime_s
    case = load_match(match_id)  # cached match data
    if case.info.get("speed_units") != "m/s":  # refuse a non-metric cache
        raise ValueError(f"{match_id}: cache is not in m/s — run the loader once first")
    masked, _ = build_mask(case, kind, run_lengths, load_tackle_rows(match_id))  # apply mask kind

    # ball track, decoder (needs CODE_D) + Viterbi/finish, emission builder
    from src.ball_fill import build_ball_track
    from src.decoder import CODE_D, finish_paths, viterbi_paths
    from src.emission import build_emission

    base = {**CFG, **FROZEN_DECODER, "fill_long_gap_s": 0.0, **SHIPPED}  # frozen config at shipped cell
    track = build_ball_track(masked, case.half, base)  # ball track from the masked ball
    # two ground truths: raw letters, and challenge-definition labels (3s hold)
    gt = {"orig": pm.label_codes(case.letters),
          "aligned": pm.label_codes(derive_letters_controlled(case.letters, hold_s=3.0))}
    in_match = case.letters != "X"  # only frames with a real GT label are scored

    def scored(labels: np.ndarray) -> dict:  # score one cell's labels
        codes = pm.label_codes(_LETTERS[np.asarray(labels, dtype=np.int8) + 1])  # ints → letters → codes
        row = {}
        for lab, gt_codes in gt.items():  # score against both ground truths
            s = _score(gt_codes, codes)
            row[f"spell_f1_{lab}"] = s["spell_f1"]  # official spell-level F1
            row[f"macro_f1_{lab}"] = s["macro_f1"]  # official macro F1
        n_lat, lat = team_switch_latency(gt["aligned"], codes)  # switch latency, aligned GT
        row["lat_n"], row["lat_mean"] = n_lat, lat  # matched count, mean frame lag
        return row

    def emit(cfg: dict, **kw):  # emission with this match's kinematic signals
        return build_emission(track, case.players, case.players_s, case.slot_team, cfg,
                              kin=kin, **kw)

    rows = []
    labels_by_w = {}  # one decode per K weight
    for wd in kds:
        cfg = {**base, "kin_w_dead": wd}  # only the K weight changes
        em = emit(cfg)  # one emission per weight
        labels_by_w[wd] = finish_paths(viterbi_paths(em, case.half, cfg), em,
                                       track.source, cfg).labels  # one Viterbi + finish
    ref_labels = labels_by_w[SHIPPED["kin_w_dead"]]  # reference = shipped 0.3 cell
    for wd in kds:
        labels = labels_by_w[wd]
        rows.append({"match_id": match_id, "kind": kind, "kin_w_dead": wd, **scored(labels),
                     "pred_d_share": float((labels[in_match] == CODE_D).mean()),  # predicted D share
                     "changed_share": float((labels != ref_labels)[in_match].mean()),  # vs the 0.3 cell
                     "runtime_s": round(time.time() - t0, 1)})  # elapsed so far

    record_rows = []
    for name, (override, kw) in RECORD.items():  # (config override, emit kwargs)
        cfg = {**base, **override}
        em_r = emit(cfg, **kw)  # extra emission flags
        labels = finish_paths(viterbi_paths(em_r, case.half, cfg), em_r, track.source, cfg).labels
        record_rows.append({"match_id": match_id, "kind": kind, "cell": name, **scored(labels),  # same metrics
                            "pred_d_share": float((labels[in_match] == CODE_D).mean()),
                            "changed_share": float((labels != ref_labels)[in_match].mean())})  # vs 0.3 cell
    return rows, record_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # first docstring line as help
    parser.add_argument("--smoke", action="store_true",  # tiny plumbing run
                        help="1 match, a few cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=7)  # pool size
    args = parser.parse_args()

    from src import data as D  # raw-array loader

    arrays = D.load_arrays()  # ball + graded arrays
    run_lengths = empirical_run_lengths(np.isnan(arrays.ball).any(axis=1) & arrays.graded)  # missing-ball runs
    del arrays  # free the big arrays before forking

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS  # smoke runs one match only
    kds = [0.3, 0.5, 1.0] if args.smoke else list(K_WEIGHTS)  # smoke: 3 of the 6 weights
    for mid in MATCH_IDS:                             # warm caches BEFORE the pool
        load_match(mid)  # all seven, even in smoke (kin fit needs them)
        load_tackle_rows(mid)
    t0 = time.time()  # kin-fit timer
    kin, _ = kin_for_matches(matches)  # leave-one-out kinematic models
    print(f"kinematic models (leave-one-out) ready at {time.time() - t0:.0f}s", flush=True)
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(kds)} cells each", flush=True)  # run plan

    # one task per (match, mask kind), carrying only its match's degraded-track signals
    tasks = [(m, k, kds, run_lengths, kin[m][0]) for m in matches for k in KINDS]
    from multiprocessing import Pool  # local import

    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}  # rows bucketed by mask kind
    record_rows: list[dict] = []
    with Pool(processes=min(args.workers, len(tasks))) as pool:  # never more workers than tasks
        for i, (rows, rrows) in enumerate(pool.imap_unordered(run_task10, tasks), 1):  # any order
            by_kind[rows[0]["kind"]].extend(rows)  # all rows share one kind
            record_rows.extend(rrows)  # record-only rows
            print(f"[{i}/{len(tasks)}] {rows[0]['match_id']} {rows[0]['kind']} "
                  f"done at {time.time() - t0:.0f}s", flush=True)  # progress

    out_dir = Path("/tmp/stage10_smoke") if args.smoke else RESULTS  # smoke to /tmp, full to results/
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {m: i for i, m in enumerate(MATCH_IDS)}  # canonical match order
    for kind, rows in by_kind.items():  # one matrix CSV per mask kind
        df = pd.DataFrame(rows)
        df["_m"] = df["match_id"].map(order)  # temp sort key
        df = df.sort_values(P_PARAMS + ["_m"], kind="stable").drop(columns="_m")  # grid order, then match
        df.to_csv(out_dir / f"stage10_{kind}.csv", index=False)
        print(f"wrote stage10_{kind}.csv ({len(df)} rows)", flush=True)
    # record-only rows, one per cell
    pd.DataFrame(record_rows).sort_values(["kind", "cell", "match_id"]).to_csv(
        out_dir / "stage10_record.csv", index=False)
    print(f"total {time.time() - t0:.0f}s", flush=True)  # wall-clock total
    return 0


if __name__ == "__main__":
    sys.exit(main())  # exit status for the shell
