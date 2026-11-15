"""Stage 10 — the pre-declared P10 pass (dossier §7, 2026-11-15): `kin_w_dead` past the
grid edge, everything else at the shipped bundle cell.

    python -m tuning.stage10 --smoke     # plumbing: 1 match, a few cells, all four kinds
    python -m tuning.stage10             # full pass: 6 cells x 7 matches x 4 mask kinds

The harness is stage 9's, one emission + one Viterbi per K weight. `changed_share` is
measured against the 0.3 cell (the shipped bundle), not the pre-P9 reference. Reading the
matrices and applying the pinned rule is `tuning.gate10` — never here.
"""

from __future__ import annotations

import argparse
import sys
import time
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
from tuning.simulate import empirical_run_lengths  # noqa: E402
from tuning.stage4 import load_tackle_rows  # noqa: E402
from tuning.stage5 import FROZEN_DECODER, KINDS, build_mask  # noqa: E402
from tuning.stage8 import team_switch_latency  # noqa: E402
from tuning.stage9 import kin_for_matches  # noqa: E402
from tuning.sweep import MASK_SEED, _LETTERS, _score  # noqa: E402

P_PARAMS = ["kin_w_dead"]
K_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
SHIPPED = {"dwell_coalesce": True, "kin_w_dead": 0.3, "kin_w_team": 0.03,
           "gate_kin_support": True, "switch_delay_s": 0.64, "switch_delay_slope": 0.0}
RECORD = {  # record-only rows, never adopted (dossier §7 P10)
    "R_K05_all": ({"kin_w_dead": 0.5}, {"kin_dead_all_frames": True}),
    "R_K10_T0": ({"kin_w_dead": 1.0, "kin_w_team": 0.0}, {}),
}
RESULTS = ROOT / "tuning" / "results"


def run_task10(args) -> tuple[list[dict], list[dict]]:
    """All K weights + the record-only rows for one (match, mask kind)."""
    match_id, kind, kds, run_lengths, kin = args
    t0 = time.time()
    case = load_match(match_id)
    if case.info.get("speed_units") != "m/s":
        raise ValueError(f"{match_id}: cache is not in m/s — run the loader once first")
    masked, _ = build_mask(case, kind, run_lengths, load_tackle_rows(match_id))

    from src.ball_fill import build_ball_track
    from src.decoder import CODE_D, finish_paths, viterbi_paths
    from src.emission import build_emission

    base = {**CFG, **FROZEN_DECODER, "fill_long_gap_s": 0.0, **SHIPPED}
    track = build_ball_track(masked, case.half, base)
    gt = {"orig": pm.label_codes(case.letters),
          "aligned": pm.label_codes(derive_letters_controlled(case.letters, hold_s=3.0))}
    in_match = case.letters != "X"

    def scored(labels: np.ndarray) -> dict:
        codes = pm.label_codes(_LETTERS[np.asarray(labels, dtype=np.int8) + 1])
        row = {}
        for lab, gt_codes in gt.items():
            s = _score(gt_codes, codes)
            row[f"spell_f1_{lab}"] = s["spell_f1"]
            row[f"macro_f1_{lab}"] = s["macro_f1"]
        n_lat, lat = team_switch_latency(gt["aligned"], codes)
        row["lat_n"], row["lat_mean"] = n_lat, lat
        return row

    def emit(cfg: dict, **kw):
        return build_emission(track, case.players, case.players_s, case.slot_team, cfg,
                              kin=kin, **kw)

    rows = []
    labels_by_w = {}
    for wd in kds:
        cfg = {**base, "kin_w_dead": wd}
        em = emit(cfg)
        labels_by_w[wd] = finish_paths(viterbi_paths(em, case.half, cfg), em,
                                       track.source, cfg).labels
    ref_labels = labels_by_w[SHIPPED["kin_w_dead"]]
    for wd in kds:
        labels = labels_by_w[wd]
        rows.append({"match_id": match_id, "kind": kind, "kin_w_dead": wd, **scored(labels),
                     "pred_d_share": float((labels[in_match] == CODE_D).mean()),
                     "changed_share": float((labels != ref_labels)[in_match].mean()),
                     "runtime_s": round(time.time() - t0, 1)})

    record_rows = []
    for name, (override, kw) in RECORD.items():
        cfg = {**base, **override}
        em_r = emit(cfg, **kw)
        labels = finish_paths(viterbi_paths(em_r, case.half, cfg), em_r, track.source, cfg).labels
        record_rows.append({"match_id": match_id, "kind": kind, "cell": name, **scored(labels),
                            "pred_d_share": float((labels[in_match] == CODE_D).mean()),
                            "changed_share": float((labels != ref_labels)[in_match].mean())})
    return rows, record_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="1 match, a few cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    from src import data as D

    arrays = D.load_arrays()
    run_lengths = empirical_run_lengths(np.isnan(arrays.ball).any(axis=1) & arrays.graded)
    del arrays

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS
    kds = [0.3, 0.5, 1.0] if args.smoke else list(K_WEIGHTS)
    for mid in MATCH_IDS:                             # warm caches BEFORE the pool
        load_match(mid)
        load_tackle_rows(mid)
    t0 = time.time()
    kin, _ = kin_for_matches(matches)
    print(f"kinematic models (leave-one-out) ready at {time.time() - t0:.0f}s", flush=True)
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(kds)} cells each", flush=True)

    tasks = [(m, k, kds, run_lengths, kin[m][0]) for m in matches for k in KINDS]
    from multiprocessing import Pool

    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}
    record_rows: list[dict] = []
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        for i, (rows, rrows) in enumerate(pool.imap_unordered(run_task10, tasks), 1):
            by_kind[rows[0]["kind"]].extend(rows)
            record_rows.extend(rrows)
            print(f"[{i}/{len(tasks)}] {rows[0]['match_id']} {rows[0]['kind']} "
                  f"done at {time.time() - t0:.0f}s", flush=True)

    out_dir = Path("/tmp/stage10_smoke") if args.smoke else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {m: i for i, m in enumerate(MATCH_IDS)}
    for kind, rows in by_kind.items():
        df = pd.DataFrame(rows)
        df["_m"] = df["match_id"].map(order)
        df = df.sort_values(P_PARAMS + ["_m"], kind="stable").drop(columns="_m")
        df.to_csv(out_dir / f"stage10_{kind}.csv", index=False)
        print(f"wrote stage10_{kind}.csv ({len(df)} rows)", flush=True)
    pd.DataFrame(record_rows).sort_values(["kind", "cell", "match_id"]).to_csv(
        out_dir / "stage10_record.csv", index=False)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
