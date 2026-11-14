"""Stage 8 — the pre-declared P8 pass (dossier §7, 2026-11-14): switch gate move mode (G),
team-switch delay (S), dead-ball minimum duration (M), plus two record-only cells.

    python -m tuning.stage8 --smoke     # plumbing: 1 match, 6 cells, all four mask kinds
    python -m tuning.stage8             # full pass: 72 cells x 7 matches x 4 mask kinds

Everything is fixed at the stage-5 reference (frozen decoder cell, shipped dead-ball term,
fill off, the stage-5 masks). The harness runs the pipeline's own decoder: one Viterbi per
task (`viterbi_paths`), then `finish_paths` per cell. Reading the matrices and applying the
pinned rule is `tuning.gate8` — never here.
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
from tuning.simulate import empirical_run_lengths  # noqa: E402
from tuning.stage4 import load_tackle_rows  # noqa: E402
from tuning.stage5 import FROZEN_DECODER, KINDS, build_mask  # noqa: E402
from tuning.sweep import _LETTERS, _score  # noqa: E402

P_PARAMS = ["gate_mode", "gap_switch_frac", "switch_delay_s", "d_min_dwell_s"]
G_CELLS = [("revert", 0.0), ("move", 0.0), ("move", 0.5)]
DELAYS = [0.0, 0.16, 0.32, 0.48, 0.64, 0.80]
D_MINS = [0.0, 3.0, 4.0, 5.0]
REFERENCE = {"gate_mode": "revert", "gap_switch_frac": 0.0,
             "switch_delay_s": 0.0, "d_min_dwell_s": 0.0}
# Record-only cells (operator idea I2): never adopted in P8 — the bed's ball is frozen
# while dead, so the dead-ball features cannot be judged on it (dossier §7 P8).
RECORD = {"R_stat0": {"d_w_stationary": 0.0}, "R_noD": {"d_bias": -30.0}}
LATENCY_TOL = 50
RESULTS = ROOT / "tuning" / "results"


def p8_cells() -> list[dict]:
    """The 72 factorial cells; the first one is the reference."""
    return [{"gate_mode": g, "gap_switch_frac": f, "switch_delay_s": d, "d_min_dwell_s": m}
            for (g, f), d, m in product(G_CELLS, DELAYS, D_MINS)]


def team_switch_latency(gt_codes: np.ndarray, pred_codes: np.ndarray) -> tuple[int, float]:
    """(matched, mean predicted − true frame) over H<->A changes, same from/to pair."""
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    team = (pm.H, pm.A)

    def switches(codes):
        return [t for t in pm.transitions(codes, runs) if t[1] in team and t[2] in team]

    pairs, _, _ = pm.match_transitions(switches(gt_codes), switches(filled),
                                       LATENCY_TOL, "pair")
    if not pairs:
        return 0, float("nan")
    return len(pairs), float(np.mean([pred[0] - gt[0] for gt, pred in pairs]))


def run_task8(args) -> tuple[list[dict], list[dict], dict]:
    """All cells + record-only cells + diagnostics for one (match, mask kind)."""
    match_id, kind, cells = args[:3]
    run_lengths = args[3]
    t0 = time.time()
    case = load_match(match_id)
    if case.info.get("speed_units") != "m/s":
        raise ValueError(f"{match_id}: cache is not in m/s — run the loader once first")
    masked, _ = build_mask(case, kind, run_lengths, load_tackle_rows(match_id))

    from src.ball_fill import build_ball_track
    from src.decoder import CODE_A, CODE_D, CODE_H, finish_paths, viterbi_paths
    from src.emission import build_emission

    base = {**CFG, **FROZEN_DECODER, "fill_long_gap_s": 0.0, **REFERENCE}
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

    em = build_emission(track, case.players, case.players_s, case.slot_team, base)
    paths = viterbi_paths(em, case.half, base)
    ref_labels = finish_paths(paths, em, track.source, base).labels

    rows = []
    for cell in cells:
        labels = finish_paths(paths, em, track.source, {**base, **cell}).labels
        rows.append({"match_id": match_id, "kind": kind, **cell, **scored(labels),
                     "pred_d_share": float((labels[in_match] == CODE_D).mean()),
                     "changed_share": float((labels != ref_labels)[in_match].mean()),
                     "runtime_s": round(time.time() - t0, 1)})

    record_rows = []
    for name, override in RECORD.items():
        cfg = {**base, **override}
        em_r = build_emission(track, case.players, case.players_s, case.slot_team, cfg)
        labels = finish_paths(viterbi_paths(em_r, case.half, cfg), em_r, track.source, cfg).labels
        record_rows.append({"match_id": match_id, "kind": kind, "cell": name, **scored(labels),
                            "pred_d_share": float((labels[in_match] == CODE_D).mean()),
                            "changed_share": float((labels != ref_labels)[in_match].mean())})

    # Diagnostics: team-swap equivariance per G cell, and the I4 shares at the reference.
    em_s = build_emission(track, case.players, case.players_s, 1 - case.slot_team, base)
    paths_s = viterbi_paths(em_s, case.half, base)
    diag = {"match_id": match_id, "kind": kind}
    for gate_mode, frac in G_CELLS:
        cfg = {**base, "gate_mode": gate_mode, "gap_switch_frac": frac}
        a = finish_paths(paths, em, track.source, cfg).labels
        b = finish_paths(paths_s, em_s, track.source, cfg).labels
        mirrored = b.copy()
        mirrored[b == CODE_H], mirrored[b == CODE_A] = CODE_A, CODE_H
        diag[f"equiv_mismatch_{gate_mode}_{frac:g}"] = float((mirrored != a)[in_match].mean())
    true_dead = (case.letters == "D") & in_match
    missed = true_dead & np.isin(ref_labels, (CODE_H, CODE_A))
    alive = np.isin(case.letters, ("H", "A"))
    diag["missed_dead_to_home"] = float((ref_labels[missed] == CODE_H).mean()) if missed.any() \
        else float("nan")
    diag["home_share_of_live"] = float((case.letters[alive] == "H").mean())
    return rows, record_rows, diag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="1 match, 6 cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from src import data as D

    arrays = D.load_arrays()
    run_lengths = empirical_run_lengths(np.isnan(arrays.ball).any(axis=1) & arrays.graded)
    del arrays

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS
    cells = p8_cells()
    if args.smoke:
        cells = [cells[0], cells[5], cells[24], cells[30], cells[48], cells[-1]]
    for mid in matches:                               # warm caches BEFORE the pool
        load_match(mid)
        load_tackle_rows(mid)
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(cells)} cells each", flush=True)

    tasks = [(m, k, cells, run_lengths) for m in matches for k in KINDS]
    from multiprocessing import Pool

    t0 = time.time()
    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}
    record_rows: list[dict] = []
    diag_rows: list[dict] = []
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        for i, (rows, rrows, diag) in enumerate(pool.imap_unordered(run_task8, tasks), 1):
            by_kind[diag["kind"]].extend(rows)
            record_rows.extend(rrows)
            diag_rows.append(diag)
            print(f"[{i}/{len(tasks)}] {diag['match_id']} {diag['kind']} "
                  f"done at {time.time() - t0:.0f}s", flush=True)

    out_dir = Path("/tmp/stage8_smoke") if args.smoke else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {m: i for i, m in enumerate(MATCH_IDS)}
    for kind, rows in by_kind.items():
        df = pd.DataFrame(rows)
        df["_m"] = df["match_id"].map(order)
        df = df.sort_values(P_PARAMS + ["_m"], kind="stable").drop(columns="_m")
        df.to_csv(out_dir / f"stage8_{kind}.csv", index=False)
        print(f"wrote stage8_{kind}.csv ({len(df)} rows)", flush=True)
    pd.DataFrame(record_rows).sort_values(["kind", "cell", "match_id"]).to_csv(
        out_dir / "stage8_record.csv", index=False)
    pd.DataFrame(diag_rows).sort_values(["kind", "match_id"]).to_csv(
        out_dir / "stage8_diag.csv", index=False)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
