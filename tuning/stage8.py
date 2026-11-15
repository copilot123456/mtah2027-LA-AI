"""Stage 8 — the pre-declared P8 pass (dossier §7, 2026-11-14): switch gate move mode (G),
team-switch delay (S), dead-ball minimum duration (M), plus two record-only cells.

    python -m tuning.stage8 --smoke     # plumbing: 1 match, 6 cells, all four mask kinds
    python -m tuning.stage8             # full pass: 72 cells x 7 matches x 4 mask kinds

Everything is fixed at the stage-5 reference (frozen decoder cell, shipped dead-ball term,
fill off, the stage-5 masks). The harness runs the pipeline's own decoder: one Viterbi per
task (`viterbi_paths`), then `finish_paths` per cell. Reading the matrices and applying the
pinned rule is `tuning.gate8` — never here.
Full runs write `stage8_<kind>.csv`, `stage8_record.csv` and `stage8_diag.csv` into
`tuning/results/`.
"""

from __future__ import annotations  # postponed type-hint evaluation

import argparse  # CLI flags
import sys  # sys.path setup below
import time  # wall-clock timings
from itertools import product  # factorial cell grid
from pathlib import Path  # path objects

import numpy as np  # array maths
import pandas as pd  # DataFrame + CSV writing

ROOT = Path(__file__).resolve().parents[1]  # repo root
for p in (str(ROOT), str(ROOT / "evaluation")):  # import roots for the modules below
    if p not in sys.path:
        sys.path.insert(0, p)  # prepend only if missing

# project modules: scoring metrics, config, match/mask loaders, cell scoring
import possession_metrics as pm  # noqa: E402
from config import CFG  # noqa: E402
from tuning.idsse import MATCH_IDS, derive_letters_controlled, load_match  # noqa: E402
from tuning.simulate import empirical_run_lengths  # noqa: E402
from tuning.stage4 import load_tackle_rows  # noqa: E402
from tuning.stage5 import FROZEN_DECODER, KINDS, build_mask  # noqa: E402
from tuning.sweep import _LETTERS, _score  # noqa: E402

P_PARAMS = ["gate_mode", "gap_switch_frac", "switch_delay_s", "d_min_dwell_s"]  # P8 knobs; CSV sort order
G_CELLS = [("revert", 0.0), ("move", 0.0), ("move", 0.5)]  # gate-mode × gap-fraction pairs (G)
DELAYS = [0.0, 0.16, 0.32, 0.48, 0.64, 0.80]  # switch-delay grid in seconds (S)
D_MINS = [0.0, 3.0, 4.0, 5.0]  # dead-ball min-dwell grid in seconds (M)
# Reference cell = stage-5 behaviour: every P8 knob at its off value.
REFERENCE = {"gate_mode": "revert", "gap_switch_frac": 0.0,
             "switch_delay_s": 0.0, "d_min_dwell_s": 0.0}
# Record-only cells (operator idea I2): never adopted in P8 — the bed's ball is frozen
# while dead, so the dead-ball features cannot be judged on it (dossier §7 P8).
RECORD = {"R_stat0": {"d_w_stationary": 0.0}, "R_noD": {"d_bias": -30.0}}
LATENCY_TOL = 50  # frames: max gap when pairing GT/pred switches
RESULTS = ROOT / "tuning" / "results"  # full-pass output dir (smoke uses /tmp)


def p8_cells() -> list[dict]:
    """The 72 factorial cells; the first one is the reference.
    Gate pairs vary slowest, dwell fastest."""
    return [{"gate_mode": g, "gap_switch_frac": f, "switch_delay_s": d, "d_min_dwell_s": m}
            for (g, f), d, m in product(G_CELLS, DELAYS, D_MINS)]


def team_switch_latency(gt_codes: np.ndarray, pred_codes: np.ndarray) -> tuple[int, float]:
    """(matched, mean predicted − true frame) over H<->A changes, same from/to pair."""
    runs = pm.valid_runs(gt_codes)  # GT segments without abstentions
    filled = pm.fill_abstentions(pred_codes, runs)  # forward-fill pred inside them
    team = (pm.H, pm.A)  # only home↔away switches count

    def switches(codes):  # frames where the team label changes
        return [t for t in pm.transitions(codes, runs) if t[1] in team and t[2] in team]

    pairs, _, _ = pm.match_transitions(switches(gt_codes), switches(filled),
                                       LATENCY_TOL, "pair")  # pair same from→to switches
    if not pairs:
        return 0, float("nan")  # nothing matched
    return len(pairs), float(np.mean([pred[0] - gt[0] for gt, pred in pairs]))  # count, mean lag


def run_task8(args) -> tuple[list[dict], list[dict], dict]:
    """All cells + record-only cells + diagnostics for one (match, mask kind)."""
    match_id, kind, cells = args[:3]  # unpack worker task
    run_lengths = args[3]  # missing-ball run lengths for the mask
    t0 = time.time()  # task clock, reported as runtime_s
    case = load_match(match_id)  # cached match data
    if case.info.get("speed_units") != "m/s":  # refuse a non-metric cache
        raise ValueError(f"{match_id}: cache is not in m/s — run the loader once first")
    masked, _ = build_mask(case, kind, run_lengths, load_tackle_rows(match_id))  # apply mask kind

    # ball track builder, decoder codes + Viterbi/finish, emission builder
    from src.ball_fill import build_ball_track
    from src.decoder import CODE_A, CODE_D, CODE_H, finish_paths, viterbi_paths
    from src.emission import build_emission

    base = {**CFG, **FROZEN_DECODER, "fill_long_gap_s": 0.0, **REFERENCE}  # frozen decoder config
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

    em = build_emission(track, case.players, case.players_s, case.slot_team, base)  # reference emission
    paths = viterbi_paths(em, case.half, base)  # one Viterbi shared by all cells
    ref_labels = finish_paths(paths, em, track.source, base).labels  # reference cell's labels

    rows = []
    for cell in cells:  # cells only change finish_paths; emission/Viterbi are shared
        labels = finish_paths(paths, em, track.source, {**base, **cell}).labels
        rows.append({"match_id": match_id, "kind": kind, **cell, **scored(labels),
                     "pred_d_share": float((labels[in_match] == CODE_D).mean()),  # predicted D share
                     "changed_share": float((labels != ref_labels)[in_match].mean()),  # vs reference cell
                     "runtime_s": round(time.time() - t0, 1)})  # elapsed so far

    record_rows = []
    for name, override in RECORD.items():  # record-only cells: own emission + own decode
        cfg = {**base, **override}
        em_r = build_emission(track, case.players, case.players_s, case.slot_team, cfg)  # override may touch emission
        labels = finish_paths(viterbi_paths(em_r, case.half, cfg), em_r, track.source, cfg).labels
        record_rows.append({"match_id": match_id, "kind": kind, "cell": name, **scored(labels),  # same metrics
                            "pred_d_share": float((labels[in_match] == CODE_D).mean()),
                            "changed_share": float((labels != ref_labels)[in_match].mean())})

    # Diagnostics: team-swap equivariance per G cell, and the I4 shares at the reference.
    em_s = build_emission(track, case.players, case.players_s, 1 - case.slot_team, base)  # sides swapped
    paths_s = viterbi_paths(em_s, case.half, base)  # decode under the swap
    diag = {"match_id": match_id, "kind": kind}
    for gate_mode, frac in G_CELLS:  # one equivariance check per gate variant
        cfg = {**base, "gate_mode": gate_mode, "gap_switch_frac": frac}
        a = finish_paths(paths, em, track.source, cfg).labels  # original-team labels
        b = finish_paths(paths_s, em_s, track.source, cfg).labels  # swapped-team labels
        mirrored = b.copy()
        mirrored[b == CODE_H], mirrored[b == CODE_A] = CODE_A, CODE_H  # undo the swap
        diag[f"equiv_mismatch_{gate_mode}_{frac:g}"] = float((mirrored != a)[in_match].mean())  # disagreeing share
    true_dead = (case.letters == "D") & in_match  # GT dead-ball frames
    missed = true_dead & np.isin(ref_labels, (CODE_H, CODE_A))  # dead frames decoded as live
    alive = np.isin(case.letters, ("H", "A"))  # GT live frames
    diag["missed_dead_to_home"] = float((ref_labels[missed] == CODE_H).mean()) if missed.any() \
        else float("nan")  # NaN when nothing was missed
    diag["home_share_of_live"] = float((case.letters[alive] == "H").mean())  # home share of live GT
    return rows, record_rows, diag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # first docstring line as help
    parser.add_argument("--smoke", action="store_true",  # tiny plumbing run
                        help="1 match, 6 cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=6)  # pool size
    args = parser.parse_args()

    from src import data as D  # raw-array loader

    arrays = D.load_arrays()  # ball + graded arrays
    run_lengths = empirical_run_lengths(np.isnan(arrays.ball).any(axis=1) & arrays.graded)  # missing-ball runs
    del arrays  # free the big arrays before forking

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS  # smoke runs one match only
    cells = p8_cells()
    if args.smoke:
        cells = [cells[0], cells[5], cells[24], cells[30], cells[48], cells[-1]]  # six cells across the grid
    for mid in matches:                               # warm caches BEFORE the pool
        load_match(mid)  # populate loaders once, in the parent
        load_tackle_rows(mid)
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(cells)} cells each", flush=True)  # run plan

    tasks = [(m, k, cells, run_lengths) for m in matches for k in KINDS]  # one task per match × kind
    from multiprocessing import Pool  # local import

    t0 = time.time()  # whole-pass timer
    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}  # rows bucketed by mask kind
    record_rows: list[dict] = []
    diag_rows: list[dict] = []
    with Pool(processes=min(args.workers, len(tasks))) as pool:  # never more workers than tasks
        for i, (rows, rrows, diag) in enumerate(pool.imap_unordered(run_task8, tasks), 1):  # any order
            by_kind[diag["kind"]].extend(rows)  # collect this kind's cells
            record_rows.extend(rrows)  # record-only rows
            diag_rows.append(diag)  # one diag per task
            print(f"[{i}/{len(tasks)}] {diag['match_id']} {diag['kind']} "
                  f"done at {time.time() - t0:.0f}s", flush=True)  # progress

    out_dir = Path("/tmp/stage8_smoke") if args.smoke else RESULTS  # smoke to /tmp, full to results/
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {m: i for i, m in enumerate(MATCH_IDS)}  # canonical match order
    for kind, rows in by_kind.items():  # one matrix CSV per mask kind
        df = pd.DataFrame(rows)
        df["_m"] = df["match_id"].map(order)  # temp sort key
        df = df.sort_values(P_PARAMS + ["_m"], kind="stable").drop(columns="_m")  # grid order, then match
        df.to_csv(out_dir / f"stage8_{kind}.csv", index=False)
        print(f"wrote stage8_{kind}.csv ({len(df)} rows)", flush=True)
    # record-only cells, one row each
    pd.DataFrame(record_rows).sort_values(["kind", "cell", "match_id"]).to_csv(
        out_dir / "stage8_record.csv", index=False)
    # diagnostics, one row per task
    pd.DataFrame(diag_rows).sort_values(["kind", "match_id"]).to_csv(
        out_dir / "stage8_diag.csv", index=False)
    print(f"total {time.time() - t0:.0f}s", flush=True)  # wall-clock total
    return 0


if __name__ == "__main__":
    sys.exit(main())  # exit status for the shell
