"""Stage 9 — the pre-declared P9 pass (dossier §7, 2026-11-14): coalescing min-dwell (C),
kinematic dead-ball and team terms on ball-less frames with the gate tier (K, T),
speed-scaled team-switch delay (S), plus one record-only cell.

    python -m tuning.stage9 --smoke     # plumbing: 1 match, a handful of cells, all four kinds
    python -m tuning.stage9             # full pass: 744 cells x 7 matches x 4 mask kinds

Everything else is the shipped configuration and the stage-5 masks. The kinematic models are
fitted leave-one-match-out (`tuning.kin_fit`) and the test match's kinematic features come
from dropout-degraded player tracks (the pinned bed handicap). The harness runs the
pipeline's own functions: one emission + one Viterbi per (K weight, T weight), then
`finish_paths` per cell. Reading the matrices and applying the pinned rule is
`tuning.gate9` — never here.
Full runs write `stage9_<kind>.csv`, `stage9_record.csv` and `stage9_diag.csv` into
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

# project modules: scoring metrics, config, kinematics, match/kin/mask loaders, cell scoring
import possession_metrics as pm  # noqa: E402
from config import CFG  # noqa: E402
from src.kinematics import Kinematics, apply_model  # noqa: E402
from tuning.idsse import MATCH_IDS, derive_letters_controlled, load_match  # noqa: E402
from tuning.kin_fit import degrade_players, fit_models, match_features  # noqa: E402
from tuning.simulate import empirical_run_lengths  # noqa: E402
from tuning.stage4 import load_tackle_rows  # noqa: E402
from tuning.stage5 import FROZEN_DECODER, KINDS, build_mask  # noqa: E402
from tuning.stage8 import team_switch_latency  # noqa: E402
from tuning.sweep import MASK_SEED, _LETTERS, _score  # noqa: E402

P_PARAMS = ["dwell_coalesce", "kin_w_dead", "kin_w_team", "gate_kin_support",
            "switch_delay_s", "switch_delay_slope"]  # the six P9 knobs; CSV sort order
WEIGHTS = [0.0, 0.03, 0.1, 0.3]  # K and T weight grid
DELAYS = [0.64, 0.80, 0.96, 1.12]  # switch-delay grid in seconds (S)
SLOPES = [0.0, 0.03, 0.06]  # speed-scaling slope grid for the delay
# Reference cell = pre-P9 behaviour: every P9 knob off, delay at the stage-8 shipped value.
REFERENCE = {"dwell_coalesce": False, "kin_w_dead": 0.0, "kin_w_team": 0.0,
             "gate_kin_support": False, "switch_delay_s": 0.64, "switch_delay_slope": 0.0}
RECORD = {"R_kin_all": {"kin_w_dead": 0.1}}        # the dead term on every frame — never adopted
KIN_SEED_BASE = 900                                # rng = default_rng(MASK_SEED + 900 + Σ ord)
RESULTS = ROOT / "tuning" / "results"  # full-pass output dir (smoke uses /tmp)


def p9_cells() -> list[dict]:
    """The factorial; the gate tier is immaterial at zero weights and is counted once."""
    cells = []  # accumulator for cells
    # full factorial over the six knobs
    for c, wd, wt, tier, delay, slope in product((False, True), WEIGHTS, WEIGHTS,
                                                 (False, True), DELAYS, SLOPES):
        if tier and wd == 0.0 and wt == 0.0:  # at zero weights the tier changes nothing
            continue  # skip the duplicate rows
        cells.append({"dwell_coalesce": c, "kin_w_dead": wd, "kin_w_team": wt,  # valid cell
                      "gate_kin_support": tier, "switch_delay_s": delay,
                      "switch_delay_slope": slope})
    return cells  # 744 cells


def kin_for_matches(matches: list[str]) -> tuple[dict, list[dict]]:
    """{match: (degraded-track signals, clean-track signals)} with leave-one-out models, and
    the IDSSE side of the parity checks Q1 / Q5."""
    feats = {}  # per-match feature cache
    for mid in MATCH_IDS:                        # the fit always uses the other six
        feats[mid] = match_features(load_match(mid))  # feature matrices for every match
    out, ref_rows = {}, []  # signals out, parity rows out
    for mid in matches:
        case = load_match(mid)  # cached match data
        model = fit_models(feats, exclude=mid)  # leave-one-out: never sees this match
        rng = np.random.default_rng(MASK_SEED + KIN_SEED_BASE + sum(ord(c) for c in mid))  # per-match seed
        deg = match_features(case, *degrade_players(case.players, case.players_s, rng))  # dropout-degraded tracks
        f = feats[mid]  # this match's clean features
        out[mid] = (apply_model(deg["x_dead"], deg["x_team"], model),  # degraded-track signals
                    apply_model(f["x_dead"], f["x_team"], model))  # clean-track signals
        m = f["in_match"]  # in-match frame mask
        ref_rows.append({"match_id": mid,  # IDSSE side of parity checks Q1 / Q5
                         "speed_med_p50": float(np.nanmedian(f["speed_med"][m])),
                         "n_vis_mean": float(np.nanmean(f["n_vis"][m])),
                         "n_vis_mean_degraded": float(np.nanmean(deg["n_vis"][m]))})
    return out, ref_rows  # signals plus parity rows


def run_task9(args) -> tuple[list[dict], list[dict], dict]:
    """All cells + the record-only cell + diagnostics for one (match, mask kind)."""
    match_id, kind, cells, run_lengths, kin, kin_clean = args  # unpack worker task
    t0 = time.time()  # task clock, reported as runtime_s
    case = load_match(match_id)  # cached match data
    if case.info.get("speed_units") != "m/s":  # refuse a non-metric cache
        raise ValueError(f"{match_id}: cache is not in m/s — run the loader once first")  # demand a metric cache
    masked, _ = build_mask(case, kind, run_lengths, load_tackle_rows(match_id))  # apply mask kind

    # ball track + ATD source flag, decoder codes + Viterbi/finish, emission builder
    from src.ball_fill import SOURCE_ATD, build_ball_track
    from src.decoder import CODE_A, CODE_D, CODE_H, finish_paths, viterbi_paths
    from src.emission import _ball_speed_ms, build_emission

    base = {**CFG, **FROZEN_DECODER, "fill_long_gap_s": 0.0, **REFERENCE}  # frozen decoder config
    track = build_ball_track(masked, case.half, base)  # ball track from the masked ball
    # two ground truths: raw letters, and challenge-definition labels (3s hold)
    gt = {"orig": pm.label_codes(case.letters),
          "aligned": pm.label_codes(derive_letters_controlled(case.letters, hold_s=3.0))}
    in_match = case.letters != "X"  # only frames with a real GT label are scored

    def scored(labels: np.ndarray) -> dict:  # score one cell's labels
        codes = pm.label_codes(_LETTERS[np.asarray(labels, dtype=np.int8) + 1])  # ints → letters → codes
        row = {}  # metrics accumulator
        for lab, gt_codes in gt.items():  # score against both ground truths
            s = _score(gt_codes, codes)  # official metric pair
            row[f"spell_f1_{lab}"] = s["spell_f1"]  # official spell-level F1
            row[f"macro_f1_{lab}"] = s["macro_f1"]  # official macro F1
        n_lat, lat = team_switch_latency(gt["aligned"], codes)  # switch latency, aligned GT
        row["lat_n"], row["lat_mean"] = n_lat, lat  # matched count, mean frame lag
        return row  # metrics for this cell

    def emit(cfg: dict, slot_team=None, k=kin, **kw):  # emission with optional overrides
        return build_emission(track, case.players, case.players_s,
                              case.slot_team if slot_team is None else slot_team, cfg,
                              kin=k, **kw)  # k defaults to this match's signals

    em_ref = emit(base)  # reference emission
    paths_ref = viterbi_paths(em_ref, case.half, base)  # reference Viterbi, reused at zero weights
    ref_labels = finish_paths(paths_ref, em_ref, track.source, base).labels  # reference cell's labels

    rows = []  # cell output rows
    by_weights: dict[tuple, list[dict]] = {}  # cells grouped by (K weight, T weight)
    for cell in cells:  # place every cell in its bucket
        by_weights.setdefault((cell["kin_w_dead"], cell["kin_w_team"]), []).append(cell)  # bucket by weight pair
    for (wd, wt), group in by_weights.items():  # emission/Viterbi depend only on the weights
        cfg_e = {**base, "kin_w_dead": wd, "kin_w_team": wt}  # config for this weight pair
        if wd == 0.0 and wt == 0.0:  # zero weights: reuse the reference decode
            em, paths = em_ref, paths_ref  # reuse reference decode
        else:
            em = emit(cfg_e)  # one emission per weight pair
            paths = viterbi_paths(em, case.half, cfg_e)  # one Viterbi per weight pair
        for cell in group:  # the other knobs only touch finish_paths
            labels = finish_paths(paths, em, track.source, {**cfg_e, **cell}).labels  # cell overrides applied
            rows.append({"match_id": match_id, "kind": kind, **cell, **scored(labels),
                         "pred_d_share": float((labels[in_match] == CODE_D).mean()),  # predicted D share
                         "changed_share": float((labels != ref_labels)[in_match].mean()),  # vs reference cell
                         "runtime_s": round(time.time() - t0, 1)})  # elapsed so far

    record_rows = []  # record-only output rows
    for name, override in RECORD.items():  # record-only cell: own emission + own decode
        cfg = {**base, **override}  # apply record override
        em_r = emit(cfg, kin_dead_all_frames=True)  # dead term applied to every frame
        labels = finish_paths(viterbi_paths(em_r, case.half, cfg), em_r, track.source, cfg).labels  # fresh decode
        record_rows.append({"match_id": match_id, "kind": kind, "cell": name, **scored(labels),  # same metrics
                            "pred_d_share": float((labels[in_match] == CODE_D).mean()),
                            "changed_share": float((labels != ref_labels)[in_match].mean())})

    # Diagnostics: team-swap mismatch (reference; largest T weight at each tier).
    swapped = 1 - case.slot_team  # home and away sides exchanged
    kin_sw = Kinematics(dead_llr=kin.dead_llr, team_logit=-kin.team_logit,  # mirrored team term
                        dead_ok=kin.dead_ok, team_ok=kin.team_ok)
    diag = {"match_id": match_id, "kind": kind}  # diag row starts

    def mismatch(cfg: dict, em_a, paths_a) -> float:  # equivariance under the team swap
        em_b = emit(cfg, slot_team=swapped, k=kin_sw)  # emission with swapped sides
        a = finish_paths(paths_a, em_a, track.source, cfg).labels  # original-team labels
        b = finish_paths(viterbi_paths(em_b, case.half, cfg), em_b, track.source, cfg).labels  # swapped labels
        mirrored = b.copy()  # copy to relabel in place
        mirrored[b == CODE_H], mirrored[b == CODE_A] = CODE_A, CODE_H  # undo the swap
        return float((mirrored != a)[in_match].mean())  # disagreeing frame share

    diag["equiv_mismatch_ref"] = mismatch(base, em_ref, paths_ref)  # at the reference cell
    if any(c["kin_w_team"] == WEIGHTS[-1] for c in cells):  # largest T weight present?
        cfg_t = {**base, "kin_w_team": WEIGHTS[-1]}  # largest-T config
        em_t = emit(cfg_t)  # emission at largest T
        paths_t = viterbi_paths(em_t, case.half, cfg_t)  # Viterbi at largest T
        for tier in (False, True):
            diag[f"equiv_mismatch_T{WEIGHTS[-1]:g}_tier{int(tier)}"] = mismatch(  # per gate tier
                {**cfg_t, "gate_kin_support": tier}, em_t, paths_t)

    if kind == KINDS[0]:
        # IDSSE side of the target parity checks Q4 / Q6 — unmasked ball, clean player tracks.
        track_u = build_ball_track(case.ball, case.half, base)  # unmasked ball track
        em_u = build_emission(track_u, case.players, case.players_s, case.slot_team, base)  # clean-data emission
        lab_u = finish_paths(viterbi_paths(em_u, case.half, base), em_u, track_u.source,
                             base).labels  # decode of the unmasked case
        sel = (in_match & (track_u.source == SOURCE_ATD) & kin_clean.team_ok  # frames usable for Q4
               & np.isin(lab_u, (CODE_H, CODE_A)))
        diag["q4_team_agreement"] = float(((kin_clean.team_logit[sel] > 0)  # kin sign vs decode
                                           == (lab_u[sel] == CODE_H)).mean())
        speed = _ball_speed_ms(track_u, base)  # ball-speed series
        alive = np.isin(lab_u, (CODE_H, CODE_A)) & in_match & np.isfinite(speed)  # live, finite frames
        diag["q6_ball_speed_p50"] = float(np.median(speed[alive]))  # median live ball speed
    return rows, record_rows, diag  # hand results to main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # first docstring line as help
    parser.add_argument("--smoke", action="store_true",  # tiny plumbing run
                        help="1 match, a handful of cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=7)  # pool size
    args = parser.parse_args()  # read command line

    from src import data as D  # raw-array loader

    arrays = D.load_arrays()  # ball + graded arrays
    run_lengths = empirical_run_lengths(np.isnan(arrays.ball).any(axis=1) & arrays.graded)  # missing-ball runs
    del arrays  # free the big arrays before forking

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS  # smoke runs one match only
    cells = p9_cells()  # full factorial grid
    if args.smoke:
        keep = [0, 1, 4, 12, 13, 100, 200, 371, 372, 500, len(cells) - 1]  # hand-picked smoke cells
        cells = [cells[i] for i in keep]  # keep only smoke cells
    for mid in MATCH_IDS:                             # warm caches BEFORE the pool
        load_match(mid)  # all seven, even in smoke (kin fit needs them)
        load_tackle_rows(mid)  # warm tackle cache too
    t0 = time.time()  # kin-fit timer
    kin, ref_rows = kin_for_matches(matches)  # leave-one-out kinematic models
    print(f"kinematic models (leave-one-out) ready at {time.time() - t0:.0f}s", flush=True)  # fit progress line
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(cells)} cells each", flush=True)  # run plan

    # one task per (match, mask kind), each carrying its match's two signal sets
    tasks = [(m, k, cells, run_lengths, kin[m][0], kin[m][1]) for m in matches for k in KINDS]
    from multiprocessing import Pool  # local import

    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}  # rows bucketed by mask kind
    record_rows: list[dict] = []  # record rows sink
    diag_rows: list[dict] = []  # diag rows sink
    with Pool(processes=min(args.workers, len(tasks))) as pool:  # never more workers than tasks
        for i, (rows, rrows, diag) in enumerate(pool.imap_unordered(run_task9, tasks), 1):  # any order
            by_kind[diag["kind"]].extend(rows)  # collect this kind's cells
            record_rows.extend(rrows)  # record-only rows
            diag_rows.append(diag)  # one diag per task
            print(f"[{i}/{len(tasks)}] {diag['match_id']} {diag['kind']} "
                  f"done at {time.time() - t0:.0f}s", flush=True)  # progress

    out_dir = Path("/tmp/stage9_smoke") if args.smoke else RESULTS  # smoke to /tmp, full to results/
    out_dir.mkdir(parents=True, exist_ok=True)  # create output folder
    order = {m: i for i, m in enumerate(MATCH_IDS)}  # canonical match order
    for kind, rows in by_kind.items():  # one matrix CSV per mask kind
        df = pd.DataFrame(rows)  # rows into one table
        df["_m"] = df["match_id"].map(order)  # temp sort key
        df = df.sort_values(P_PARAMS + ["_m"], kind="stable").drop(columns="_m")  # grid order, then match
        df.to_csv(out_dir / f"stage9_{kind}.csv", index=False)  # write this kind's matrix
        print(f"wrote stage9_{kind}.csv ({len(df)} rows)", flush=True)  # confirm write
    # record-only cells, one row each
    pd.DataFrame(record_rows).sort_values(["kind", "cell", "match_id"]).to_csv(
        out_dir / "stage9_record.csv", index=False)
    diag = pd.DataFrame(diag_rows).merge(pd.DataFrame(ref_rows), on="match_id")  # attach Q1/Q5 numbers
    diag.sort_values(["kind", "match_id"]).to_csv(out_dir / "stage9_diag.csv", index=False)  # write diagnostics
    print(f"total {time.time() - t0:.0f}s", flush=True)  # wall-clock total
    return 0  # success exit code


if __name__ == "__main__":
    sys.exit(main())  # exit status for the shell
