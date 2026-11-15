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
from src.kinematics import Kinematics, apply_model  # noqa: E402
from tuning.idsse import MATCH_IDS, derive_letters_controlled, load_match  # noqa: E402
from tuning.kin_fit import degrade_players, fit_models, match_features  # noqa: E402
from tuning.simulate import empirical_run_lengths  # noqa: E402
from tuning.stage4 import load_tackle_rows  # noqa: E402
from tuning.stage5 import FROZEN_DECODER, KINDS, build_mask  # noqa: E402
from tuning.stage8 import team_switch_latency  # noqa: E402
from tuning.sweep import MASK_SEED, _LETTERS, _score  # noqa: E402

P_PARAMS = ["dwell_coalesce", "kin_w_dead", "kin_w_team", "gate_kin_support",
            "switch_delay_s", "switch_delay_slope"]
WEIGHTS = [0.0, 0.03, 0.1, 0.3]
DELAYS = [0.64, 0.80, 0.96, 1.12]
SLOPES = [0.0, 0.03, 0.06]
REFERENCE = {"dwell_coalesce": False, "kin_w_dead": 0.0, "kin_w_team": 0.0,
             "gate_kin_support": False, "switch_delay_s": 0.64, "switch_delay_slope": 0.0}
RECORD = {"R_kin_all": {"kin_w_dead": 0.1}}        # the dead term on every frame — never adopted
KIN_SEED_BASE = 900                                # rng = default_rng(MASK_SEED + 900 + Σ ord)
RESULTS = ROOT / "tuning" / "results"


def p9_cells() -> list[dict]:
    """The factorial; the gate tier is immaterial at zero weights and is counted once."""
    cells = []
    for c, wd, wt, tier, delay, slope in product((False, True), WEIGHTS, WEIGHTS,
                                                 (False, True), DELAYS, SLOPES):
        if tier and wd == 0.0 and wt == 0.0:
            continue
        cells.append({"dwell_coalesce": c, "kin_w_dead": wd, "kin_w_team": wt,
                      "gate_kin_support": tier, "switch_delay_s": delay,
                      "switch_delay_slope": slope})
    return cells


def kin_for_matches(matches: list[str]) -> tuple[dict, list[dict]]:
    """{match: (degraded-track signals, clean-track signals)} with leave-one-out models, and
    the IDSSE side of the parity checks Q1 / Q5."""
    feats = {}
    for mid in MATCH_IDS:                        # the fit always uses the other six
        feats[mid] = match_features(load_match(mid))
    out, ref_rows = {}, []
    for mid in matches:
        case = load_match(mid)
        model = fit_models(feats, exclude=mid)
        rng = np.random.default_rng(MASK_SEED + KIN_SEED_BASE + sum(ord(c) for c in mid))
        deg = match_features(case, *degrade_players(case.players, case.players_s, rng))
        f = feats[mid]
        out[mid] = (apply_model(deg["x_dead"], deg["x_team"], model),
                    apply_model(f["x_dead"], f["x_team"], model))
        m = f["in_match"]
        ref_rows.append({"match_id": mid,
                         "speed_med_p50": float(np.nanmedian(f["speed_med"][m])),
                         "n_vis_mean": float(np.nanmean(f["n_vis"][m])),
                         "n_vis_mean_degraded": float(np.nanmean(deg["n_vis"][m]))})
    return out, ref_rows


def run_task9(args) -> tuple[list[dict], list[dict], dict]:
    """All cells + the record-only cell + diagnostics for one (match, mask kind)."""
    match_id, kind, cells, run_lengths, kin, kin_clean = args
    t0 = time.time()
    case = load_match(match_id)
    if case.info.get("speed_units") != "m/s":
        raise ValueError(f"{match_id}: cache is not in m/s — run the loader once first")
    masked, _ = build_mask(case, kind, run_lengths, load_tackle_rows(match_id))

    from src.ball_fill import SOURCE_ATD, build_ball_track
    from src.decoder import CODE_A, CODE_D, CODE_H, finish_paths, viterbi_paths
    from src.emission import _ball_speed_ms, build_emission

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

    def emit(cfg: dict, slot_team=None, k=kin, **kw):
        return build_emission(track, case.players, case.players_s,
                              case.slot_team if slot_team is None else slot_team, cfg,
                              kin=k, **kw)

    em_ref = emit(base)
    paths_ref = viterbi_paths(em_ref, case.half, base)
    ref_labels = finish_paths(paths_ref, em_ref, track.source, base).labels

    rows = []
    by_weights: dict[tuple, list[dict]] = {}
    for cell in cells:
        by_weights.setdefault((cell["kin_w_dead"], cell["kin_w_team"]), []).append(cell)
    for (wd, wt), group in by_weights.items():
        cfg_e = {**base, "kin_w_dead": wd, "kin_w_team": wt}
        if wd == 0.0 and wt == 0.0:
            em, paths = em_ref, paths_ref
        else:
            em = emit(cfg_e)
            paths = viterbi_paths(em, case.half, cfg_e)
        for cell in group:
            labels = finish_paths(paths, em, track.source, {**cfg_e, **cell}).labels
            rows.append({"match_id": match_id, "kind": kind, **cell, **scored(labels),
                         "pred_d_share": float((labels[in_match] == CODE_D).mean()),
                         "changed_share": float((labels != ref_labels)[in_match].mean()),
                         "runtime_s": round(time.time() - t0, 1)})

    record_rows = []
    for name, override in RECORD.items():
        cfg = {**base, **override}
        em_r = emit(cfg, kin_dead_all_frames=True)
        labels = finish_paths(viterbi_paths(em_r, case.half, cfg), em_r, track.source, cfg).labels
        record_rows.append({"match_id": match_id, "kind": kind, "cell": name, **scored(labels),
                            "pred_d_share": float((labels[in_match] == CODE_D).mean()),
                            "changed_share": float((labels != ref_labels)[in_match].mean())})

    # Diagnostics: team-swap mismatch (reference; largest T weight at each tier).
    swapped = 1 - case.slot_team
    kin_sw = Kinematics(dead_llr=kin.dead_llr, team_logit=-kin.team_logit,
                        dead_ok=kin.dead_ok, team_ok=kin.team_ok)
    diag = {"match_id": match_id, "kind": kind}

    def mismatch(cfg: dict, em_a, paths_a) -> float:
        em_b = emit(cfg, slot_team=swapped, k=kin_sw)
        a = finish_paths(paths_a, em_a, track.source, cfg).labels
        b = finish_paths(viterbi_paths(em_b, case.half, cfg), em_b, track.source, cfg).labels
        mirrored = b.copy()
        mirrored[b == CODE_H], mirrored[b == CODE_A] = CODE_A, CODE_H
        return float((mirrored != a)[in_match].mean())

    diag["equiv_mismatch_ref"] = mismatch(base, em_ref, paths_ref)
    if any(c["kin_w_team"] == WEIGHTS[-1] for c in cells):
        cfg_t = {**base, "kin_w_team": WEIGHTS[-1]}
        em_t = emit(cfg_t)
        paths_t = viterbi_paths(em_t, case.half, cfg_t)
        for tier in (False, True):
            diag[f"equiv_mismatch_T{WEIGHTS[-1]:g}_tier{int(tier)}"] = mismatch(
                {**cfg_t, "gate_kin_support": tier}, em_t, paths_t)

    if kind == KINDS[0]:
        # IDSSE side of the target parity checks Q4 / Q6 — unmasked ball, clean player tracks.
        track_u = build_ball_track(case.ball, case.half, base)
        em_u = build_emission(track_u, case.players, case.players_s, case.slot_team, base)
        lab_u = finish_paths(viterbi_paths(em_u, case.half, base), em_u, track_u.source,
                             base).labels
        sel = (in_match & (track_u.source == SOURCE_ATD) & kin_clean.team_ok
               & np.isin(lab_u, (CODE_H, CODE_A)))
        diag["q4_team_agreement"] = float(((kin_clean.team_logit[sel] > 0)
                                           == (lab_u[sel] == CODE_H)).mean())
        speed = _ball_speed_ms(track_u, base)
        alive = np.isin(lab_u, (CODE_H, CODE_A)) & in_match & np.isfinite(speed)
        diag["q6_ball_speed_p50"] = float(np.median(speed[alive]))
    return rows, record_rows, diag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="1 match, a handful of cells, all four kinds, writes to /tmp")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    from src import data as D

    arrays = D.load_arrays()
    run_lengths = empirical_run_lengths(np.isnan(arrays.ball).any(axis=1) & arrays.graded)
    del arrays

    matches = [MATCH_IDS[0]] if args.smoke else MATCH_IDS
    cells = p9_cells()
    if args.smoke:
        keep = [0, 1, 4, 12, 13, 100, 200, 371, 372, 500, len(cells) - 1]
        cells = [cells[i] for i in keep]
    for mid in MATCH_IDS:                             # warm caches BEFORE the pool
        load_match(mid)
        load_tackle_rows(mid)
    t0 = time.time()
    kin, ref_rows = kin_for_matches(matches)
    print(f"kinematic models (leave-one-out) ready at {time.time() - t0:.0f}s", flush=True)
    print(f"{len(matches)} matches x {len(KINDS)} kinds, {len(cells)} cells each", flush=True)

    tasks = [(m, k, cells, run_lengths, kin[m][0], kin[m][1]) for m in matches for k in KINDS]
    from multiprocessing import Pool

    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}
    record_rows: list[dict] = []
    diag_rows: list[dict] = []
    with Pool(processes=min(args.workers, len(tasks))) as pool:
        for i, (rows, rrows, diag) in enumerate(pool.imap_unordered(run_task9, tasks), 1):
            by_kind[diag["kind"]].extend(rows)
            record_rows.extend(rrows)
            diag_rows.append(diag)
            print(f"[{i}/{len(tasks)}] {diag['match_id']} {diag['kind']} "
                  f"done at {time.time() - t0:.0f}s", flush=True)

    out_dir = Path("/tmp/stage9_smoke") if args.smoke else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {m: i for i, m in enumerate(MATCH_IDS)}
    for kind, rows in by_kind.items():
        df = pd.DataFrame(rows)
        df["_m"] = df["match_id"].map(order)
        df = df.sort_values(P_PARAMS + ["_m"], kind="stable").drop(columns="_m")
        df.to_csv(out_dir / f"stage9_{kind}.csv", index=False)
        print(f"wrote stage9_{kind}.csv ({len(df)} rows)", flush=True)
    pd.DataFrame(record_rows).sort_values(["kind", "cell", "match_id"]).to_csv(
        out_dir / "stage9_record.csv", index=False)
    diag = pd.DataFrame(diag_rows).merge(pd.DataFrame(ref_rows), on="match_id")
    diag.sort_values(["kind", "match_id"]).to_csv(out_dir / "stage9_diag.csv", index=False)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
