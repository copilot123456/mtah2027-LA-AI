"""E7 — set-piece dead-ball prior gate + band-C D-feature audit (dossier §7, 2026-11-09).

    python -m tuning.e7        # gate + audit -> e7_gate.csv, e7_audit.csv, report_e7.md

Frozen cell only, four decision tables (uniform/phase α=2 × original/aligned labels),
primary = phase × aligned — the standing Gate-1 primary. The OFF side reuses the
committed stage-4 frozen-cell rows (identical masks, seeds and code path: with
`d_w_set_piece = 0` the extra dead-logit term is a no-op); a fresh OFF decode is
computed alongside as a determinism check against those committed rows. The ON side is
14 decodes at `d_w_set_piece = 1.0` with noisy IDSSE set-piece input (jitter ±1 s,
drop 25 %, add round(0.10 × n_real) spurious — a priori, seeded per §7).

The audit is read-only feature bookkeeping against the E1-aligned labels: how each dead
feature fires on {D-correct, missed-D, false-D} frames, including f_sp's coverage of
missed-D frames. It never adjusts a weight — the gate decides alone.
"""

from __future__ import annotations

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
from tuning.gate1 import FROZEN, cell  # noqa: E402
from tuning.idsse import MATCH_IDS, derive_letters_controlled, load_match  # noqa: E402
from tuning.simulate import empirical_run_lengths, mask_ball, phase_weights  # noqa: E402
from tuning.stage4 import cues_for, load_event_rows, load_tackle_rows  # noqa: E402
from tuning.sweep import MASK_SEED, _LETTERS, _score  # noqa: E402

SET_PIECE_TAGS = ("ThrowIn", "FreeKick", "GoalKick", "CornerKick", "KickOff")
E7_SEED_BASE = 700                     # rng = default_rng(MASK_SEED + 700 + Σ ord) (§7)
ON_VALUE = 1.0
KINDS = ("uniform", "phase_a2")        # the mask kinds of the four decision tables
FEATURES = ("out_of_pitch", "stationary", "near_boundary",
            "long_missing", "low_activity", "set_piece")
RES = ROOT / "tuning" / "results"


def noisy_setpiece(rows: np.ndarray, n_frames: int, match_id: str) -> np.ndarray:
    """Jitter ±1 s, drop 25 %, add round(0.10 × n_real) spurious — a priori, seeded (§7)."""
    rng = np.random.default_rng(MASK_SEED + E7_SEED_BASE + sum(ord(c) for c in match_id))
    raw = np.asarray(rows, dtype=np.int64)
    n_real = len(raw)
    jittered = np.clip(raw + rng.integers(-25, 26, size=n_real), 0, n_frames - 1)
    kept = jittered[rng.random(n_real) >= 0.25]
    n_spur = int(round(0.10 * n_real))
    if n_spur:
        kept = np.concatenate([kept, rng.integers(0, n_frames, size=n_spur)])
    return np.sort(kept)


def run_task(args) -> tuple[dict, list[dict], dict]:
    """One (match, mask kind): ON + fresh-OFF decodes and the feature audit."""
    match_id, kind, run_lengths = args
    from src.ball_fill import build_ball_track
    from src.decoder import decode
    from src.emission import _dead_features, build_emission

    case = load_match(match_id)
    n = len(case.letters)
    raw_sp = load_event_rows(match_id, SET_PIECE_TAGS, "setpiece")
    sp = noisy_setpiece(raw_sp, n, match_id)

    rng = np.random.default_rng(MASK_SEED + sum(ord(c) for c in match_id))
    if kind == "uniform":
        masked, _ = mask_ball(case.ball, run_lengths, rng)
    else:
        weights = phase_weights(cues_for(case, load_tackle_rows(match_id)), 2.0)
        masked, _ = mask_ball(case.ball, run_lengths, rng, weights=weights)

    base = dict(CFG)
    base["fill_long_gap_s"] = 0.0                       # both decision tables are fill-off
    cfg_on = {**base, "d_w_set_piece": ON_VALUE}
    track = build_ball_track(masked, case.half, base)

    gt_orig = pm.label_codes(case.letters)
    gt_aligned_letters = derive_letters_controlled(case.letters, hold_s=3.0)
    gt_aligned = pm.label_codes(gt_aligned_letters)

    em_off = build_emission(track, case.players, case.players_s, case.slot_team, base)
    em_on = build_emission(track, case.players, case.players_s, case.slot_team, cfg_on,
                           setpiece_starts=sp)
    pred_off = _LETTERS[np.asarray(decode(em_off, track.source, case.half,
                                          {**base, **FROZEN}).labels, dtype=np.int8) + 1]
    pred_on = _LETTERS[np.asarray(decode(em_on, track.source, case.half,
                                         {**cfg_on, **FROZEN}).labels, dtype=np.int8) + 1]
    po, pn = pm.label_codes(pred_off), pm.label_codes(pred_on)

    gate_row: dict = {"match_id": match_id, "kind": kind}
    for label, gt in (("orig", gt_orig), ("aligned", gt_aligned)):
        s_off, s_on = _score(gt, po), _score(gt, pn)
        gate_row[f"spell_f1_{label}_off"] = s_off["spell_f1"]
        gate_row[f"spell_f1_{label}_on"] = s_on["spell_f1"]
        gate_row[f"macro_f1_{label}_off"] = s_off["macro_f1"]
        gate_row[f"macro_f1_{label}_on"] = s_on["macro_f1"]
    for p, v in FROZEN.items():
        gate_row[p] = v

    # ---- audit (read-only): dead-feature firing by D-outcome bucket, aligned labels ----
    feats = _dead_features(track, case.players_s, base, setpiece_starts=sp)
    pred_l = np.asarray(pred_off, dtype=object)
    gt_l = np.asarray(gt_aligned_letters, dtype=object)
    buckets = {
        "d_correct": (pred_l == "D") & (gt_l == "D"),
        "missed_d": (pred_l != "D") & (gt_l == "D"),
        "false_d": (pred_l == "D") & (gt_l != "D"),
        "base_all": np.ones(n, dtype=bool),
        "base_gt_d": gt_l == "D",
    }
    audit_rows = []
    for bucket, mask in buckets.items():
        if not mask.any():
            continue
        for feat in FEATURES:
            audit_rows.append({
                "match_id": match_id, "kind": kind, "bucket": bucket, "feature": feat,
                "firing_rate": float(np.asarray(feats[feat])[mask].mean()),
                "n_frames": int(mask.sum()),
            })

    noise = {"match_id": match_id, "kind": kind, "raw": int(len(raw_sp)),
             "noisy": int(len(sp)), "spurious": int(round(0.10 * len(raw_sp)))}
    return gate_row, audit_rows, noise


def load_stage4_frozen(kind: str) -> pd.DataFrame:
    """Committed OFF side: stage-4 frozen-cell rows for one mask kind."""
    return cell(pd.read_csv(RES / f"stage4_{kind}.csv"), FROZEN)


def main() -> int:
    from src import data as D

    arrays = D.load_arrays()
    missing_graded = np.isnan(arrays.ball).any(axis=1) & arrays.graded
    run_lengths = empirical_run_lengths(missing_graded)

    t0 = time.time()
    gate_rows: list[dict] = []
    audit_rows: list[dict] = []
    noise_rows: list[dict] = []
    for mid in MATCH_IDS:
        for kind in KINDS:
            g, a, nz = run_task((mid, kind, run_lengths))
            gate_rows.append(g)
            audit_rows.extend(a)
            noise_rows.append(nz)
            print(f"[{len(gate_rows):2d}/14] {mid} {kind} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    on = pd.DataFrame(gate_rows)
    pd.DataFrame(audit_rows).to_csv(RES / "e7_audit.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(RES / "e7_noise.csv", index=False)

    # ---- merge with the committed stage-4 OFF rows; determinism check ----
    table_lines: list[tuple[str, float, int, float]] = []
    primary_delta, primary_pos = 0.0, 0
    gate_out = []
    for kind in KINDS:
        off = load_stage4_frozen(kind).rename(
            columns={c: f"{c}_off_s4" for c in
                     ("spell_f1_orig", "spell_f1_aligned")})
        m = on[on["kind"] == kind].merge(
            off[["match_id", "spell_f1_orig_off_s4", "spell_f1_aligned_off_s4"]],
            on="match_id")
        drift = float(max(
            (m["spell_f1_orig_off"] - m["spell_f1_orig_off_s4"]).abs().max(),
            (m["spell_f1_aligned_off"] - m["spell_f1_aligned_off_s4"]).abs().max()))
        for label in ("orig", "aligned"):
            d = m[f"spell_f1_{label}_on"] - m[f"spell_f1_{label}_off_s4"]
            n_pos = int((d > 0).sum())
            name = f"{kind.replace('_', ' ')} × {label}"
            table_lines.append((name, float(d.mean()), n_pos, drift))
            if kind == "phase_a2" and label == "aligned":
                primary_delta, primary_pos = float(d.mean()), n_pos
            for _, r in m.iterrows():
                gate_out.append({
                    "match_id": r["match_id"], "kind": kind, "label": label,
                    "off_committed": r[f"spell_f1_{label}_off_s4"],
                    "off_fresh": r[f"spell_f1_{label}_off"],
                    "on": r[f"spell_f1_{label}_on"],
                    "delta_vs_committed": r[f"spell_f1_{label}_on"]
                    - r[f"spell_f1_{label}_off_s4"],
                    "off_drift": r[f"spell_f1_{label}_off"]
                    - r[f"spell_f1_{label}_off_s4"],
                })
    pd.DataFrame(gate_out).to_csv(RES / "e7_gate.csv", index=False)

    primary_ok = primary_delta > 0.02
    pos_ok = primary_pos >= 5
    harm_ok = all(dm >= -0.02 for name, dm, _, _ in table_lines
                  if name != "phase a2 × aligned")
    verdict = "ADOPT (d_w_set_piece = 1.0)" if (primary_ok and pos_ok and harm_ok) \
        else "REJECT → stays OFF"

    # ---- report ----
    noise = pd.DataFrame(noise_rows)
    audit = pd.DataFrame(audit_rows)
    summary = (audit[audit["kind"] == "phase_a2"]
               .groupby(["feature", "bucket"])["firing_rate"]
               .mean().unstack().reindex(FEATURES))
    lines = [
        "# E7 — set-piece dead-ball prior: gate + D-feature audit",
        "",
        "Pre-declared: `dossier.md` §7 (2026-11-09, commit `7388a47`) — frozen cell only,",
        "four decision tables, primary = phase α=2 × aligned, ON value 1.0, gate",
        "> 0.02 paired (≥ 5/7) + no-harm. OFF side = committed stage-4 frozen rows;",
        "ON side = 14 fresh decodes with noisy IDSSE set-piece input (jitter ±1 s,",
        "drop 25 %, +10 % spurious, seeded).",
        "",
        "## Gate (ON − OFF, mean spellF1; OFF fresh-vs-committed drift last)",
        "",
        "| table | Δ | positive | OFF drift |",
        "|---|---|---|---|",
    ]
    for name, dm, n_pos, drift in table_lines:
        lines.append(f"| {name} | {dm:+.4f} | {n_pos}/7 | {drift:.6f} |")
    lines += [
        "",
        f"**Verdict: {verdict}** — primary Δ {primary_delta:+.4f} "
        f"(> 0.02: {'yes' if primary_ok else 'no'}), "
        f"positive {primary_pos}/7 (≥ 5: {'yes' if pos_ok else 'no'}), "
        f"no-harm: {'yes' if harm_ok else 'no'}.",
        "",
        "The plan's prior (+0.00 … +0.01) sits below the bar; a rejection here is the",
        "expected outcome and leaves the frozen submission untouched.",
        "",
        "## Tagging-noise simulation (a priori; per match raw → noisy, +spurious)",
        "",
        f"raw mean {noise['raw'].mean():.0f} (range {noise['raw'].min()}–"
        f"{noise['raw'].max()}), noisy mean {noise['noisy'].mean():.0f}, "
        f"spurious {int(noise['spurious'].iloc[0])} per match.",
        "",
        "## D-feature audit (band C; phase masks, firing rate by bucket, 7-match mean)",
        "",
        "| feature | D-correct | missed-D | false-D | base (all) | base (gt D) |",
        "|---|---|---|---|---|---|",
    ]
    for feat in FEATURES:
        row = summary.loc[feat]
        cells = []
        for b in ("d_correct", "missed_d", "false_d", "base_all", "base_gt_d"):
            v = row[b] if b in summary.columns else np.nan
            cells.append(f"{v:.3f}" if np.isfinite(v) else "—")
        lines.append(f"| {feat} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Read: which features separate missed-D from D-correct frames is the external",
        "picture of the band-C `f1_D low` flag; `set_piece`'s missed-D coverage is the",
        "E7 upside estimate. The audit informs — the gate above is the only decider.",
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python -m tuning.e7        # this gate + audit (~3 min)",
        ".venv/bin/python -m pytest -q",
        "```",
        "",
    ]
    report = RES / "report_e7.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== E7 GATE ===")
    for name, dm, n_pos, drift in table_lines:
        print(f"  {name:24s} {dm:+.4f}  ({n_pos}/7)  drift {drift:.6f}")
    print(f"VERDICT: {verdict}   (primary {primary_delta:+.4f}, {primary_pos}/7)")
    print(f"wrote {RES / 'e7_gate.csv'}, e7_audit.csv, e7_noise.csv")
    print(f"wrote {report}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
