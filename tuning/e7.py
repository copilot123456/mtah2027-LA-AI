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

import sys  # sys.path bootstrap and exit code
import time  # runtime stamps in logs
from pathlib import Path  # path objects

import numpy as np  # array maths
import pandas as pd  # CSV read/write

ROOT = Path(__file__).resolve().parents[1]  # repo root
for p in (str(ROOT), str(ROOT / "evaluation")):  # expose repo + evaluation to imports
    if p not in sys.path:  # avoid duplicate entries
        sys.path.insert(0, p)  # make path importable

import possession_metrics as pm  # noqa: E402  # official scoring helpers
from config import CFG  # noqa: E402  # pitch and decoder config
from tuning.gate1 import FROZEN, cell  # noqa: E402  # frozen cell + row filter
from tuning.idsse import MATCH_IDS, derive_letters_controlled, load_match  # noqa: E402  # IDSSE loader + label sets
from tuning.simulate import empirical_run_lengths, mask_ball, phase_weights  # noqa: E402  # masks and run lengths
from tuning.stage4 import cues_for, load_event_rows, load_tackle_rows  # noqa: E402  # event + cue helpers
from tuning.sweep import MASK_SEED, _LETTERS, _score  # noqa: E402  # seed, letters, scoring

SET_PIECE_TAGS = ("ThrowIn", "FreeKick", "GoalKick", "CornerKick", "KickOff")  # set-piece event tags
E7_SEED_BASE = 700                     # rng = default_rng(MASK_SEED + 700 + Σ ord) (§7)
ON_VALUE = 1.0  # d_w_set_piece value on the ON side
KINDS = ("uniform", "phase_a2")        # the mask kinds of the four decision tables
FEATURES = ("out_of_pitch", "stationary", "near_boundary",  # dead features audited in band C
            "long_missing", "low_activity", "set_piece")
RES = ROOT / "tuning" / "results"  # stage matrices and the report live here


def noisy_setpiece(rows: np.ndarray, n_frames: int, match_id: str) -> np.ndarray:
    """Jitter ±1 s, drop 25 %, add round(0.10 × n_real) spurious — a priori, seeded (§7)."""
    rng = np.random.default_rng(MASK_SEED + E7_SEED_BASE + sum(ord(c) for c in match_id))  # seeded noise
    raw = np.asarray(rows, dtype=np.int64)  # real set-piece rows
    n_real = len(raw)  # how many real rows we start from
    jittered = np.clip(raw + rng.integers(-25, 26, size=n_real), 0, n_frames - 1)  # ±1 s jitter, clipped
    kept = jittered[rng.random(n_real) >= 0.25]  # drop 25 % of the real rows
    n_spur = int(round(0.10 * n_real))  # spurious count = 10 % of real
    if n_spur:
        kept = np.concatenate([kept, rng.integers(0, n_frames, size=n_spur)])  # add spurious rows
    return np.sort(kept)  # back to chronological order


def run_task(args) -> tuple[dict, list[dict], dict]:
    """One (match, mask kind): ON + fresh-OFF decodes and the feature audit."""
    match_id, kind, run_lengths = args  # unpack the task
    from src.ball_fill import build_ball_track  # gap-fill step on the masked ball
    from src.decoder import decode  # the possession decoder
    from src.emission import _dead_features, build_emission  # emission + dead-feature values

    case = load_match(match_id)  # cached match arrays
    n = len(case.letters)  # frames in the match
    raw_sp = load_event_rows(match_id, SET_PIECE_TAGS, "setpiece")  # real set-piece rows
    sp = noisy_setpiece(raw_sp, n, match_id)  # noisy input for the ON side

    rng = np.random.default_rng(MASK_SEED + sum(ord(c) for c in match_id))  # same seed as stage 4
    if kind == "uniform":
        masked, _ = mask_ball(case.ball, run_lengths, rng)  # uniform mask
    else:
        weights = phase_weights(cues_for(case, load_tackle_rows(match_id)), 2.0)  # α=2 cues
        masked, _ = mask_ball(case.ball, run_lengths, rng, weights=weights)  # phase mask

    base = dict(CFG)  # plain config for the OFF side
    base["fill_long_gap_s"] = 0.0                       # both decision tables are fill-off
    cfg_on = {**base, "d_w_set_piece": ON_VALUE}  # ON config: set-piece prior on
    track = build_ball_track(masked, case.half, base)  # ball track from masked input

    gt_orig = pm.label_codes(case.letters)  # orig label set (per-touch)
    gt_aligned_letters = derive_letters_controlled(case.letters, hold_s=3.0)  # aligned letters (E1)
    gt_aligned = pm.label_codes(gt_aligned_letters)  # aligned label codes

    em_off = build_emission(track, case.players, case.players_s, case.slot_team, base)  # OFF emission
    em_on = build_emission(track, case.players, case.players_s, case.slot_team, cfg_on,  # ON emission
                           setpiece_starts=sp)  # fed the noisy set-piece rows
    pred_off = _LETTERS[np.asarray(decode(em_off, track.source, case.half,  # OFF decode, frozen cell
                                          {**base, **FROZEN}).labels, dtype=np.int8) + 1]
    pred_on = _LETTERS[np.asarray(decode(em_on, track.source, case.half,  # ON decode, frozen cell
                                         {**cfg_on, **FROZEN}).labels, dtype=np.int8) + 1]
    po, pn = pm.label_codes(pred_off), pm.label_codes(pred_on)  # decoded letters -> codes

    gate_row: dict = {"match_id": match_id, "kind": kind}  # one gate row per task
    for label, gt in (("orig", gt_orig), ("aligned", gt_aligned)):  # both label sets
        s_off, s_on = _score(gt, po), _score(gt, pn)  # score OFF and ON
        gate_row[f"spell_f1_{label}_off"] = s_off["spell_f1"]  # OFF spell F1
        gate_row[f"spell_f1_{label}_on"] = s_on["spell_f1"]  # ON spell F1
        gate_row[f"macro_f1_{label}_off"] = s_off["macro_f1"]  # OFF macro F1
        gate_row[f"macro_f1_{label}_on"] = s_on["macro_f1"]  # ON macro F1
    for p, v in FROZEN.items():
        gate_row[p] = v  # record the frozen cell in the row

    # ---- audit (read-only): dead-feature firing by D-outcome bucket, aligned labels ----
    feats = _dead_features(track, case.players_s, base, setpiece_starts=sp)  # feature values per frame
    pred_l = np.asarray(pred_off, dtype=object)  # OFF predictions as letters
    gt_l = np.asarray(gt_aligned_letters, dtype=object)  # aligned GT letters
    buckets = {  # frame groups the audit reports on
        "d_correct": (pred_l == "D") & (gt_l == "D"),  # predicted D, truly D
        "missed_d": (pred_l != "D") & (gt_l == "D"),  # truly D, not predicted
        "false_d": (pred_l == "D") & (gt_l != "D"),  # predicted D, not truly D
        "base_all": np.ones(n, dtype=bool),  # every frame (baseline rate)
        "base_gt_d": gt_l == "D",  # all truly-D frames (baseline rate)
    }
    audit_rows = []  # audit rows accumulate here
    for bucket, mask in buckets.items():
        if not mask.any():
            continue  # empty bucket -> nothing to record
        for feat in FEATURES:
            audit_rows.append({  # one row per bucket x feature
                "match_id": match_id, "kind": kind, "bucket": bucket, "feature": feat,
                "firing_rate": float(np.asarray(feats[feat])[mask].mean()),  # share firing in bucket
                "n_frames": int(mask.sum()),  # bucket size
            })

    noise = {"match_id": match_id, "kind": kind, "raw": int(len(raw_sp)),  # real set-piece rows
             "noisy": int(len(sp)), "spurious": int(round(0.10 * len(raw_sp)))}  # after corruption
    return gate_row, audit_rows, noise  # all three task outputs


def load_stage4_frozen(kind: str) -> pd.DataFrame:
    """Committed OFF side: stage-4 frozen-cell rows for one mask kind."""
    return cell(pd.read_csv(RES / f"stage4_{kind}.csv"), FROZEN)  # keep the frozen cell rows


def main() -> int:
    from src import data as D  # deferred: heavy import

    arrays = D.load_arrays()  # provided arrays, only for run lengths here
    missing_graded = np.isnan(arrays.ball).any(axis=1) & arrays.graded  # graded missing frames
    run_lengths = empirical_run_lengths(missing_graded)  # empirical missing-run lengths

    t0 = time.time()  # total runtime starts here
    gate_rows: list[dict] = []  # ON-side gate rows
    audit_rows: list[dict] = []  # feature-audit rows
    noise_rows: list[dict] = []  # tagging-noise counts
    for mid in MATCH_IDS:  # 7 matches x 2 kinds = 14 ON decodes
        for kind in KINDS:
            g, a, nz = run_task((mid, kind, run_lengths))  # one task per pair
            gate_rows.append(g)  # store gate row
            audit_rows.extend(a)  # store audit rows
            noise_rows.append(nz)  # store noise counts
            print(f"[{len(gate_rows):2d}/14] {mid} {kind} "  # progress counter
                  f"({time.time() - t0:.0f}s)", flush=True)

    on = pd.DataFrame(gate_rows)  # ON-side matrix
    pd.DataFrame(audit_rows).to_csv(RES / "e7_audit.csv", index=False)  # audit matrix
    pd.DataFrame(noise_rows).to_csv(RES / "e7_noise.csv", index=False)  # noise counts

    # ---- merge with the committed stage-4 OFF rows; determinism check ----
    table_lines: list[tuple[str, float, int, float]] = []  # per-table summary rows
    primary_delta, primary_pos = 0.0, 0  # filled from the primary table below
    gate_out = []  # per-match gate output rows
    for kind in KINDS:
        off = load_stage4_frozen(kind).rename(  # committed OFF scores, renamed _off_s4
            columns={c: f"{c}_off_s4" for c in
                     ("spell_f1_orig", "spell_f1_aligned")})
        m = on[on["kind"] == kind].merge(  # pair fresh ON/OFF with committed OFF
            off[["match_id", "spell_f1_orig_off_s4", "spell_f1_aligned_off_s4"]],
            on="match_id")
        drift = float(max(  # fresh-vs-committed OFF drift (determinism check)
            (m["spell_f1_orig_off"] - m["spell_f1_orig_off_s4"]).abs().max(),
            (m["spell_f1_aligned_off"] - m["spell_f1_aligned_off_s4"]).abs().max()))
        for label in ("orig", "aligned"):
            d = m[f"spell_f1_{label}_on"] - m[f"spell_f1_{label}_off_s4"]  # ON − committed OFF
            n_pos = int((d > 0).sum())  # matches improved by ON
            name = f"{kind.replace('_', ' ')} × {label}"  # readable table name
            table_lines.append((name, float(d.mean()), n_pos, drift))  # mean Δ, positives, drift
            if kind == "phase_a2" and label == "aligned":
                primary_delta, primary_pos = float(d.mean()), n_pos  # the primary table
            for _, r in m.iterrows():
                gate_out.append({  # one detail row per match and table
                    "match_id": r["match_id"], "kind": kind, "label": label,
                    "off_committed": r[f"spell_f1_{label}_off_s4"],  # stage-4 OFF score
                    "off_fresh": r[f"spell_f1_{label}_off"],  # this run's OFF score
                    "on": r[f"spell_f1_{label}_on"],  # ON score
                    "delta_vs_committed": r[f"spell_f1_{label}_on"]
                    - r[f"spell_f1_{label}_off_s4"],  # ON − committed OFF
                    "off_drift": r[f"spell_f1_{label}_off"]
                    - r[f"spell_f1_{label}_off_s4"],  # fresh − committed OFF
                })
    pd.DataFrame(gate_out).to_csv(RES / "e7_gate.csv", index=False)  # gate matrix

    primary_ok = primary_delta > 0.02  # primary Δ must clear 0.02
    pos_ok = primary_pos >= 5  # at least 5 of 7 matches must improve
    harm_ok = all(dm >= -0.02 for name, dm, _, _ in table_lines  # no other table
                  if name != "phase a2 × aligned")  # may lose more than 0.02
    verdict = "ADOPT (d_w_set_piece = 1.0)" if (primary_ok and pos_ok and harm_ok) \
        else "REJECT → stays OFF"  # pinned gate rule

    # ---- report ----
    noise = pd.DataFrame(noise_rows)  # tagging-noise counts
    audit = pd.DataFrame(audit_rows)  # audit rows
    summary = (audit[audit["kind"] == "phase_a2"]  # phase masks only
               .groupby(["feature", "bucket"])["firing_rate"]
               .mean().unstack().reindex(FEATURES))  # mean rate per feature x bucket
    lines = [  # report skeleton: title, provenance, gate-table header
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
        lines.append(f"| {name} | {dm:+.4f} | {n_pos}/7 | {drift:.6f} |")  # one row per table
    lines += [  # verdict paragraph, noise summary, audit table header
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
        row = summary.loc[feat]  # this feature's rates per bucket
        cells = []  # formatted bucket values
        for b in ("d_correct", "missed_d", "false_d", "base_all", "base_gt_d"):
            v = row[b] if b in summary.columns else np.nan  # missing bucket -> NaN
            cells.append(f"{v:.3f}" if np.isfinite(v) else "—")  # format or em-dash
        lines.append(f"| {feat} | " + " | ".join(cells) + " |")  # one row per feature
    lines += [  # reading note and reproduce commands
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
    report = RES / "report_e7.md"  # report output path
    report.write_text("\n".join(lines), encoding="utf-8")  # write the markdown report

    print("\n=== E7 GATE ===")  # gate section header
    for name, dm, n_pos, drift in table_lines:
        print(f"  {name:24s} {dm:+.4f}  ({n_pos}/7)  drift {drift:.6f}")  # one console line per table
    print(f"VERDICT: {verdict}   (primary {primary_delta:+.4f}, {primary_pos}/7)")  # print the verdict
    print(f"wrote {RES / 'e7_gate.csv'}, e7_audit.csv, e7_noise.csv")  # list written matrices
    print(f"wrote {report}  ({time.time() - t0:.0f}s)")  # report path and runtime
    return 0  # zero means success


if __name__ == "__main__":  # module entry point
    sys.exit(main())  # run as a script
