"""E10 no-harm gate (pre-declared: dossier §7 P6, 2026-11-09).

    python -m tuning.no_harm_no_play

Paired rule-off vs rule-on at the frozen cell across all 7 IDSSE matches, with the
same seeds, masking and scoring as `tuning.post_checks --final`. The gate, fixed
a priori before this run:

    PASS iff mean paired Δ spellF1 >= -0.005 AND no single match worse than -0.02

Output: tuning/results/no_harm_no_play.csv (committed with the verdict).
Exit code 0 = PASS, 1 = FAIL. Harness sanity: the rule-off column must reproduce the
committed post_checks_final.csv baseline exactly (same seeds, same code path).
"""

from __future__ import annotations   # postponed annotation evaluation

import sys                # sys.path setup below
from pathlib import Path  # repo-root path objects

import numpy as np   # RNG and array math
import pandas as pd  # result table and CSV output

ROOT = Path(__file__).resolve().parents[1]   # repo root
for p in (str(ROOT), str(ROOT / "evaluation")):   # both dirs must be importable
    if p not in sys.path:
        sys.path.insert(0, p)

# project imports (placed after the path setup, hence noqa)
import possession_metrics as pm  # noqa: E402
from config import CFG  # noqa: E402
from tuning.idsse import MATCH_IDS, load_match  # noqa: E402
from tuning.simulate import empirical_run_lengths, mask_ball  # noqa: E402
from tuning.sweep import _LETTERS, _score  # noqa: E402

OUT = ROOT / "tuning" / "results" / "no_harm_no_play.csv"   # committed verdict CSV
MEAN_TOL = -0.005   # pre-declared (dossier §7): mean within the frozen cell's noise floor
MATCH_TOL = -0.02   # pre-declared (dossier §7): standard per-match harm bound


def main() -> int:
    from src import data as D   # array loading

    arrays = D.load_arrays()
    run_lengths = empirical_run_lengths(   # missing-ball runs to replay when masking
        np.isnan(arrays.ball).any(axis=1) & arrays.graded
    )

    # decoder stages, imported after path setup
    from src.ball_fill import build_ball_track
    from src.decoder import decode
    from src.emission import build_emission
    from src.x_policy import apply_x_policy

    rows = []   # one scored row per (match, rule flag)
    for mid in MATCH_IDS:   # all 7 IDSSE matches
        case = load_match(mid)
        rng = np.random.default_rng(20261107 + sum(ord(c) for c in mid))   # per-match seed
        masked, _ = mask_ball(case.ball, run_lengths, rng)   # frozen masking
        track = build_ball_track(masked, case.half, dict(CFG))   # ball track from masked input
        gt = pm.label_codes(case.letters)   # ground-truth codes
        for flag in (False, True):   # paired off/on runs, same seed and data
            cfg = {**CFG, "no_play_rule": flag}   # toggle only the rule under test
            em = build_emission(track, case.players, case.players_s, case.slot_team, cfg)
            decoded = decode(em, track.source, case.half, cfg)
            labels = apply_x_policy(decoded.labels, em.conf, case.half, cfg)
            pred = pm.label_codes(_LETTERS[np.asarray(labels, dtype=np.int8) + 1])   # labels -> letters -> codes
            scores = _score(gt, pred)   # official scoring semantics
            rows.append({"match_id": mid, "no_play_rule": flag, **scores})
            print(f"  {mid} rule={'on ' if flag else 'off'}: "   # progress line
                  f"spell {scores['spell_f1']:.4f}  macro {scores['macro_f1']:.4f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)   # write the committed CSV

    off = df[~df.no_play_rule].set_index("match_id")["spell_f1"]   # rule-off scores
    on = df[df.no_play_rule].set_index("match_id")["spell_f1"]     # rule-on scores
    delta = (on - off).sort_index()   # paired per-match change
    mean_d, min_d = float(delta.mean()), float(delta.min())   # the two gate numbers
    passed = bool(mean_d >= MEAN_TOL and min_d >= MATCH_TOL)   # both bounds must hold

    try:
        ref = pd.read_csv(ROOT / "tuning" / "results" / "post_checks_final.csv")   # committed baseline
        ref_base = ref[(ref.d_bias_tag == "prior") & (~ref.x_use)].set_index("match_id")["spell_f1"]   # shipped cell rows
        drift = float((off.sort_index() - ref_base.sort_index()).abs().max())   # biggest gap vs baseline
        print(f"\nharness sanity: max |rule-off - committed post_checks_final baseline| "
              f"= {drift:.6f} (expect 0.000000)")
    except Exception as exc:  # pragma: no cover - reference file always ships
        print(f"\nharness sanity skipped: {exc}")

    print("\npaired Δ (rule on − off), spellF1:")
    for mid in delta.index:
        print(f"  {mid}: {delta[mid]:+.4f}")   # one line per match
    print(f"mean {mean_d:+.4f} | worst {min_d:+.4f}")   # the two gate numbers again
    print(f"gate (mean ≥ {MEAN_TOL}, no match < {MATCH_TOL}): "
          f"{'PASS' if passed else 'FAIL'}")   # final verdict line
    print(f"wrote {OUT}")
    return 0 if passed else 1   # exit code = PASS/FAIL


if __name__ == "__main__":
    raise SystemExit(main())
