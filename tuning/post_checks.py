"""Post-selection checks (P5): re-derive d_bias externally; ablate the X-policy.

    python -m tuning.post_checks            # stage-1 cell, kinematic scenario (as recorded)
    python -m tuning.post_checks --final    # shipped config.py cell, fill tier as shipped

Both checks run across all 7 IDSSE matches (same seeds as the sweep) and carry a
pre-declared adoption rule:

    adopt the variant only if mean spellF1 gain > 0.02 over the shipped value
    (external differences of 0.01-0.02 are ties — plan R-U4)

d_bias:    shipped -0.70 (prior, logit of ~1/3 dead) vs logit of the pooled
           IDSSE provider-label dead share (external_fit candidate).
x_use:     shipped False (emit-dominates analysis) vs True (docs rule active).

The default mode reproduces the committed `tuning/results/post_checks.csv`: the
stage-1 chosen cell under the `kinematic` scenario (fill tier on, as config.py stood
when the checks were first run on 2026-11-07). `--final` re-runs the same two checks
at the frozen cell with the shipped fill setting and writes `post_checks_final.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import possession_metrics as pm  # noqa: E402
from config import CFG  # noqa: E402
from tuning.idsse import MATCH_IDS, load_match  # noqa: E402
from tuning.simulate import empirical_run_lengths, mask_ball  # noqa: E402
from tuning.sweep import PARAMS, RESULTS, _LETTERS, _score, flat_select  # noqa: E402

ADOPT_GAIN = 0.02
OUT = ROOT / "tuning" / "results" / "post_checks.csv"
OUT_FINAL = ROOT / "tuning" / "results" / "post_checks_final.csv"
STAGE1_FILL_LONG_GAP_S = 10.0   # the prior value in force when the stage-1 checks ran


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true",
                        help="evaluate at the shipped config.py cell with the shipped fill "
                             "setting (default: stage-1 chosen cell, kinematic scenario)")
    parser.add_argument("--out", default=None, help="output CSV (default per mode)")
    args = parser.parse_args()

    if args.final:
        chosen = {p: float(CFG[p]) for p in PARAMS}
        base = dict(CFG)
        out_path = Path(args.out) if args.out else OUT_FINAL
    else:
        df = pd.read_csv(RESULTS)
        chosen, _ = flat_select(df)
        base = {**CFG, "fill_long_gap_s": STAGE1_FILL_LONG_GAP_S}
        out_path = Path(args.out) if args.out else OUT
    print("chosen cell:", chosen, flush=True)
    print(f"fill_long_gap_s = {base['fill_long_gap_s']}", flush=True)

    from src import data as D

    arrays = D.load_arrays()
    run_lengths = empirical_run_lengths(
        np.isnan(arrays.ball).any(axis=1) & arrays.graded)

    # --- external d_bias candidate from provider labels (pooled) ---
    dead_total = alive_total = 0
    for mid in MATCH_IDS:
        case = load_match(mid)
        letters = case.letters
        in_match = letters != "X"
        dead_total += int(((letters == "D") & in_match).sum())
        alive_total += int(in_match.sum())
    d_share = dead_total / alive_total
    d_bias_fit = float(np.log(d_share / (1 - d_share)))
    print(f"IDSSE pooled D share {d_share:.3f} -> d_bias external_fit {d_bias_fit:+.3f}", flush=True)

    from src.ball_fill import build_ball_track
    from src.decoder import decode
    from src.emission import build_emission
    from src.x_policy import apply_x_policy

    variants = [
        ("prior", CFG["d_bias"], False),
        ("prior", CFG["d_bias"], True),
        ("fit", d_bias_fit, False),
        ("fit", d_bias_fit, True),
    ]
    rows = []
    for mid in MATCH_IDS:
        case = load_match(mid)
        rng = np.random.default_rng(20261107 + sum(ord(c) for c in mid))
        masked, _ = mask_ball(case.ball, run_lengths, rng)
        track = build_ball_track(masked, case.half, base)
        gt = pm.label_codes(case.letters)
        runs = pm.valid_runs(gt)
        for tag, bias, x_on in variants:
            cfg = {**base, **chosen, "d_bias": bias, "x_use": x_on}
            em = build_emission(track, case.players, case.players_s, case.slot_team, cfg)
            decoded = decode(em, track.source, case.half, cfg)
            labels = apply_x_policy(decoded.labels, em.conf, case.half, cfg)
            pred = pm.label_codes(_LETTERS[np.asarray(labels, dtype=np.int8) + 1])
            spell = pm.spell_scores(gt, pred, runs)["mean_f1"]
            cm3 = pm.scored_confusion(pm.confusion(gt, pred))
            rows.append({"match_id": mid, "d_bias_tag": tag, "d_bias": bias,
                         "x_use": x_on, "spell_f1": float(spell),
                         "macro_f1": float(pm.macro_f1(cm3))})
            print(f"  {mid} {tag:5s} x_use={x_on}: spell {spell:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)

    base = out[(out.d_bias_tag == "prior") & (~out.x_use)]["spell_f1"]
    print(f"\nbaseline (prior, x off): mean {base.mean():.4f}")
    verdicts = {}
    for tag, x_on, name in [("fit", False, "d_bias fit"), ("prior", True, "x_use on"),
                            ("fit", True, "both")]:
        sub = out[(out.d_bias_tag == tag) & (out.x_use == x_on)]["spell_f1"]
        delta = float(sub.mean() - base.mean())
        adopt = delta > ADOPT_GAIN
        verdicts[name] = (delta, adopt)
        print(f"{name:10s}: mean {sub.mean():.4f}  delta {delta:+.4f}  "
              f"{'ADOPT' if adopt else 'keep shipped (<= 0.02 tie rule)'}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
