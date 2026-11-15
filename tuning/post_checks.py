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

from __future__ import annotations   # postponed annotation evaluation

import sys                # sys.path setup below
from pathlib import Path  # repo-root path objects

import numpy as np   # RNG and array math
import pandas as pd  # result tables and CSV output

ROOT = Path(__file__).resolve().parents[1]   # repo root
for p in (str(ROOT), str(ROOT / "evaluation")):   # both dirs must be importable
    if p not in sys.path:   # skip if already on path
        sys.path.insert(0, p)   # add this directory to path

# project imports (placed after the path setup, hence noqa)
import possession_metrics as pm  # noqa: E402
from config import CFG  # noqa: E402
from tuning.idsse import MATCH_IDS, load_match  # noqa: E402
from tuning.simulate import empirical_run_lengths, mask_ball  # noqa: E402
from tuning.sweep import PARAMS, RESULTS, _LETTERS, _score, flat_select  # noqa: E402

ADOPT_GAIN = 0.02   # mean gain needed to adopt; smaller is a tie (keep shipped)
OUT = ROOT / "tuning" / "results" / "post_checks.csv"           # default (stage-1) output
OUT_FINAL = ROOT / "tuning" / "results" / "post_checks_final.csv"   # --final output
STAGE1_FILL_LONG_GAP_S = 10.0   # the prior value in force when the stage-1 checks ran


def main() -> int:
    import argparse   # CLI flags below

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])   # parser headed by docstring line
    parser.add_argument("--final", action="store_true",   # flag for shipped-cell mode
                        help="evaluate at the shipped config.py cell with the shipped fill "
                             "setting (default: stage-1 chosen cell, kinematic scenario)")
    parser.add_argument("--out", default=None, help="output CSV (default per mode)")   # optional output path override
    args = parser.parse_args()   # read command-line arguments

    if args.final:
        chosen = {p: float(CFG[p]) for p in PARAMS}   # shipped cell is the chosen one
        base = dict(CFG)                              # shipped config as the base
        out_path = Path(args.out) if args.out else OUT_FINAL   # default output for final mode
    else:
        df = pd.read_csv(RESULTS)                     # committed sweep results
        chosen, _ = flat_select(df)                   # stage-1 chosen cell
        base = {**CFG, "fill_long_gap_s": STAGE1_FILL_LONG_GAP_S}   # prior fill setting
        out_path = Path(args.out) if args.out else OUT   # default output for stage-1 mode
    print("chosen cell:", chosen, flush=True)   # echo the cell under test
    print(f"fill_long_gap_s = {base['fill_long_gap_s']}", flush=True)   # echo the fill setting used

    from src import data as D   # array loading

    arrays = D.load_arrays()   # load frame arrays
    run_lengths = empirical_run_lengths(   # missing-ball runs to replay when masking
        np.isnan(arrays.ball).any(axis=1) & arrays.graded)

    # --- external d_bias candidate from provider labels (pooled) ---
    dead_total = alive_total = 0   # pooled counters across matches
    for mid in MATCH_IDS:
        case = load_match(mid)   # this match's data
        letters = case.letters   # provider label letters
        in_match = letters != "X"   # frames that count as in-match
        dead_total += int(((letters == "D") & in_match).sum())   # provider dead frames
        alive_total += int(in_match.sum())                       # all in-match frames
    d_share = dead_total / alive_total   # pooled dead share
    d_bias_fit = float(np.log(d_share / (1 - d_share)))   # logit form the decoder wants
    print(f"IDSSE pooled D share {d_share:.3f} -> d_bias external_fit {d_bias_fit:+.3f}", flush=True)   # report the fitted d_bias

    # decoder stages, imported after path setup
    from src.ball_fill import build_ball_track
    from src.decoder import decode
    from src.emission import build_emission
    from src.x_policy import apply_x_policy

    variants = [   # the four (d_bias tag, d_bias value, x_use) cells to score
        ("prior", CFG["d_bias"], False),   # shipped
        ("prior", CFG["d_bias"], True),    # shipped bias, x on
        ("fit", d_bias_fit, False),        # fitted bias, x off
        ("fit", d_bias_fit, True),         # both changes
    ]
    rows = []   # one scored row per (match, variant)
    for mid in MATCH_IDS:   # all 7 IDSSE matches
        case = load_match(mid)   # this match's data
        rng = np.random.default_rng(20261107 + sum(ord(c) for c in mid))   # sweep's per-match seed
        masked, _ = mask_ball(case.ball, run_lengths, rng)   # frozen masking
        track = build_ball_track(masked, case.half, base)   # fill setting from `base`
        gt = pm.label_codes(case.letters)   # ground-truth codes
        for tag, bias, x_on in variants:
            cfg = {**base, **chosen, "d_bias": bias, "x_use": x_on}   # cell under test
            em = build_emission(track, case.players, case.players_s, case.slot_team, cfg)   # emission for this variant
            decoded = decode(em, track.source, case.half, cfg)   # decode emissions into labels
            labels = apply_x_policy(decoded.labels, em.conf, case.half, cfg)   # apply X-policy to labels
            pred = pm.label_codes(_LETTERS[np.asarray(labels, dtype=np.int8) + 1])   # labels -> letters -> codes
            scores = _score(gt, pred)          # official semantics (X forward-filled)
            rows.append({"match_id": mid, "d_bias_tag": tag, "d_bias": bias,   # record this variant's scores
                         "x_use": x_on, **scores})
            print(f"  {mid} {tag:5s} x_use={x_on}: spell {scores['spell_f1']:.4f}  "   # progress line
                  f"macro {scores['macro_f1']:.4f}", flush=True)

    out = pd.DataFrame(rows)   # table of all rows
    out.to_csv(out_path, index=False)   # write the results CSV

    base = out[(out.d_bias_tag == "prior") & (~out.x_use)]["spell_f1"]   # shipped baseline scores
    print(f"\nbaseline (prior, x off): mean {base.mean():.4f}")   # print the baseline mean
    base_macro = out[(out.d_bias_tag == "prior") & (~out.x_use)]["macro_f1"]   # baseline macro scores
    verdicts = {}   # collect adoption verdicts
    for tag, x_on, name in [("fit", False, "d_bias fit"), ("prior", True, "x_use on"),
                            ("fit", True, "both")]:   # each candidate vs the baseline
        sel = out[(out.d_bias_tag == tag) & (out.x_use == x_on)]   # that candidate's rows
        delta = float(sel["spell_f1"].mean() - base.mean())            # mean spellF1 gain
        delta_macro = float(sel["macro_f1"].mean() - base_macro.mean())   # macro gain, report only
        adopt = delta > ADOPT_GAIN   # must beat the tie rule strictly
        verdicts[name] = (delta, adopt)   # store this candidate's verdict
        print(f"{name:10s}: mean {sel['spell_f1'].mean():.4f}  delta {delta:+.4f}  "   # report gain against baseline
              f"(macroF1 delta {delta_macro:+.4f})  "
              f"{'ADOPT' if adopt else 'keep shipped (<= 0.02 tie rule)'}")
    print(f"wrote {out_path}")   # confirm CSV written
    return 0   # report clean completion


if __name__ == "__main__":   # run only as a script
    raise SystemExit(main())   # exit with main's code
