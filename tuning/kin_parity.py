"""Target parity checks Q1–Q6 of dossier §7 P9 — PASS / FAIL only, never a figure.

    python -m tuning.kin_parity

The kinematic models are fitted on IDSSE. Before a kinematic component is adopted, these
checks ask whether the provided match looks, to the model, like the matches it was fitted
on. They read the provided tracking and the pipeline's own ball-only output (kinematic
terms forced OFF) — no ground truth, no video — and print one verdict per check. The IDSSE
side of each threshold comes from `tuning/results/stage9_diag.csv`.

  Q1 speed scale       median per-frame median player speed within [0.5 x min, 2 x max] of IDSSE
  Q2 dead share        share of graded frames with a positive dead logit within [0.20, 0.65]
  Q3 team balance      share of graded frames whose team logit favours home within [0.30, 0.70]
  Q4 team agreement    on ball-visible graded frames the ball-only output labels H / A, the
                       team logit agrees at least as often as the weakest IDSSE match - 0.05
  Q5 visible players   mean visible slots per graded frame >= 18
  Q6 ball speed scale  median ball speed on graded H / A frames within [0.5 x min, 2 x max]

Q1 and Q5 hold K and T; Q2 holds K; Q3 and Q4 hold T; Q6 holds the slope of S.
"""

from __future__ import annotations  # postponed annotations

import sys  # sys.path bootstrapping
from pathlib import Path  # path constants

import numpy as np  # medians and means
import pandas as pd  # diag read + verdict CSV

ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # make the repo importable from any cwd

# pipeline config for feature and ball parameters, imported after the path bootstrap
from config import CFG  # noqa: E402

RES = ROOT / "tuning" / "results"  # diag read here; verdict CSV written here
# components each failed check holds for the operator
HOLDS = {"Q1": ("K", "T"), "Q2": ("K",), "Q3": ("T",), "Q4": ("T",), "Q5": ("K", "T"),
         "Q6": ("slope",)}
NAMES = {"Q1": "speed scale", "Q2": "dead share", "Q3": "team balance",  # check → output name
         "Q4": "team agreement", "Q5": "visible players", "Q6": "ball speed scale"}


def run_checks(res: Path = RES) -> dict[str, bool]:
    """{check: passed}. Values stay inside this function."""
    from src import data as D  # provided match arrays
    from src.ball_fill import build_ball_track  # ball track for Q6
    from src.decoder import CODE_A, CODE_H  # team label codes
    from src.emission import _ball_speed_ms  # ball speed for Q6
    from src.kinematics import apply_model, base_series, dead_features, load_model, team_features
    from src.pipeline import run  # ball-only decode for Q4

    ref = pd.read_csv(res / "stage9_diag.csv")  # IDSSE side of every threshold
    ref = ref[ref["q4_team_agreement"].notna()]  # keep matches with a Q4 value
    if len(ref) != 7:
        raise ValueError("stage9_diag.csv must carry the IDSSE side for seven matches")

    arrays = D.load_arrays()  # provided match data
    graded = arrays.graded  # graded-frame mask
    model = load_model()  # shipped kinematic models
    base = base_series(arrays.players, arrays.players_s, arrays.slot_team, arrays.half, CFG)  # clean features
    kin = apply_model(dead_features(base, CFG), team_features(base, CFG), model)  # logits

    ball_only = {**CFG, "kin_w_dead": 0.0, "kin_w_team": 0.0, "gate_kin_support": False}
    labels, _ = run(ball_only)  # kinematic terms forced OFF
    alive = np.isin(labels, (CODE_H, CODE_A)) & graded  # graded H/A frames

    out: dict[str, bool] = {}
    speed = float(np.nanmedian(base["speed_med"][graded]))  # Q1: median player speed
    out["Q1"] = 0.5 * ref["speed_med_p50"].min() <= speed <= 2.0 * ref["speed_med_p50"].max()

    d_ok = graded & kin.dead_ok  # frames the dead model judges
    dead_share = float((kin.dead_llr[d_ok] + float(model["dead"]["prior_logit"]) > 0).mean())
    out["Q2"] = 0.20 <= dead_share <= 0.65  # Q2: positive-dead-logit share

    t_ok = graded & kin.team_ok  # frames the team model judges
    out["Q3"] = 0.30 <= float((kin.team_logit[t_ok] > 0).mean()) <= 0.70  # Q3: home share

    seen = alive & t_ok & np.isfinite(arrays.ball).all(axis=1)  # ball-visible judged frames
    agree = float(((kin.team_logit[seen] > 0) == (labels[seen] == CODE_H)).mean())
    out["Q4"] = agree >= ref["q4_team_agreement"].min() - 0.05  # Q4: weakest IDSSE − slack

    out["Q5"] = float(np.nanmean(base["n_vis"][graded])) >= 18.0  # Q5: visible slots

    track = build_ball_track(arrays.ball, arrays.half, CFG)  # ball-only track
    ball_speed = _ball_speed_ms(track, CFG)
    sel = alive & np.isfinite(ball_speed)  # frames with a finite ball speed
    v = float(np.median(ball_speed[sel]))  # Q6: median ball speed
    out["Q6"] = (0.5 * ref["q6_ball_speed_p50"].min() <= v
                 <= 2.0 * ref["q6_ball_speed_p50"].max())  # Q6: speed scale
    return {k: bool(v) for k, v in out.items()}  # plain bools


def held_components(verdicts: dict[str, bool]) -> list[str]:
    held = {c for q, ok in verdicts.items() if not ok for c in HOLDS[q]}  # failed checks
    return sorted(held)


def main() -> int:
    verdicts = run_checks()
    for q in sorted(verdicts):
        print(f"{q} {NAMES[q]}: {'PASS' if verdicts[q] else 'FAIL'}")  # one line per check
    held = held_components(verdicts)
    print("held components: " + (", ".join(held) if held else "none"))  # for the operator
    pd.DataFrame([{"check": q, "name": NAMES[q],  # verdict table
                   "verdict": "PASS" if verdicts[q] else "FAIL",
                   "holds": "+".join(HOLDS[q])} for q in sorted(verdicts)]
                 ).to_csv(RES / "kin_parity.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
