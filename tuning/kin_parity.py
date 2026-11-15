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

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CFG  # noqa: E402

RES = ROOT / "tuning" / "results"
HOLDS = {"Q1": ("K", "T"), "Q2": ("K",), "Q3": ("T",), "Q4": ("T",), "Q5": ("K", "T"),
         "Q6": ("slope",)}
NAMES = {"Q1": "speed scale", "Q2": "dead share", "Q3": "team balance",
         "Q4": "team agreement", "Q5": "visible players", "Q6": "ball speed scale"}


def run_checks(res: Path = RES) -> dict[str, bool]:
    """{check: passed}. Values stay inside this function."""
    from src import data as D
    from src.ball_fill import build_ball_track
    from src.decoder import CODE_A, CODE_H
    from src.emission import _ball_speed_ms
    from src.kinematics import apply_model, base_series, dead_features, load_model, team_features
    from src.pipeline import run

    ref = pd.read_csv(res / "stage9_diag.csv")
    ref = ref[ref["q4_team_agreement"].notna()]
    if len(ref) != 7:
        raise ValueError("stage9_diag.csv must carry the IDSSE side for seven matches")

    arrays = D.load_arrays()
    graded = arrays.graded
    model = load_model()
    base = base_series(arrays.players, arrays.players_s, arrays.slot_team, arrays.half, CFG)
    kin = apply_model(dead_features(base, CFG), team_features(base, CFG), model)

    ball_only = {**CFG, "kin_w_dead": 0.0, "kin_w_team": 0.0, "gate_kin_support": False}
    labels, _ = run(ball_only)
    alive = np.isin(labels, (CODE_H, CODE_A)) & graded

    out: dict[str, bool] = {}
    speed = float(np.nanmedian(base["speed_med"][graded]))
    out["Q1"] = 0.5 * ref["speed_med_p50"].min() <= speed <= 2.0 * ref["speed_med_p50"].max()

    d_ok = graded & kin.dead_ok
    dead_share = float((kin.dead_llr[d_ok] + float(model["dead"]["prior_logit"]) > 0).mean())
    out["Q2"] = 0.20 <= dead_share <= 0.65

    t_ok = graded & kin.team_ok
    out["Q3"] = 0.30 <= float((kin.team_logit[t_ok] > 0).mean()) <= 0.70

    seen = alive & t_ok & np.isfinite(arrays.ball).all(axis=1)
    agree = float(((kin.team_logit[seen] > 0) == (labels[seen] == CODE_H)).mean())
    out["Q4"] = agree >= ref["q4_team_agreement"].min() - 0.05

    out["Q5"] = float(np.nanmean(base["n_vis"][graded])) >= 18.0

    track = build_ball_track(arrays.ball, arrays.half, CFG)
    ball_speed = _ball_speed_ms(track, CFG)
    sel = alive & np.isfinite(ball_speed)
    v = float(np.median(ball_speed[sel]))
    out["Q6"] = (0.5 * ref["q6_ball_speed_p50"].min() <= v
                 <= 2.0 * ref["q6_ball_speed_p50"].max())
    return {k: bool(v) for k, v in out.items()}


def held_components(verdicts: dict[str, bool]) -> list[str]:
    held = {c for q, ok in verdicts.items() if not ok for c in HOLDS[q]}
    return sorted(held)


def main() -> int:
    verdicts = run_checks()
    for q in sorted(verdicts):
        print(f"{q} {NAMES[q]}: {'PASS' if verdicts[q] else 'FAIL'}")
    held = held_components(verdicts)
    print("held components: " + (", ".join(held) if held else "none"))
    pd.DataFrame([{"check": q, "name": NAMES[q],
                   "verdict": "PASS" if verdicts[q] else "FAIL",
                   "holds": "+".join(HOLDS[q])} for q in sorted(verdicts)]
                 ).to_csv(RES / "kin_parity.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
