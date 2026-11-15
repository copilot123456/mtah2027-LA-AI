"""Fit the P9 player-only kinematic models on IDSSE (dossier §7 P9, pre-declared 2026-11-14).

    python -m tuning.kin_fit        # fit on all seven matches -> src/kin_model.json

Two logistic regressions on the features of `src/kinematics.py`:
  dead — `D` of the provider letters vs alive, with intercept, over in-match frames;
  team — `H` vs `A` among alive frames of the aligned (challenge-definition) letters,
         without intercept and without centring (team-swap antisymmetric).
Every 5th frame, standardised features, L2 1e-3, IRLS. No ball coordinate and nothing from
the provided match enters. `tuning.stage9` calls `fit_models(..., exclude=<test match>)`
for the leave-one-match-out pass. `degrade_players` builds the degraded tracks (the
pinned external-bed handicap) that stage 9 feeds to `match_features`.
"""

from __future__ import annotations  # postponed annotations

import json  # model serialisation
import sys  # sys.path bootstrapping
from pathlib import Path  # path constants

import numpy as np  # arrays and RNG

ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # make the repo importable from any cwd

# pinned pipeline config and feature builders, imported after the path bootstrap
from config import CFG  # noqa: E402
# feature names, model path, series/feature builders
from src.kinematics import (DEAD_FEATURES, MODEL_PATH, TEAM_FEATURES, base_series,  # noqa: E402
                            dead_features, team_features)
# external IDSSE matches: loading, controlled letters, match ids
from tuning.idsse import MATCH_IDS, derive_letters_controlled, load_match  # noqa: E402

STRIDE = 5  # train on every 5th frame
L2 = 1e-3  # ridge strength for both fits
DROP_SHARE = 0.07        # bed handicap: share of visible player-frames hidden (22 -> ~20.5)
DROP_MEAN_RUN = 100      # frames (4 s) — mean length of a hidden run


def degrade_players(players: np.ndarray, players_s: np.ndarray, rng: np.random.Generator,
                    share: float = DROP_SHARE, mean_run: int = DROP_MEAN_RUN
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Copies with every slot hidden in exponential runs so ~`share` of its frames vanish."""
    p, s = players.copy(), players_s.copy()  # copies; caller's arrays untouched
    n, k = s.shape
    mean_gap = mean_run * (1.0 - share) / share  # visible gap mean hitting the target share
    for j in range(k):  # each player slot hidden independently
        t = 0
        while t < n:
            t += int(rng.exponential(mean_gap)) + 1  # frames visible before the next run
            run = int(rng.exponential(mean_run)) + 1  # hidden run length
            p[t:t + run, j] = np.nan  # hide this slot for the run
            s[t:t + run, j] = np.nan
            t += run  # continue after the run
    return p, s  # dropout-degraded copies


def match_features(case, players=None, players_s=None) -> dict:
    """Feature matrices and training targets for one IDSSE match."""
    players = case.players if players is None else players  # clean tracks by default
    players_s = case.players_s if players_s is None else players_s
    base = base_series(players, players_s, case.slot_team, case.half, CFG)  # per-frame features
    aligned = derive_letters_controlled(case.letters, hold_s=3.0)  # challenge letters, 3 s hold
    return {"x_dead": dead_features(base, CFG), "x_team": team_features(base, CFG),  # features
            "in_match": case.letters != "X", "dead": case.letters == "D",  # dead target
            "alive": np.isin(aligned, ("H", "A")), "home": aligned == "H",  # team target
            "speed_med": base["speed_med"], "n_vis": base["n_vis"]}  # parity diagnostics


def fit_logistic(x: np.ndarray, y: np.ndarray, intercept: bool, l2: float = L2,
                 iters: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IRLS logistic regression on standardised columns -> (w [+ intercept last], mu, sd).

    Without an intercept the columns are scaled by their root mean square and NOT centred,
    so negating x negates the fitted logit exactly."""
    mu = x.mean(axis=0) if intercept else np.zeros(x.shape[1])  # centre only with intercept
    sd = x.std(axis=0) if intercept else np.sqrt((x ** 2).mean(axis=0))  # std, or RMS
    sd = np.where(sd > 0, sd, 1.0)  # never divide by zero
    z = (x - mu) / sd  # standardised design matrix
    if intercept:
        z = np.column_stack([z, np.ones(len(z))])  # bias column appended last
    w = np.zeros(z.shape[1])  # start from zero weights
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30.0, 30.0)))  # sigmoid, clipped logits
        grad = z.T @ (p - y) / len(y) + l2 * w  # mean gradient + L2
        hess = (z * (p * (1.0 - p))[:, None]).T @ z / len(y) + l2 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)  # Newton step
        w = w - step
        if np.abs(step).max() < 1e-10:
            break  # converged
    return w, mu, sd  # weights (bias last), scaling used


def fit_models(feats: dict[str, dict], exclude: str | None = None) -> dict:
    """The model dict `src.kinematics.apply_model` reads, fitted on every match but `exclude`."""
    used = [m for m in feats if m != exclude]  # leave-one-match-out pool
    xd, yd, xt, yt = [], [], [], []
    for m in used:
        f = feats[m]
        sel = np.flatnonzero(f["in_match"])[::STRIDE]  # every STRIDE-th in-match frame
        ok = np.isfinite(f["x_dead"][sel]).all(axis=1)  # drop NaN feature rows
        xd.append(f["x_dead"][sel][ok])
        yd.append(f["dead"][sel][ok].astype(np.float64))
        sel = np.flatnonzero(f["alive"])[::STRIDE]  # alive frames for the team model
        ok = np.isfinite(f["x_team"][sel]).all(axis=1)
        xt.append(f["x_team"][sel][ok])
        yt.append(f["home"][sel][ok].astype(np.float64))
    xd, yd, xt, yt = np.vstack(xd), np.concatenate(yd), np.vstack(xt), np.concatenate(yt)
    wd, mud, sdd = fit_logistic(xd, yd, intercept=True)  # dead model with bias
    wt, _, sdt = fit_logistic(xt, yt, intercept=False)  # team model, no bias (antisymmetric)
    share = float(yd.mean())  # dead base rate → prior logit
    return {
        # intercept split into w + b; prior_logit is the bias the decoder adds
        "dead": {"features": list(DEAD_FEATURES), "mu": mud.tolist(), "sd": sdd.tolist(),
                 "w": wd[:-1].tolist(), "b": float(wd[-1]),
                 "prior_logit": float(np.log(share / (1.0 - share)))},
        "team": {"features": list(TEAM_FEATURES), "sd": sdt.tolist(), "w": wt.tolist()},
        # provenance recorded with the shipped model
        "fit": {"source": "IDSSE (CC BY 4.0)", "matches": used, "stride": STRIDE, "l2": L2,
                "n_dead_rows": int(len(yd)), "n_team_rows": int(len(yt)),
                "dead_share": share, "home_share": float(yt.mean())},
    }


def main() -> int:
    feats = {}
    for mid in MATCH_IDS:  # all 7 external matches
        feats[mid] = match_features(load_match(mid))
        print(f"{mid}: features ready", flush=True)
    model = fit_models(feats)  # fit on every match (no exclude)
    MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")  # ship model
    print(f"wrote {MODEL_PATH.relative_to(ROOT)}: dead rows {model['fit']['n_dead_rows']}, "
          f"team rows {model['fit']['n_team_rows']}")
    for name in ("dead", "team"):
        print(f"  {name}: " + ", ".join(f"{f} {w:+.3f}" for f, w in  # coefficient readout
                                         zip(model[name]["features"], model[name]["w"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
