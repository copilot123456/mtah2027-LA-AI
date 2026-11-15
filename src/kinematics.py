"""Player-only collective kinematics — P9 (dossier §7 P9, pre-declared 2026-11-14).

Two per-frame signals computed from the player tracks alone (no ball coordinate enters):

  * `dead_llr`   — log-likelihood ratio dead : alive from how the players move (speeds,
                   share of slow / fast players, collective drift);
  * `team_logit` — log-odds that the home team (slot team 0) has the ball, from four
                   home-minus-away features. The model has no intercept and no centring,
                   so swapping the two teams negates the logit exactly.

The decoder uses them only on frames without a ball observation, and only when
`kin_w_dead` / `kin_w_team` are positive (both 0 by default — the default pipeline never
calls this module). Coefficients are fitted on the external IDSSE matches by
`tuning/kin_fit.py` and stored in `src/kin_model.json`; nothing here is fitted on the
provided match.

Pure functions: inputs are never modified.
"""

from __future__ import annotations  # postponed type hints

import json          # read model coefficients
import warnings      # silence expected NaN warnings
from dataclasses import dataclass  # frozen result container
from pathlib import Path  # locate the model file

import numpy as np  # array maths

from config import CFG  # default config values

MODEL_PATH = Path(__file__).with_name("kin_model.json")  # coefficients sit next to this file

SLOW_MS = 1.5          # a slot below this speed counts as slow (walking)
FAST_MS = 4.0          # a slot above this speed counts as fast (running)
VEL_CLIP_MS = 12.0     # per-slot velocity above this is a slot swap, not a player
DEAD_FEATURES = ("sm1", "sm5", "slow5", "fast5", "vabs1", "vabs5", "dec3")  # dead feature order
TEAM_FEATURES = ("vx3", "terr", "dsx", "dsy")  # team feature order


def rolling_nanmean(x: np.ndarray, half_window: int) -> np.ndarray:
    """Centred mean over +/- `half_window` frames ignoring NaN; NaN where nothing is finite."""
    x = np.asarray(x, dtype=np.float64)   # work in float64
    if half_window <= 0:
        return x.copy()                   # no smoothing; still copy input
    finite = np.isfinite(x)               # mask of usable samples
    vals = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))  # prefix sums, 0-leading
    cnts = np.concatenate(([0], np.cumsum(finite)))  # prefix counts, 0-leading
    i = np.arange(len(x))                 # frame indices
    lo, hi = np.maximum(0, i - half_window), np.minimum(len(x), i + half_window + 1)  # window bounds
    n = cnts[hi] - cnts[lo]               # valid samples inside window
    return np.where(n > 0, (vals[hi] - vals[lo]) / np.maximum(n, 1), np.nan)  # mean, NaN if empty


def base_series(players: np.ndarray, players_s: np.ndarray, slot_team: np.ndarray,
                half: np.ndarray, cfg: dict = CFG) -> dict:
    """Per-frame collective quantities in metres and m/s; NaN where undefined."""
    fps = int(cfg["fps"])                 # frames per second
    lag = fps // 2                        # half-second lag for differences
    n = len(players)                      # frame count
    x = players[..., 0] * cfg["pitch_length_m"]  # x in metres
    y = players[..., 1] * cfg["pitch_width_m"]   # y in metres
    vis = np.isfinite(x) & np.isfinite(y)  # slot visible this frame
    out: dict = {}                        # per-frame series collected here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # NaN medians warn; expected
        s = np.where(vis, players_s, np.nan)  # invisible speeds become NaN
        ok = np.isfinite(s)               # usable speed samples
        cnt = np.maximum(ok.sum(axis=1), 1)  # per-frame sample count, min 1
        any_ok = ok.any(axis=1)           # frames with any speed sample
        out["n_vis"] = vis.sum(axis=1).astype(np.float64)  # visible slot count
        out["speed_med"] = np.nanmedian(s, axis=1)  # median player speed
        out["slow_share"] = np.where(any_ok, (ok & (s < SLOW_MS)).sum(axis=1) / cnt, np.nan)  # walking fraction
        out["fast_share"] = np.where(any_ok, (ok & (s > FAST_MS)).sum(axis=1) / cnt, np.nan)  # running fraction
        vx = np.full(x.shape, np.nan)     # per-slot x velocity, NaN by default
        if n > 2 * lag:
            vx[lag:-lag] = (x[2 * lag:] - x[:-2 * lag]) * (fps / (2.0 * lag))  # centred difference
        vx[np.abs(vx) > VEL_CLIP_MS] = np.nan  # huge jumps are ID swaps
        out["vx_all"] = np.nanmean(vx, axis=1)  # mean drift across slots
        xv = np.where(vis, x, np.nan)     # invisible x as NaN
        out["cx_all"] = np.nanmean(xv, axis=1)  # mean x of visible players
        for t in (0, 1):                  # one block per team
            m = vis & (slot_team[None, :] == t)  # visible slots of team t
            out[f"cx{t}"] = np.nanmean(np.where(m, x, np.nan), axis=1)  # team mean x
            out[f"sx{t}"] = np.nanstd(np.where(m, x, np.nan), axis=1)   # team x spread
            out[f"sy{t}"] = np.nanstd(np.where(m, y, np.nan), axis=1)   # team y spread
        # +1 where team 0 sits on the low-x side (median over the half), -1 otherwise
        sign = np.zeros(n)                # side flag starts unassigned
        for h in (1, 2):                  # side decided per half
            sel = half == h            # frames of this half
            if sel.any():
                d = np.nanmedian(out["cx0"][sel] - out["cx1"][sel])  # team0 minus team1 mean x
                sign[sel] = 1.0 if d < 0 else -1.0  # low-x side gets +1
        out["s0"] = sign                  # store side flag
    return out                        # the per-frame series dict


def dead_features(base: dict, cfg: dict = CFG) -> np.ndarray:
    """(N, 7) features in `DEAD_FEATURES` order — unchanged when the two teams are swapped."""
    fps = int(cfg["fps"])                 # frames per second
    w1, w5, lag = fps // 2, int(2.5 * fps), 3 * fps  # windows in frames
    sm1 = rolling_nanmean(base["speed_med"], w1)  # short-window mean speed
    dec = np.full(len(sm1), np.nan)        # deceleration starts NaN
    if len(sm1) > lag:
        dec[lag:] = sm1[lag:] - sm1[:-lag]  # speed change over last 3 s
    cols = [sm1,                           # sm1: 0.5 s mean speed
            rolling_nanmean(base["speed_med"], w5),  # sm5: 2.5 s mean speed
            rolling_nanmean(base["slow_share"], w5),  # slow5: long-run slow share
            rolling_nanmean(base["fast_share"], w5),  # fast5: long-run fast share
            np.abs(rolling_nanmean(base["vx_all"], w1)),  # vabs1: short-run |drift|
            np.abs(rolling_nanmean(base["vx_all"], w5)),  # vabs5: long-run |drift|
            dec]                          # dec3: 3 s speed drop
    return np.column_stack(cols)           # stack into (N, 7)


def team_features(base: dict, cfg: dict = CFG) -> np.ndarray:
    """(N, 4) home-minus-away features in `TEAM_FEATURES` order — every column changes sign
    exactly when the two teams are swapped."""
    fps = int(cfg["fps"])                 # frames per second
    w1, w3 = fps // 2, int(1.5 * fps)      # smoothing windows in frames
    s0 = base["s0"]                       # side flag flips sign with teams
    cols = [s0 * rolling_nanmean(base["vx_all"], w3),  # vx3: drift oriented by side
            s0 * (rolling_nanmean(base["cx_all"], w1) - cfg["pitch_length_m"] / 2.0),  # terr: centre offset, oriented by side
            rolling_nanmean(base["sx0"] - base["sx1"], w1),  # dsx: home minus away x spread
            rolling_nanmean(base["sy0"] - base["sy1"], w1)]  # dsy: home minus away y spread
    return np.column_stack(cols)           # stack into (N, 4)


@dataclass(frozen=True)
class Kinematics:
    dead_llr: np.ndarray    # (N,) log-likelihood ratio dead : alive; 0 where `dead_ok` is False
    team_logit: np.ndarray  # (N,) log-odds home : away; 0 where `team_ok` is False
    dead_ok: np.ndarray     # (N,) bool — the dead features were all finite
    team_ok: np.ndarray     # (N,) bool — the team features were all finite


def load_model(path: Path = MODEL_PATH) -> dict:
    """The shipped coefficients (fitted on IDSSE by `tuning/kin_fit.py`)."""
    with open(path, encoding="utf-8") as fh:  # read the JSON file
        return json.load(fh)              # plain dict of coefficients


def apply_model(x_dead: np.ndarray, x_team: np.ndarray, model: dict) -> Kinematics:
    """Turn feature matrices into the two signals with the given coefficients."""
    d, t = model["dead"], model["team"]   # dead and team coefficient blocks
    if tuple(d["features"]) != DEAD_FEATURES or tuple(t["features"]) != TEAM_FEATURES:  # model must match feature lists
        raise ValueError("kinematic model was fitted on a different feature list")  # abort on feature mismatch
    dead_ok = np.isfinite(x_dead).all(axis=1)  # rows with all dead features finite
    team_ok = np.isfinite(x_team).all(axis=1)  # rows with all team features finite
    zd = (np.where(dead_ok[:, None], x_dead, 0.0) - np.asarray(d["mu"])) / np.asarray(d["sd"])  # standardise; bad rows zeroed
    llr = zd @ np.asarray(d["w"]) + (float(d["b"]) - float(d["prior_logit"]))  # weight sum, prior removed
    zt = np.where(team_ok[:, None], x_team, 0.0) / np.asarray(t["sd"])  # standardise, no centring
    logit = zt @ np.asarray(t["w"])       # weight sum, no intercept
    return Kinematics(dead_llr=np.where(dead_ok, llr, 0.0),  # 0 where features missing
                      team_logit=np.where(team_ok, logit, 0.0),  # 0 where features missing
                      dead_ok=dead_ok, team_ok=team_ok)


def build_kinematics(players: np.ndarray, players_s: np.ndarray, slot_team: np.ndarray,
                     half: np.ndarray, cfg: dict = CFG, model: dict | None = None) -> Kinematics:
    """Both signals for one match from its player tracks."""
    base = base_series(players, players_s, slot_team, half, cfg)  # collective series
    return apply_model(dead_features(base, cfg), team_features(base, cfg),  # features -> signals
                       load_model() if model is None else model)
