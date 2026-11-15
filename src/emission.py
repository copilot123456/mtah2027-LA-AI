"""Emission model — P2.

For every frame, a log-emission over the three states (H, A, D):

    e_H = conf * log(p_team_H) + log(1 - p_d)
    e_A = conf * log(p_team_A) + log(1 - p_d)
    e_D = log(p_d)

Design notes (tests enforce the behaviours):
  * Team evidence  ev_t = sum of the top-k exp(-(d/sigma_eff)^2) over that team's
    VISIBLE slots, distances in metres (normalized coords x pitch dims).
    sigma_eff widens for kinematic fills (err_kinematic_m prior).
  * conf: atd = c0_atd; kinematic = c0_kinematic * exp(-age_s / tau_kinematic_s);
    none = 0. At conf = 0 the H/A columns are exactly equal (a frame with no
    trustworthy ball evidence says nothing about which team has it) while the
    dead-ball term still speaks (tier-3 rule).
  * p_d = sigmoid(d_bias + sum w_i f_i) with six dead-ball features
    (out_of_pitch, stationary, near boundary, long missing run, low activity,
    set-piece — E7, OFF by default).
  * Duel (both teams equally close) => equal H/A columns by construction.
  * A constant added to all three columns of one frame cannot change any Viterbi
    path, so only the *contrast* matters; the formulation keeps that contrast
    proportional to conf for the team part and feature-driven for D.
  * Kinematic terms (P9, dossier §7; both weights 0 by default, so the default output is
    unchanged): on frames WITHOUT a ball observation the dead logit gains
    kin_w_dead * (player-only dead log-likelihood ratio) and the H / A columns gain
    kin_w_team * log sigmoid(+/- player-only team logit). Ball-visible frames are untouched.

Pure function: inputs are never modified.
"""

from __future__ import annotations  # postponed type hints

from dataclasses import dataclass  # frozen result container

import numpy as np  # array maths

from config import CFG  # default config values
from src.ball_fill import SOURCE_ATD, SOURCE_KINEMATIC, SOURCE_NONE, BallTrack  # provenance codes and track type

CLASS_H, CLASS_A, CLASS_D = 0, 1, 2  # column indices for H, A, D
CLASS_NAMES = ("H", "A", "D")        # labels in column order

EPS = 1e-12  # floor so logs never see zero


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))  # logistic, clipped for stability


@dataclass(frozen=True)
class Emission:
    log_e: np.ndarray   # (N, 3) float64, columns H A D
    conf: np.ndarray    # (N,) float64 in [0, 1]
    p_d: np.ndarray     # (N,) float64 in (0, 1)
    ev: np.ndarray      # (N, 2) raw team kernel masses (H, A)
    activity: np.ndarray  # (N,) per-frame mean player speed (m/s), NaN where unknown (E10)
    ball_speed: np.ndarray | None = None  # (N,) backward-difference ball speed, m/s (P9 S)
    kin_dead: np.ndarray | None = None    # (N,) bool: kinematic dead term applied (P9 K)
    kin_team: np.ndarray | None = None    # (N,) bool: kinematic team term applied (P9 T)

    @property
    def margin(self) -> np.ndarray:
        """log-odds H vs A after tempering (0 exactly where conf=0)."""
        return self.log_e[:, CLASS_H] - self.log_e[:, CLASS_A]  # H minus A columns


def team_confidence(track: BallTrack, cfg: dict = CFG) -> np.ndarray:
    conf = np.zeros(len(track.source), dtype=np.float64)  # no evidence: zero confidence
    conf[track.source == SOURCE_ATD] = cfg["c0_atd"]  # observed tracking: full constant
    kin = track.source == SOURCE_KINEMATIC  # kinematic-fill frames
    conf[kin] = cfg["c0_kinematic"] * np.exp(-track.age_s[kin] / cfg["tau_kinematic_s"])  # decays with age
    # SOURCE_DETECTOR (stretch tier): confidence TBD when that tier is implemented
    return np.clip(conf, 0.0, 1.0)        # keep within [0, 1]


def _team_evidence(track: BallTrack, players: np.ndarray, slot_team: np.ndarray,
                   cfg: dict) -> np.ndarray:
    """(N, 2): kernel mass of each team's top-k visible slots around the ball (metres)."""
    n = len(track.source)                 # frame count
    ev = np.zeros((n, 2), dtype=np.float64)  # evidence starts at zero
    has_ball = ~np.isnan(track.xy).any(axis=1)  # frames holding a ball position
    if not has_ball.any():
        return ev                         # no ball anywhere: all zeros

    length, width = cfg["pitch_length_m"], cfg["pitch_width_m"]  # metres per normalized unit
    ball = track.xy[has_ball]                       # (m, 2) normalized
    slots = players[has_ball]                       # (m, 46, 2) normalized, NaN where invisible

    dx = (ball[:, 0, None] - slots[..., 0]) * length  # x gap in metres
    dy = (ball[:, 1, None] - slots[..., 1]) * width   # y gap in metres
    dist = np.hypot(dx, dy)                         # metres, NaN where invisible

    err = np.where(track.source[has_ball] == SOURCE_KINEMATIC, cfg["err_kinematic_m"], 0.0)  # extra error for fills
    sigma = np.sqrt(cfg["sigma_m"] ** 2 + err ** 2)[:, None]  # combine base and fill error
    kernel = np.exp(-((dist / sigma) ** 2))  # Gaussian over distance in metres
    kernel = np.where(np.isnan(dist), 0.0, kernel)  # invisible slots contribute nothing

    k = int(cfg["topk"])                   # slots kept per team
    for t in (0, 1):                       # one evidence score per team
        masked = np.where(slot_team[None, :] == t, kernel, 0.0)  # zero out other team
        top = -np.sort(-masked, axis=1)[:, :k]      # largest k per row (zeros pad short teams)
        ev[has_ball, t] = top.sum(axis=1)  # store team's top-k mass
    return ev                        # hand back team evidence


def _ball_speed_ms(track: BallTrack, cfg: dict) -> np.ndarray:
    """Backward-difference ball speed in m/s; NaN where either endpoint is missing."""
    n = len(track.source)                 # frame count
    speed = np.full(n, np.nan)            # first frame stays NaN
    pos = ~np.isnan(track.xy).any(axis=1)  # frames holding a position
    delta = np.diff(track.xy, axis=0)                     # delta[i]: i -> i+1
    steps = np.hypot(delta[:, 0] * cfg["pitch_length_m"],  # step length in metres
                     delta[:, 1] * cfg["pitch_width_m"]) * cfg["fps"]  # times fps
    valid = pos[:-1] & pos[1:]                            # validity of each step
    speed[1:] = np.where(valid, steps, np.nan)            # speed[j] = step (j-1 -> j)
    return speed                      # per-frame speed array


def _set_piece_feature(starts: np.ndarray | None, n: int, cfg: dict) -> np.ndarray:
    """f_sp (E7): 1.0 within [start − sp_pre_s, start + sp_post_s] of any tag start."""
    out = np.zeros(n, dtype=np.float64)  # default: not near a set piece
    if starts is None or len(starts) == 0:
        return out                       # no tags: all zeros
    pre = int(round(cfg["sp_pre_s"] * cfg["fps"]))   # frames before each start
    post = int(round(cfg["sp_post_s"] * cfg["fps"]))  # frames after each start
    for s in np.asarray(starts, dtype=np.int64):  # each tagged set-piece start
        a, b = max(int(s) - pre, 0), min(int(s) + post + 1, n)  # window, clipped
        if a < b:
            out[a:b] = 1.0              # mark window as set piece
    return out                          # binary feature per frame


def _dead_features(track: BallTrack, players_s: np.ndarray, cfg: dict,
                   setpiece_starts: np.ndarray | None = None) -> dict:
    n = len(track.source)                 # frame count
    has_pos = ~np.isnan(track.xy).any(axis=1)  # frames with any ball position

    speed = _ball_speed_ms(track, cfg)    # ball speed series
    stationary = (speed < cfg["ball_stationary_thr_ms"]) & np.isfinite(speed)  # slow and defined

    x, y = track.xy[:, 0], track.xy[:, 1]  # ball coordinates
    near_boundary = has_pos & (            # within margin of any pitch edge
        (np.minimum(x, 1 - x) < cfg["pitch_margin"])
        | (np.minimum(y, 1 - y) < cfg["pitch_margin"])
    )

    no_evidence = (track.source == SOURCE_NONE).astype(np.float64)  # 1 where tier is none
    long_missing = no_evidence * np.minimum(1.0, track.age_s / cfg["d_missing_run_s"])  # ramps to 1 over time

    counts = np.sum(~np.isnan(players_s), axis=1)  # players with a speed
    sums = np.nansum(players_s, axis=1)  # sum of those speeds
    activity = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)  # mean speed, NaN if none
    low_activity = np.where(np.isfinite(activity),
                            activity < cfg["d_activity_thr_ms"], False)  # below threshold

    return {
        "out_of_pitch": track.out_of_pitch.astype(np.float64),  # flag from ball_fill
        "stationary": stationary.astype(np.float64),  # ball not moving
        "near_boundary": near_boundary.astype(np.float64),  # ball near edge
        "long_missing": long_missing,    # prolonged ball absence
        "low_activity": low_activity.astype(np.float64),  # players barely moving
        "set_piece": _set_piece_feature(setpiece_starts, n, cfg),  # near a restart
        "activity": activity,          # raw per-frame mean speed (E10 no-play rule)
        "ball_speed": speed,           # m/s, NaN where an endpoint is missing (P9 S)
    }


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)        # numerically stable log(sigmoid(x))


def build_emission(track: BallTrack, players: np.ndarray, players_s: np.ndarray,
                   slot_team: np.ndarray, cfg: dict = CFG,
                   setpiece_starts: np.ndarray | None = None,
                   kin=None, *, kin_dead_all_frames: bool = False) -> Emission:
    """`setpiece_starts` feeds the E7 dead-logit term; with `d_w_set_piece = 0`
    (the default) the output is identical to passing nothing.

    `kin` (a `src.kinematics.Kinematics`) feeds the P9 terms on ball-less frames; with
    `kin_w_dead = kin_w_team = 0` (the default) or `kin=None` the output is identical.
    `kin_dead_all_frames` is the record-only variant of the stage-9 harness (the dead term
    on every frame); the pipeline never sets it."""
    n = len(track.source)                 # frame count
    if not (len(players) == len(players_s) == n):
        raise ValueError("track and player arrays must have the same length")  # inputs must align

    conf = team_confidence(track, cfg)    # per-frame team confidence
    ev = _team_evidence(track, players, slot_team, cfg)  # raw kernel masses per team

    total = ev.sum(axis=1)                # combined kernel mass
    p_team = np.full((n, 2), 0.5, dtype=np.float64)  # default 50/50 without evidence
    live = total > EPS                    # frames with any team evidence
    p_team[live] = ev[live] / total[live, None]  # normalize masses to shares

    feats = _dead_features(track, players_s, cfg, setpiece_starts)  # dead-ball features
    z = np.full(n, cfg["d_bias"], dtype=np.float64)  # dead logit starts at bias
    # add each dead feature times its configured weight
    z += cfg["d_w_out_of_pitch"] * feats["out_of_pitch"]
    z += cfg["d_w_stationary"] * feats["stationary"]  # ball not moving
    z += cfg["d_w_boundary"] * feats["near_boundary"]  # ball close to edge
    z += cfg["d_w_long_missing"] * feats["long_missing"]  # ball missing a long time
    z += cfg["d_w_low_activity"] * feats["low_activity"]  # players barely moving
    z += cfg["d_w_set_piece"] * feats["set_piece"]  # near a set piece

    w_dead = float(cfg.get("kin_w_dead", 0.0)) if kin is not None else 0.0  # P9 dead weight
    w_team = float(cfg.get("kin_w_team", 0.0)) if kin is not None else 0.0  # P9 team weight
    no_ball = track.source == SOURCE_NONE  # frames with no ball evidence
    kin_dead = kin_team = None            # flags stay None unless used
    if w_dead > 0:
        kin_dead = kin.dead_ok & (no_ball | kin_dead_all_frames)  # valid ball-less rows
        z[kin_dead] += w_dead * kin.dead_llr[kin_dead]  # add weighted dead LLR
    p_d = _sigmoid(z)                     # logit becomes dead probability

    log_e = np.empty((n, 3), dtype=np.float64)  # per-frame scores, H A D
    alive = np.log(np.clip(1.0 - p_d, EPS, 1.0))  # shared alive term
    log_e[:, CLASS_H] = conf * np.log(np.clip(p_team[:, 0], EPS, 1.0)) + alive  # home column
    log_e[:, CLASS_A] = conf * np.log(np.clip(p_team[:, 1], EPS, 1.0)) + alive  # away column
    log_e[:, CLASS_D] = np.log(np.clip(p_d, EPS, 1.0))  # dead column
    if w_team > 0:
        kin_team = kin.team_ok & no_ball  # ball-less rows with valid features
        log_e[kin_team, CLASS_H] += w_team * _log_sigmoid(kin.team_logit[kin_team])  # home gains
        log_e[kin_team, CLASS_A] += w_team * _log_sigmoid(-kin.team_logit[kin_team])  # away mirrored

    return Emission(log_e=log_e, conf=conf, p_d=p_d, ev=ev, activity=feats["activity"],  # bundle frame scores
                    ball_speed=feats["ball_speed"], kin_dead=kin_dead, kin_team=kin_team)
