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
  * p_d = sigmoid(d_bias + sum w_i f_i) with five dead-ball features
    (out_of_pitch, stationary, near boundary, long missing run, low activity).
  * Duel (both teams equally close) => equal H/A columns by construction.
  * A constant added to all three columns of one frame cannot change any Viterbi
    path, so only the *contrast* matters; the formulation keeps that contrast
    proportional to conf for the team part and feature-driven for D.

Pure function: inputs are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CFG
from src.ball_fill import SOURCE_ATD, SOURCE_KINEMATIC, SOURCE_NONE, BallTrack

CLASS_H, CLASS_A, CLASS_D = 0, 1, 2
CLASS_NAMES = ("H", "A", "D")

EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


@dataclass(frozen=True)
class Emission:
    log_e: np.ndarray   # (N, 3) float64, columns H A D
    conf: np.ndarray    # (N,) float64 in [0, 1]
    p_d: np.ndarray     # (N,) float64 in (0, 1)
    ev: np.ndarray      # (N, 2) raw team kernel masses (H, A)

    @property
    def margin(self) -> np.ndarray:
        """log-odds H vs A after tempering (0 exactly where conf=0)."""
        return self.log_e[:, CLASS_H] - self.log_e[:, CLASS_A]


def team_confidence(track: BallTrack, cfg: dict = CFG) -> np.ndarray:
    conf = np.zeros(len(track.source), dtype=np.float64)
    conf[track.source == SOURCE_ATD] = cfg["c0_atd"]
    kin = track.source == SOURCE_KINEMATIC
    conf[kin] = cfg["c0_kinematic"] * np.exp(-track.age_s[kin] / cfg["tau_kinematic_s"])
    # SOURCE_DETECTOR (stretch tier): confidence TBD when that tier is implemented
    return np.clip(conf, 0.0, 1.0)


def _team_evidence(track: BallTrack, players: np.ndarray, slot_team: np.ndarray,
                   cfg: dict) -> np.ndarray:
    """(N, 2): kernel mass of each team's top-k visible slots around the ball (metres)."""
    n = len(track.source)
    ev = np.zeros((n, 2), dtype=np.float64)
    has_ball = ~np.isnan(track.xy).any(axis=1)
    if not has_ball.any():
        return ev

    length, width = cfg["pitch_length_m"], cfg["pitch_width_m"]
    ball = track.xy[has_ball]                       # (m, 2) normalized
    slots = players[has_ball]                       # (m, 46, 2) normalized, NaN where invisible

    dx = (ball[:, 0, None] - slots[..., 0]) * length
    dy = (ball[:, 1, None] - slots[..., 1]) * width
    dist = np.hypot(dx, dy)                         # metres, NaN where invisible

    err = np.where(track.source[has_ball] == SOURCE_KINEMATIC, cfg["err_kinematic_m"], 0.0)
    sigma = np.sqrt(cfg["sigma_m"] ** 2 + err ** 2)[:, None]
    kernel = np.exp(-((dist / sigma) ** 2))
    kernel = np.where(np.isnan(dist), 0.0, kernel)

    k = int(cfg["topk"])
    for t in (0, 1):
        masked = np.where(slot_team[None, :] == t, kernel, 0.0)
        top = -np.sort(-masked, axis=1)[:, :k]      # largest k per row (zeros pad short teams)
        ev[has_ball, t] = top.sum(axis=1)
    return ev


def _ball_speed_ms(track: BallTrack, cfg: dict) -> np.ndarray:
    """Backward-difference ball speed in m/s; NaN where either endpoint is missing."""
    n = len(track.source)
    speed = np.full(n, np.nan)
    pos = ~np.isnan(track.xy).any(axis=1)
    delta = np.diff(track.xy, axis=0)                     # delta[i]: i -> i+1
    steps = np.hypot(delta[:, 0] * cfg["pitch_length_m"],
                     delta[:, 1] * cfg["pitch_width_m"]) * cfg["fps"]
    valid = pos[:-1] & pos[1:]                            # validity of each step
    speed[1:] = np.where(valid, steps, np.nan)            # speed[j] = step (j-1 -> j)
    return speed


def _dead_features(track: BallTrack, players_s: np.ndarray, cfg: dict) -> dict:
    n = len(track.source)
    has_pos = ~np.isnan(track.xy).any(axis=1)

    speed = _ball_speed_ms(track, cfg)
    stationary = (speed < cfg["ball_stationary_thr_ms"]) & np.isfinite(speed)

    x, y = track.xy[:, 0], track.xy[:, 1]
    near_boundary = has_pos & (
        (np.minimum(x, 1 - x) < cfg["pitch_margin"])
        | (np.minimum(y, 1 - y) < cfg["pitch_margin"])
    )

    no_evidence = (track.source == SOURCE_NONE).astype(np.float64)
    long_missing = no_evidence * np.minimum(1.0, track.age_s / cfg["d_missing_run_s"])

    counts = np.sum(~np.isnan(players_s), axis=1)
    sums = np.nansum(players_s, axis=1)
    activity = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    low_activity = np.where(np.isfinite(activity),
                            activity < cfg["d_activity_thr_ms"], False)

    return {
        "out_of_pitch": track.out_of_pitch.astype(np.float64),
        "stationary": stationary.astype(np.float64),
        "near_boundary": near_boundary.astype(np.float64),
        "long_missing": long_missing,
        "low_activity": low_activity.astype(np.float64),
    }


def build_emission(track: BallTrack, players: np.ndarray, players_s: np.ndarray,
                   slot_team: np.ndarray, cfg: dict = CFG) -> Emission:
    n = len(track.source)
    if not (len(players) == len(players_s) == n):
        raise ValueError("track and player arrays must have the same length")

    conf = team_confidence(track, cfg)
    ev = _team_evidence(track, players, slot_team, cfg)

    total = ev.sum(axis=1)
    p_team = np.full((n, 2), 0.5, dtype=np.float64)
    live = total > EPS
    p_team[live] = ev[live] / total[live, None]

    feats = _dead_features(track, players_s, cfg)
    z = np.full(n, cfg["d_bias"], dtype=np.float64)
    z += cfg["d_w_out_of_pitch"] * feats["out_of_pitch"]
    z += cfg["d_w_stationary"] * feats["stationary"]
    z += cfg["d_w_boundary"] * feats["near_boundary"]
    z += cfg["d_w_long_missing"] * feats["long_missing"]
    z += cfg["d_w_low_activity"] * feats["low_activity"]
    p_d = _sigmoid(z)

    log_e = np.empty((n, 3), dtype=np.float64)
    alive = np.log(np.clip(1.0 - p_d, EPS, 1.0))
    log_e[:, CLASS_H] = conf * np.log(np.clip(p_team[:, 0], EPS, 1.0)) + alive
    log_e[:, CLASS_A] = conf * np.log(np.clip(p_team[:, 1], EPS, 1.0)) + alive
    log_e[:, CLASS_D] = np.log(np.clip(p_d, EPS, 1.0))

    return Emission(log_e=log_e, conf=conf, p_d=p_d, ev=ev)
