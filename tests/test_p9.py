"""P9 components (dossier §7 P9): coalescing min-dwell, kinematic terms and gate tier,
speed-scaled team-switch delay, the player-only kinematic features and their fit."""

import numpy as np

from config import CFG
from src.ball_fill import SOURCE_ATD, SOURCE_NONE, BallTrack
from src.decoder import (CODE_A, CODE_D, CODE_H, apply_gating, delay_team_switches,
                         enforce_min_dwell)
from src.emission import CLASS_A, CLASS_D, CLASS_H, build_emission
from src.kinematics import (DEAD_FEATURES, TEAM_FEATURES, Kinematics, apply_model, base_series,
                            dead_features, rolling_nanmean, team_features)

H, A, D = CODE_H, CODE_A, CODE_D
# The P9 keys at their OFF values (the shipped config has them ON since the adoption).
OFF = {**CFG, "dwell_coalesce": False, "kin_w_dead": 0.0, "kin_w_team": 0.0,
       "gate_kin_support": False, "switch_delay_slope": 0.0}


def _log_e(winners):
    out = np.full((len(winners), 3), -5.0)
    out[np.arange(len(winners)), winners] = 0.0
    return out


# ---------------- C: coalescing min-dwell ----------------

def _coalesce_case():
    # H 100 | D 30 | H 30 | A 200, minimum 50: the D merges into the left H; the two H
    # segments are then neighbours and together 160 frames long.
    return np.array([H] * 100 + [D] * 30 + [H] * 30 + [A] * 200, dtype=np.int8)


def test_min_dwell_without_coalescing_gives_the_short_twin_to_the_other_team():
    out = enforce_min_dwell(_coalesce_case(), 50)
    assert (out[:130] == H).all() and (out[130:] == A).all()      # the defect, on record


def test_min_dwell_with_coalescing_keeps_the_joined_segment():
    out = enforce_min_dwell(_coalesce_case(), 50, coalesce=True)
    assert (out[:160] == H).all() and (out[160:] == A).all()


def test_coalescing_is_a_noop_when_no_merge_creates_twins():
    labels = np.array([H] * 100 + [A] * 20 + [H] * 100 + [D] * 80, dtype=np.int8)
    a = enforce_min_dwell(labels, 50)
    b = enforce_min_dwell(labels, 50, coalesce=True)
    assert (a == b).all() and (b[:220] == H).all()


# ---------------- gate tier ----------------

def _dead_gap_case():
    """H evidence 0-59, ball gap 60-139 that Viterbi labelled D, H evidence 140-199."""
    labels = np.array([H] * 60 + [D] * 80 + [H] * 60, dtype=np.int8)
    source = np.full(200, SOURCE_ATD, dtype=np.int8)
    source[60:140] = SOURCE_NONE
    log_e = _log_e(np.array([H] * 60 + [D] * 80 + [H] * 60))
    kin = source == SOURCE_NONE
    return labels, source, log_e, kin


def test_gate_without_kin_support_reverts_a_dead_segment_the_ball_never_showed():
    labels, source, log_e, kin = _dead_gap_case()
    out, reverted = apply_gating(labels, source, log_e, {**OFF, "gate_mode": "move"},
                                 kin_dead=kin)
    assert reverted == 1 and (out == H).all()           # flag off: the mask is ignored


def test_gate_with_kin_support_keeps_it():
    labels, source, log_e, kin = _dead_gap_case()
    cfg = {**CFG, "gate_mode": "move", "gate_kin_support": True}
    out, reverted = apply_gating(labels, source, log_e, cfg, kin_dead=kin)
    assert reverted == 0 and (out == labels).all()


def test_kin_support_is_per_label_type():
    labels, source, log_e, kin = _dead_gap_case()
    cfg = {**CFG, "gate_mode": "move", "gate_kin_support": True}
    out, reverted = apply_gating(labels, source, log_e, cfg, kin_team=kin)   # team mask only
    assert reverted == 1 and (out == H).all()


def test_kin_team_support_lets_a_team_switch_stand_inside_a_gap():
    labels = np.array([H] * 60 + [A] * 80 + [H] * 60, dtype=np.int8)
    source = np.full(200, SOURCE_ATD, dtype=np.int8)
    source[60:140] = SOURCE_NONE
    log_e = _log_e(np.array([H] * 60 + [A] * 80 + [H] * 60))
    kin = source == SOURCE_NONE
    cfg = {**OFF, "gate_mode": "move"}
    off, _ = apply_gating(labels, source, log_e, cfg, kin_team=kin)
    on, _ = apply_gating(labels, source, log_e, {**cfg, "gate_kin_support": True}, kin_team=kin)
    assert (off == H).all() and (on == labels).all()


# ---------------- S: speed-scaled delay ----------------

def _switch_case():
    labels = np.array([H] * 100 + [A] * 100, dtype=np.int8)
    source = np.full(200, SOURCE_ATD, dtype=np.int8)
    return labels, source


def _first_a(out):
    return int(np.flatnonzero(out == A)[0])


def test_slope_zero_is_the_constant_delay():
    labels, source = _switch_case()
    speed = np.full(200, 20.0)
    a = delay_team_switches(labels, source, 16)
    b = delay_team_switches(labels, source, 16, ball_speed=speed, slope_s=0.0,
                            ref_ms=9.5, max_delay_s=1.6)
    assert (a == b).all() and _first_a(a) == 116


def test_a_fast_ball_is_delayed_more_and_a_slow_one_less():
    labels, source = _switch_case()
    kw = dict(slope_s=0.06, ref_ms=9.5, max_delay_s=1.6, fps=25)
    fast = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 19.5), **kw)
    slow = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 4.5), **kw)
    ref = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 9.5), **kw)
    assert _first_a(ref) == 116
    assert _first_a(fast) == 100 + round((0.64 + 0.6) * 25)      # 1.24 s
    assert _first_a(slow) == 100 + round((0.64 - 0.3) * 25)      # 0.34 s


def test_sloped_delay_is_clipped_and_falls_back_without_a_speed():
    labels, source = _switch_case()
    kw = dict(slope_s=0.06, ref_ms=9.5, max_delay_s=1.6, fps=25)
    huge = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 80.0), **kw)
    still = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 0.0), **kw)
    blind = delay_team_switches(labels, source, 16, ball_speed=np.full(200, np.nan), **kw)
    assert _first_a(huge) == 140                                  # 1.6 s clip
    assert _first_a(still) == 100 + round((0.64 - 0.57) * 25)
    assert _first_a(blind) == 116


# ---------------- emission terms ----------------

def _emission_case(n=60):
    ball = np.full((n, 2), 0.5)
    ball[20:40] = np.nan
    source = np.full(n, SOURCE_ATD, dtype=np.int8)
    source[20:40] = SOURCE_NONE
    age = np.zeros(n)
    age[20:40] = 0.4
    track = BallTrack(xy=ball, source=source, age_s=age, out_of_pitch=np.zeros(n, dtype=bool))
    xs = np.linspace(0.05, 0.95, 46)
    players = np.tile(np.stack([xs, np.resize(np.linspace(0.05, 0.95, 24), 46)], axis=-1),
                      (n, 1, 1))
    speeds = np.full((n, 46), 3.0)
    teams = (np.arange(46) % 2).astype(np.int8)
    rng = np.random.default_rng(3)
    kin = Kinematics(dead_llr=rng.normal(size=n), team_logit=rng.normal(size=n),
                     dead_ok=np.ones(n, dtype=bool), team_ok=np.ones(n, dtype=bool))
    return track, players, speeds, teams, kin


def test_zero_weights_ignore_the_kinematics():
    track, players, speeds, teams, kin = _emission_case()
    a = build_emission(track, players, speeds, teams, OFF)
    b = build_emission(track, players, speeds, teams, OFF, kin=kin)
    assert (a.log_e == b.log_e).all() and b.kin_dead is None and b.kin_team is None


def test_kinematic_terms_touch_ball_less_frames_only():
    track, players, speeds, teams, kin = _emission_case()
    a = build_emission(track, players, speeds, teams, OFF)
    b = build_emission(track, players, speeds, teams,
                       {**OFF, "kin_w_dead": 0.3, "kin_w_team": 0.3}, kin=kin)
    seen = track.source == SOURCE_ATD
    assert (a.log_e[seen] == b.log_e[seen]).all()
    assert (a.log_e[~seen] != b.log_e[~seen]).any()
    assert (b.kin_dead == ~seen).all() and (b.kin_team == ~seen).all()
    contrast = b.log_e[~seen, CLASS_H] - b.log_e[~seen, CLASS_A]
    assert (np.sign(contrast) == np.sign(kin.team_logit[~seen])).all()


def test_dead_term_raises_d_where_the_players_say_dead():
    track, players, speeds, teams, kin = _emission_case()
    strong = Kinematics(dead_llr=np.full(60, 4.0), team_logit=np.zeros(60),
                        dead_ok=kin.dead_ok, team_ok=kin.team_ok)
    a = build_emission(track, players, speeds, teams, OFF)
    b = build_emission(track, players, speeds, teams, {**OFF, "kin_w_dead": 0.3}, kin=strong)
    gap = track.source == SOURCE_NONE
    assert (b.log_e[gap, CLASS_D] > a.log_e[gap, CLASS_D]).all()
    assert (b.log_e[gap, CLASS_H] < a.log_e[gap, CLASS_H]).all()


def test_team_term_mirrors_exactly_under_a_team_swap():
    track, players, speeds, teams, kin = _emission_case()
    cfg = {**OFF, "kin_w_team": 0.3, "kin_w_dead": 0.1}
    swapped = Kinematics(dead_llr=kin.dead_llr, team_logit=-kin.team_logit,
                         dead_ok=kin.dead_ok, team_ok=kin.team_ok)
    a = build_emission(track, players, speeds, teams, cfg, kin=kin)
    b = build_emission(track, players, speeds, 1 - teams, cfg, kin=swapped)
    gap = track.source == SOURCE_NONE
    assert (a.log_e[gap, CLASS_H] == b.log_e[gap, CLASS_A]).all()
    assert (a.log_e[gap, CLASS_A] == b.log_e[gap, CLASS_H]).all()
    assert (a.log_e[gap, CLASS_D] == b.log_e[gap, CLASS_D]).all()


def test_frames_without_finite_features_carry_no_kinematic_evidence():
    track, players, speeds, teams, kin = _emission_case()
    blind = Kinematics(dead_llr=kin.dead_llr, team_logit=kin.team_logit,
                       dead_ok=np.zeros(60, dtype=bool), team_ok=np.zeros(60, dtype=bool))
    a = build_emission(track, players, speeds, teams, OFF)
    b = build_emission(track, players, speeds, teams,
                       {**OFF, "kin_w_dead": 0.3, "kin_w_team": 0.3}, kin=blind)
    assert (a.log_e == b.log_e).all()


# ---------------- kinematic features and model ----------------

def _random_match(n=400, k=24, seed=5):
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.1, 0.9, size=(k, 2))
    walk = np.cumsum(rng.normal(scale=0.002, size=(n, k, 2)), axis=0)
    players = np.clip(start[None] + walk, 0.0, 1.0)
    players[50:80, 3] = np.nan                       # a slot vanishes for a while
    speeds = np.abs(rng.normal(2.0, 1.5, size=(n, k)))
    speeds[50:80, 3] = np.nan
    teams = (np.arange(k) % 2).astype(np.int8)
    half = np.ones(n, dtype=np.int8)
    return players, speeds, teams, half


def test_rolling_nanmean_ignores_nan_and_is_centred():
    x = np.array([1.0, np.nan, 3.0, 5.0, np.nan])
    out = rolling_nanmean(x, 1)
    np.testing.assert_allclose(out, [1.0, 2.0, 4.0, 4.0, 5.0])
    assert np.isnan(rolling_nanmean(np.array([np.nan, np.nan]), 1)).all()


def test_feature_matrices_have_the_pinned_columns():
    players, speeds, teams, half = _random_match()
    base = base_series(players, speeds, teams, half)
    assert dead_features(base).shape == (400, len(DEAD_FEATURES)) == (400, 7)
    assert team_features(base).shape == (400, len(TEAM_FEATURES)) == (400, 4)


def test_team_swap_negates_team_features_exactly_and_leaves_dead_features_alone():
    players, speeds, teams, half = _random_match()
    a = base_series(players, speeds, teams, half)
    b = base_series(players, speeds, 1 - teams, half)
    ta, tb = team_features(a), team_features(b)
    ok = np.isfinite(ta).all(axis=1)
    assert ok.sum() > 300 and (np.isfinite(tb).all(axis=1) == ok).all()
    assert (ta[ok] == -tb[ok]).all()
    da, db = dead_features(a), dead_features(b)
    ok = np.isfinite(da).all(axis=1)
    assert (da[ok] == db[ok]).all()


def _toy_model():
    return {"dead": {"features": list(DEAD_FEATURES), "mu": [1.0] * 7, "sd": [2.0] * 7,
                     "w": [0.5, -0.25, 1.0, -1.0, 0.3, -0.3, 0.1], "b": -0.2,
                     "prior_logit": -0.2},
            "team": {"features": list(TEAM_FEATURES), "sd": [1.5, 20.0, 3.0, 4.0],
                     "w": [0.8, 1.2, 0.3, 1.9]}}


def test_model_logit_is_exactly_antisymmetric_and_blind_rows_are_zero():
    players, speeds, teams, half = _random_match()
    a = base_series(players, speeds, teams, half)
    b = base_series(players, speeds, 1 - teams, half)
    ka = apply_model(dead_features(a), team_features(a), _toy_model())
    kb = apply_model(dead_features(b), team_features(b), _toy_model())
    assert (ka.team_logit == -kb.team_logit).all()
    assert (ka.dead_llr == kb.dead_llr).all()
    assert (ka.team_logit[~ka.team_ok] == 0).all() and (ka.dead_llr[~ka.dead_ok] == 0).all()
    assert not ka.dead_ok[:75].any()              # the 3 s deceleration needs history


def test_model_with_another_feature_list_is_refused():
    model = _toy_model()
    model["team"]["features"] = ["vx3", "terr", "dsx"]
    try:
        apply_model(np.zeros((2, 7)), np.zeros((2, 4)), model)
    except ValueError:
        return
    raise AssertionError("a model fitted on other features must be refused")


def test_fit_recovers_a_planted_direction_and_stays_antisymmetric():
    from tuning.kin_fit import degrade_players, fit_logistic

    rng = np.random.default_rng(11)
    x = rng.normal(size=(4000, 3)) * np.array([1.0, 5.0, 0.2])
    y = (x[:, 0] - 0.2 * x[:, 1] + rng.logistic(size=4000) * 0.5 > 0).astype(float)
    w, mu, sd = fit_logistic(x, y, intercept=False)
    assert w[0] > 0 > w[1] and (mu == 0).all()
    w2, _, sd2 = fit_logistic(-x, 1.0 - y, intercept=False)
    np.testing.assert_allclose(w, w2, rtol=1e-9)
    np.testing.assert_allclose(sd, sd2, rtol=1e-12)

    players = np.zeros((20000, 6, 2))
    hidden_p, hidden_s = degrade_players(players, np.ones((20000, 6)), rng)
    share = np.isnan(hidden_s).mean()
    assert 0.04 < share < 0.10 and (np.isnan(hidden_p[..., 0]) == np.isnan(hidden_s)).all()


# ---------------- harness and defaults ----------------

def test_stage9_grid_is_the_pinned_factorial():
    from tuning.stage9 import DELAYS, P_PARAMS, REFERENCE, SLOPES, WEIGHTS, p9_cells

    cells = p9_cells()
    assert len(cells) == 2 * (1 + 15 * 2) * 12 == 744
    assert cells[0] == REFERENCE
    assert len({tuple(c[p] for p in P_PARAMS) for c in cells}) == 744
    assert WEIGHTS == [0.0, 0.03, 0.1, 0.3]
    assert DELAYS == [0.64, 0.80, 0.96, 1.12] and SLOPES == [0.0, 0.03, 0.06]
    assert not any(c["gate_kin_support"] and c["kin_w_dead"] == 0 and c["kin_w_team"] == 0
                   for c in cells)


def test_shipped_model_is_the_pinned_shape():
    from src.kinematics import load_model

    model = load_model()
    assert tuple(model["dead"]["features"]) == DEAD_FEATURES
    assert tuple(model["team"]["features"]) == TEAM_FEATURES
    assert set(model["team"]) == {"features", "sd", "w"}          # no intercept, no centring
    for part, keys in (("dead", ("mu", "sd", "w")), ("team", ("sd", "w"))):
        for k in keys:
            assert np.isfinite(model[part][k]).all() and len(model[part][k]) == len(
                model[part]["features"])
    assert len(model["fit"]["matches"]) == 7 and model["fit"]["source"].startswith("IDSSE")


def _synthetic_tables(effect):
    import pandas as pd

    from tuning.idsse import MATCH_IDS
    from tuning.stage5 import KINDS
    from tuning.stage9 import P_PARAMS, p9_cells

    rows = []
    for c in p9_cells():
        for i, m in enumerate(MATCH_IDS):
            v = 0.40 + effect(c) + 0.001 * i
            rows.append({**c, "match_id": m, "spell_f1_orig": v, "spell_f1_aligned": v})
    table = pd.DataFrame(rows).set_index(P_PARAMS + ["match_id"]).sort_index()
    return {k: table for k in KINDS}


def _planted(c):
    k = {0.0: 0.0, 0.03: 0.004, 0.1: 0.020, 0.3: 0.021}[c["kin_w_dead"]]
    t = {0.0: 0.0, 0.03: 0.0, 0.1: -0.01, 0.3: -0.03}[c["kin_w_team"]]
    return (0.01 * c["dwell_coalesce"] + k * (1.0 if c["gate_kin_support"] else 0.2) + t
            + (0.006 if c["switch_delay_s"] == 0.96 else 0.0)
            + 0.002 * (c["switch_delay_slope"] > 0))


def test_gate9_rule_keeps_steps_in_order_with_the_pinned_tie_break():
    from tuning import gate9

    rule = gate9.apply_rule(_synthetic_tables(_planted))
    assert rule["C"]["keep"] and rule["K"]["keep"] and rule["S"]["keep"]
    assert not rule["T"]["keep"]
    bundle = rule["bundle"]
    assert bundle["kin_w_dead"] == 0.1 and bundle["gate_kin_support"] is True   # 0.3 is a tie
    assert bundle["kin_w_team"] == 0.0
    assert bundle["switch_delay_s"] == 0.96 and bundle["switch_delay_slope"] == 0.0
    assert rule["adopt"] and abs(rule["bundle_primary"] - 0.036) < 1e-9
    alone = gate9.step_alone(rule, "K")
    assert alone["kin_w_dead"] == 0.1 and alone["dwell_coalesce"] is False


def test_gate9_hold_switches_a_component_off_and_re_reads():
    from tuning import gate9

    tables = _synthetic_tables(_planted)
    held = gate9.apply_rule(tables, frozenset({"K"}))
    assert held["K"]["held"] and held["bundle"]["kin_w_dead"] == 0.0
    assert held["bundle"]["gate_kin_support"] is False
    assert not held["adopt"]                                      # 0.016 is below the bar
    no_slope = gate9.candidates("S", dict(gate9.REFERENCE), frozenset({"slope"}))
    assert all(c["switch_delay_slope"] == 0.0 for c in no_slope) and len(no_slope) == 3
    assert len(gate9.candidates("S", dict(gate9.REFERENCE), frozenset())) == 11


def test_committed_stage9_read():
    """The recorded read: C, K, T kept, S dropped, bundle passes the bar on all 8 tables —
    and the shipped config is exactly that bundle (hold released by the operator)."""
    from pathlib import Path

    import pytest

    from tuning import gate9

    if not (Path(gate9.RES) / "stage9_uniform.csv").exists():
        pytest.skip("stage-9 matrices not present")
    tables = gate9.load_tables()
    assert gate9.harness_pin(tables) <= gate9.PIN_TOL
    rule = gate9.apply_rule(tables)
    assert rule["C"]["keep"] and rule["K"]["keep"] and rule["T"]["keep"]
    assert not rule["S"]["keep"]
    bundle = rule["bundle"]
    assert (bundle["dwell_coalesce"], bundle["kin_w_dead"], bundle["kin_w_team"],
            bundle["gate_kin_support"]) == (True, 0.3, 0.03, True)
    assert rule["adopt"] and rule["bundle_primary"] > 0.02 and rule["bundle_wins"] == 7
    assert all(v[0] > 0.08 and v[1] == 7 for v in rule["bundle_deltas"].values())
    for key in gate9.P_PARAMS:
        assert CFG[key] == bundle[key], key
