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

H, A, D = CODE_H, CODE_A, CODE_D                  # short label aliases
# The P9 keys at their OFF values (the shipped config has them ON since the adoption).
OFF = {**CFG, "dwell_coalesce": False, "kin_w_dead": 0.0, "kin_w_team": 0.0,
       "gate_kin_support": False, "switch_delay_slope": 0.0}


def _log_e(winners):
    """(N, 3) emissions whose per-frame argmax is `winners`."""
    out = np.full((len(winners), 3), -5.0)        # fill low everywhere
    out[np.arange(len(winners)), winners] = 0.0   # chosen label scores top
    return out                                    # fake emissions


# ---------------- C: coalescing min-dwell ----------------

def _coalesce_case():
    # H 100 | D 30 | H 30 | A 200, minimum 50: the D merges into the left H; the two H
    # segments are then neighbours and together 160 frames long.
    return np.array([H] * 100 + [D] * 30 + [H] * 30 + [A] * 200, dtype=np.int8)


def test_min_dwell_without_coalescing_gives_the_short_twin_to_the_other_team():
    """Without coalescing, the merged twin flips to the other team."""
    out = enforce_min_dwell(_coalesce_case(), 50) # min-dwell 50, no coalescing
    assert (out[:130] == H).all() and (out[130:] == A).all()      # the defect, on record


def test_min_dwell_with_coalescing_keeps_the_joined_segment():
    """Coalescing keeps the joined segment on its own team."""
    out = enforce_min_dwell(_coalesce_case(), 50, coalesce=True) # min-dwell 50 with coalescing
    assert (out[:160] == H).all() and (out[160:] == A).all() # joined segment stays H


def test_coalescing_is_a_noop_when_no_merge_creates_twins():
    """Coalescing changes nothing when no merge creates twins."""
    labels = np.array([H] * 100 + [A] * 20 + [H] * 100 + [D] * 80, dtype=np.int8) # no merge makes twins
    a = enforce_min_dwell(labels, 50)             # plain min-dwell run
    b = enforce_min_dwell(labels, 50, coalesce=True) # coalescing run
    assert (a == b).all() and (b[:220] == H).all() # same result either way


# ---------------- gate tier ----------------

def _dead_gap_case():
    """H evidence 0-59, ball gap 60-139 that Viterbi labelled D, H evidence 140-199."""
    labels = np.array([H] * 60 + [D] * 80 + [H] * 60, dtype=np.int8) # H, dead gap, H
    source = np.full(200, SOURCE_ATD, dtype=np.int8) # ATD everywhere first
    source[60:140] = SOURCE_NONE                  # gap in the middle
    log_e = _log_e(np.array([H] * 60 + [D] * 80 + [H] * 60)) # emissions follow labels
    kin = source == SOURCE_NONE                   # kinematic mask for gap
    return labels, source, log_e, kin             # hand back the case


def test_gate_without_kin_support_reverts_a_dead_segment_the_ball_never_showed():
    """Kinematic gate tier off: the dead segment reverts as before."""
    labels, source, log_e, kin = _dead_gap_case() # reuse dead-gap case
    out, reverted = apply_gating(labels, source, log_e, {**OFF, "gate_mode": "move"}, # kin gate tier off
                                 kin_dead=kin)
    assert reverted == 1 and (out == H).all()           # flag off: the mask is ignored


def test_gate_with_kin_support_keeps_it():
    """Kinematic dead support keeps the dead segment in place."""
    labels, source, log_e, kin = _dead_gap_case() # reuse dead-gap case
    cfg = {**CFG, "gate_mode": "move", "gate_kin_support": True} # kin gate tier on
    out, reverted = apply_gating(labels, source, log_e, cfg, kin_dead=kin) # run gate with dead mask
    assert reverted == 0 and (out == labels).all() # dead segment kept


def test_kin_support_is_per_label_type():
    """Support is per label type: a team mask can't back a dead segment."""
    labels, source, log_e, kin = _dead_gap_case() # reuse dead-gap case
    cfg = {**CFG, "gate_mode": "move", "gate_kin_support": True} # kin gate tier on
    out, reverted = apply_gating(labels, source, log_e, cfg, kin_team=kin)   # team mask only
    assert reverted == 1 and (out == H).all()     # wrong mask, still reverts


def test_kin_team_support_lets_a_team_switch_stand_inside_a_gap():
    """Team kinematic support lets a team switch stand inside a gap."""
    labels = np.array([H] * 60 + [A] * 80 + [H] * 60, dtype=np.int8) # H, team gap, H
    source = np.full(200, SOURCE_ATD, dtype=np.int8) # ATD everywhere first
    source[60:140] = SOURCE_NONE                  # gap in the middle
    log_e = _log_e(np.array([H] * 60 + [A] * 80 + [H] * 60)) # emissions follow labels
    kin = source == SOURCE_NONE                   # kinematic mask for gap
    cfg = {**OFF, "gate_mode": "move"}            # kin gate tier off
    off, _ = apply_gating(labels, source, log_e, cfg, kin_team=kin) # run with tier off
    on, _ = apply_gating(labels, source, log_e, {**cfg, "gate_kin_support": True}, kin_team=kin) # run with tier on
    assert (off == H).all() and (on == labels).all() # on keeps, off reverts


# ---------------- S: speed-scaled delay ----------------

def _switch_case():
    """H->A switch with ball evidence everywhere."""
    labels = np.array([H] * 100 + [A] * 100, dtype=np.int8) # H then A labels
    source = np.full(200, SOURCE_ATD, dtype=np.int8) # ball visible throughout
    return labels, source                         # hand back the case


def _first_a(out):
    """Frame index of the first A label."""
    return int(np.flatnonzero(out == A)[0])       # first A frame


def test_slope_zero_is_the_constant_delay():
    """Zero slope reproduces the constant delay exactly."""
    labels, source = _switch_case()               # reuse switch case
    speed = np.full(200, 20.0)                    # speed unused at slope zero
    a = delay_team_switches(labels, source, 16)   # constant-delay run
    b = delay_team_switches(labels, source, 16, ball_speed=speed, slope_s=0.0, # sloped call, slope zero
                            ref_ms=9.5, max_delay_s=1.6)
    assert (a == b).all() and _first_a(a) == 116  # both delay 16 frames


def test_a_fast_ball_is_delayed_more_and_a_slow_one_less():
    """Delay grows with ball speed and shrinks when it is slow."""
    labels, source = _switch_case()               # reuse switch case
    kw = dict(slope_s=0.06, ref_ms=9.5, max_delay_s=1.6, fps=25) # shared sloped-delay settings
    fast = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 19.5), **kw) # fast ball run
    slow = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 4.5), **kw) # slow ball run
    ref = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 9.5), **kw) # reference-speed run
    assert _first_a(ref) == 116                   # reference delay unchanged
    assert _first_a(fast) == 100 + round((0.64 + 0.6) * 25)      # 1.24 s
    assert _first_a(slow) == 100 + round((0.64 - 0.3) * 25)      # 0.34 s


def test_sloped_delay_is_clipped_and_falls_back_without_a_speed():
    """Sloped delay clips at the cap and falls back when speed is missing."""
    labels, source = _switch_case()               # reuse switch case
    kw = dict(slope_s=0.06, ref_ms=9.5, max_delay_s=1.6, fps=25) # shared sloped-delay settings
    huge = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 80.0), **kw) # very fast ball
    still = delay_team_switches(labels, source, 16, ball_speed=np.full(200, 0.0), **kw) # stationary ball
    blind = delay_team_switches(labels, source, 16, ball_speed=np.full(200, np.nan), **kw) # missing speed value
    assert _first_a(huge) == 140                                  # 1.6 s clip
    # slowest ball gets the least delay
    assert _first_a(still) == 100 + round((0.64 - 0.57) * 25)
    # missing speed: constant delay
    assert _first_a(blind) == 116


# ---------------- emission terms ----------------

def _emission_case(n=60):
    """Synthetic match: ball gap, random kinematics, alternating teams."""
    ball = np.full((n, 2), 0.5)                   # ball at centre
    ball[20:40] = np.nan                          # ball lost mid-match
    source = np.full(n, SOURCE_ATD, dtype=np.int8) # ATD everywhere first
    source[20:40] = SOURCE_NONE                   # gap in the middle
    age = np.zeros(n)                             # fresh ball age
    age[20:40] = 0.4                              # aged inside gap
    track = BallTrack(xy=ball, source=source, age_s=age, out_of_pitch=np.zeros(n, dtype=bool)) # assemble ball track
    xs = np.linspace(0.05, 0.95, 46)              # x positions across pitch
    players = np.tile(np.stack([xs, np.resize(np.linspace(0.05, 0.95, 24), 46)], axis=-1), # players tiled on grid
                      (n, 1, 1))
    speeds = np.full((n, 46), 3.0)                # constant player speeds
    teams = (np.arange(46) % 2).astype(np.int8)   # alternating team colours
    rng = np.random.default_rng(3)                # fixed seed
    kin = Kinematics(dead_llr=rng.normal(size=n), team_logit=rng.normal(size=n), # random kinematic scores
                     dead_ok=np.ones(n, dtype=bool), team_ok=np.ones(n, dtype=bool))
    return track, players, speeds, teams, kin     # hand back the case


def test_zero_weights_ignore_the_kinematics():
    """Kinematic weights at zero leave emissions untouched."""
    track, players, speeds, teams, kin = _emission_case() # reuse emission case
    a = build_emission(track, players, speeds, teams, OFF) # weights off, no kin
    b = build_emission(track, players, speeds, teams, OFF, kin=kin) # weights off, with kin
    assert (a.log_e == b.log_e).all() and b.kin_dead is None and b.kin_team is None # identical, no masks kept


def test_kinematic_terms_touch_ball_less_frames_only():
    """Kinematic terms only touch ball-less frames."""
    track, players, speeds, teams, kin = _emission_case() # reuse emission case
    a = build_emission(track, players, speeds, teams, OFF)  # kinematic terms off
    b = build_emission(track, players, speeds, teams, # weights turned on
                       {**OFF, "kin_w_dead": 0.3, "kin_w_team": 0.3}, kin=kin)
    seen = track.source == SOURCE_ATD             # frames with ball
    assert (a.log_e[seen] == b.log_e[seen]).all() # seen frames unchanged
    assert (a.log_e[~seen] != b.log_e[~seen]).any() # unseen frames changed
    # masks mark exactly the ball-less frames
    assert (b.kin_dead == ~seen).all() and (b.kin_team == ~seen).all()
    # team term sign follows the team logit
    contrast = b.log_e[~seen, CLASS_H] - b.log_e[~seen, CLASS_A]
    assert (np.sign(contrast) == np.sign(kin.team_logit[~seen])).all() # sign matches team logit


def test_dead_term_raises_d_where_the_players_say_dead():
    """Dead evidence raises D and lowers H in the gap."""
    track, players, speeds, teams, kin = _emission_case() # reuse emission case
    strong = Kinematics(dead_llr=np.full(60, 4.0), team_logit=np.zeros(60), # strong dead evidence
                        dead_ok=kin.dead_ok, team_ok=kin.team_ok)
    a = build_emission(track, players, speeds, teams, OFF) # weights off
    b = build_emission(track, players, speeds, teams, {**OFF, "kin_w_dead": 0.3}, kin=strong) # dead weight on
    gap = track.source == SOURCE_NONE             # gap frames mask
    assert (b.log_e[gap, CLASS_D] > a.log_e[gap, CLASS_D]).all() # D raised in gap
    assert (b.log_e[gap, CLASS_H] < a.log_e[gap, CLASS_H]).all() # H lowered in gap


def test_team_term_mirrors_exactly_under_a_team_swap():
    """Swapping teams mirrors H/A exactly and leaves D alone."""
    track, players, speeds, teams, kin = _emission_case() # reuse emission case
    cfg = {**OFF, "kin_w_team": 0.3, "kin_w_dead": 0.1} # small kin weights
    swapped = Kinematics(dead_llr=kin.dead_llr, team_logit=-kin.team_logit, # team logit negated
                         dead_ok=kin.dead_ok, team_ok=kin.team_ok)
    a = build_emission(track, players, speeds, teams, cfg, kin=kin) # original teams
    b = build_emission(track, players, speeds, 1 - teams, cfg, kin=swapped) # swapped teams
    gap = track.source == SOURCE_NONE             # gap frames mask
    assert (a.log_e[gap, CLASS_H] == b.log_e[gap, CLASS_A]).all() # H mirrors to A
    assert (a.log_e[gap, CLASS_A] == b.log_e[gap, CLASS_H]).all() # A mirrors to H
    assert (a.log_e[gap, CLASS_D] == b.log_e[gap, CLASS_D]).all() # D stays put


def test_frames_without_finite_features_carry_no_kinematic_evidence():
    """Frames flagged not-ok carry no kinematic evidence."""
    track, players, speeds, teams, kin = _emission_case() # reuse emission case
    blind = Kinematics(dead_llr=kin.dead_llr, team_logit=kin.team_logit, # all features flagged bad
                       dead_ok=np.zeros(60, dtype=bool), team_ok=np.zeros(60, dtype=bool))
    a = build_emission(track, players, speeds, teams, OFF) # weights off
    b = build_emission(track, players, speeds, teams, # weights turned on
                       {**OFF, "kin_w_dead": 0.3, "kin_w_team": 0.3}, kin=blind)
    assert (a.log_e == b.log_e).all()             # identical emissions


# ---------------- kinematic features and model ----------------

def _random_match(n=400, k=24, seed=5):
    """Synthetic players and speeds with one vanishing slot."""
    rng = np.random.default_rng(seed)             # fixed seed
    start = rng.uniform(0.1, 0.9, size=(k, 2))    # random start spots
    walk = np.cumsum(rng.normal(scale=0.002, size=(n, k, 2)), axis=0) # small random walks
    players = np.clip(start[None] + walk, 0.0, 1.0) # stay on the pitch
    players[50:80, 3] = np.nan                       # a slot vanishes for a while
    speeds = np.abs(rng.normal(2.0, 1.5, size=(n, k))) # positive speeds
    speeds[50:80, 3] = np.nan                     # same slot drops out
    teams = (np.arange(k) % 2).astype(np.int8)    # alternating teams
    half = np.ones(n, dtype=np.int8)              # all in half one
    return players, speeds, teams, half           # hand back the case


def test_rolling_nanmean_ignores_nan_and_is_centred():
    """Rolling mean skips NaN over a centred window."""
    x = np.array([1.0, np.nan, 3.0, 5.0, np.nan]) # series with gaps
    out = rolling_nanmean(x, 1)                   # window radius one
    np.testing.assert_allclose(out, [1.0, 2.0, 4.0, 4.0, 5.0]) # NaNs skipped, window centred
    assert np.isnan(rolling_nanmean(np.array([np.nan, np.nan]), 1)).all() # all-NaN stays NaN


def test_feature_matrices_have_the_pinned_columns():
    """Feature matrices have the pinned column counts."""
    players, speeds, teams, half = _random_match() # reuse random match
    base = base_series(players, speeds, teams, half) # build base series
    assert dead_features(base).shape == (400, len(DEAD_FEATURES)) == (400, 7) # seven dead columns
    assert team_features(base).shape == (400, len(TEAM_FEATURES)) == (400, 4) # four team columns


def test_team_swap_negates_team_features_exactly_and_leaves_dead_features_alone():
    """Team swap negates team features and leaves dead features alone."""
    players, speeds, teams, half = _random_match() # reuse random match
    a = base_series(players, speeds, teams, half) # original teams
    b = base_series(players, speeds, 1 - teams, half) # swapped teams
    ta, tb = team_features(a), team_features(b)   # team features both ways
    ok = np.isfinite(ta).all(axis=1)              # rows with full data
    # plenty of finite rows left to compare
    assert ok.sum() > 300 and (np.isfinite(tb).all(axis=1) == ok).all()
    # exact negation
    assert (ta[ok] == -tb[ok]).all()
    da, db = dead_features(a), dead_features(b)   # dead features both ways
    ok = np.isfinite(da).all(axis=1)              # rows with full data
    # dead features unchanged
    assert (da[ok] == db[ok]).all()


def _toy_model():
    """Small hand-written model with pinned weights."""
    return {"dead": {"features": list(DEAD_FEATURES), "mu": [1.0] * 7, "sd": [2.0] * 7, # hand-built model dict
                     "w": [0.5, -0.25, 1.0, -1.0, 0.3, -0.3, 0.1], "b": -0.2,
                     "prior_logit": -0.2},
            "team": {"features": list(TEAM_FEATURES), "sd": [1.5, 20.0, 3.0, 4.0],
                     "w": [0.8, 1.2, 0.3, 1.9]}}


def test_model_logit_is_exactly_antisymmetric_and_blind_rows_are_zero():
    """Team logit is antisymmetric; blind rows are zero."""
    players, speeds, teams, half = _random_match() # reuse random match
    a = base_series(players, speeds, teams, half) # original teams
    b = base_series(players, speeds, 1 - teams, half) # swapped teams
    ka = apply_model(dead_features(a), team_features(a), _toy_model()) # model on originals
    kb = apply_model(dead_features(b), team_features(b), _toy_model()) # model on swapped
    assert (ka.team_logit == -kb.team_logit).all() # exactly opposite
    assert (ka.dead_llr == kb.dead_llr).all()     # dead term unchanged
    assert (ka.team_logit[~ka.team_ok] == 0).all() and (ka.dead_llr[~ka.dead_ok] == 0).all() # blind rows are zero
    assert not ka.dead_ok[:75].any()              # the 3 s deceleration needs history


def test_model_with_another_feature_list_is_refused():
    """A model fitted on other features must be refused."""
    model = _toy_model()                          # start from toy model
    model["team"]["features"] = ["vx3", "terr", "dsx"] # swap in other features
    # expect ValueError; anything else fails below
    try:
        apply_model(np.zeros((2, 7)), np.zeros((2, 4)), model) # call must raise
    except ValueError:
        return                                    # refusal confirmed
    raise AssertionError("a model fitted on other features must be refused") # no refusal, fail test


def test_fit_recovers_a_planted_direction_and_stays_antisymmetric():
    """Fit recovers planted directions and mirrors under negation."""
    from tuning.kin_fit import degrade_players, fit_logistic

    rng = np.random.default_rng(11)               # fixed seed
    x = rng.normal(size=(4000, 3)) * np.array([1.0, 5.0, 0.2]) # features at set scales
    y = (x[:, 0] - 0.2 * x[:, 1] + rng.logistic(size=4000) * 0.5 > 0).astype(float) # planted label rule
    w, mu, sd = fit_logistic(x, y, intercept=False) # fit without intercept
    # planted directions recovered
    assert w[0] > 0 > w[1] and (mu == 0).all()
    w2, _, sd2 = fit_logistic(-x, 1.0 - y, intercept=False) # refit on negated data
    # negated data gives the same weights and spreads
    np.testing.assert_allclose(w, w2, rtol=1e-9)
    np.testing.assert_allclose(sd, sd2, rtol=1e-12) # spreads match too

    players = np.zeros((20000, 6, 2))             # big blank match
    hidden_p, hidden_s = degrade_players(players, np.ones((20000, 6)), rng) # apply dropout
    share = np.isnan(hidden_s).mean()             # measure hidden share
    # dropout share lands in the planted range
    assert 0.04 < share < 0.10 and (np.isnan(hidden_p[..., 0]) == np.isnan(hidden_s)).all()


# ---------------- harness and defaults ----------------

def test_stage9_grid_is_the_pinned_factorial():
    """Stage-9 grid is the pinned factorial."""
    from tuning.stage9 import DELAYS, P_PARAMS, REFERENCE, SLOPES, WEIGHTS, p9_cells

    cells = p9_cells()                            # build the grid
    assert len(cells) == 2 * (1 + 15 * 2) * 12 == 744 # pinned grid size
    assert cells[0] == REFERENCE                  # reference comes first
    assert len({tuple(c[p] for p in P_PARAMS) for c in cells}) == 744 # all cells unique
    assert WEIGHTS == [0.0, 0.03, 0.1, 0.3]       # pinned weight ladder
    assert DELAYS == [0.64, 0.80, 0.96, 1.12] and SLOPES == [0.0, 0.03, 0.06] # pinned delays and slopes
    # kin gate never paired with zero weights
    assert not any(c["gate_kin_support"] and c["kin_w_dead"] == 0 and c["kin_w_team"] == 0
                   for c in cells)


def test_shipped_model_is_the_pinned_shape():
    """Shipped model matches the pinned feature lists and shape."""
    from src.kinematics import load_model

    model = load_model()                          # load shipped model
    assert tuple(model["dead"]["features"]) == DEAD_FEATURES # dead list matches
    assert tuple(model["team"]["features"]) == TEAM_FEATURES # team list matches
    assert set(model["team"]) == {"features", "sd", "w"}          # no intercept, no centring
    for part, keys in (("dead", ("mu", "sd", "w")), ("team", ("sd", "w"))): # check each part
        for k in keys:                            # every parameter array
            assert np.isfinite(model[part][k]).all() and len(model[part][k]) == len( # finite and right length
                model[part]["features"])
    # fit provenance: 7 IDSSE matches
    assert len(model["fit"]["matches"]) == 7 and model["fit"]["source"].startswith("IDSSE")


def _synthetic_tables(effect):
    """Fake stage-9 tables built from the planted effect."""
    import pandas as pd

    from tuning.idsse import MATCH_IDS
    from tuning.stage5 import KINDS
    from tuning.stage9 import P_PARAMS, p9_cells

    rows = []                                     # accumulate table rows
    for c in p9_cells():                          # every grid cell
        for i, m in enumerate(MATCH_IDS):         # every match
            v = 0.40 + effect(c) + 0.001 * i      # planted score
            rows.append({**c, "match_id": m, "spell_f1_orig": v, "spell_f1_aligned": v}) # same score both kinds
    table = pd.DataFrame(rows).set_index(P_PARAMS + ["match_id"]).sort_index() # index by cell and match
    return {k: table for k in KINDS}              # same table for every kind


def _planted(c):
    """Planted per-cell effect the rule tests must rediscover."""
    k = {0.0: 0.0, 0.03: 0.004, 0.1: 0.020, 0.3: 0.021}[c["kin_w_dead"]] # dead-weight contribution
    t = {0.0: 0.0, 0.03: 0.0, 0.1: -0.01, 0.3: -0.03}[c["kin_w_team"]] # team-weight contribution
    return (0.01 * c["dwell_coalesce"] + k * (1.0 if c["gate_kin_support"] else 0.2) + t # total planted effect
            + (0.006 if c["switch_delay_s"] == 0.96 else 0.0)
            + 0.002 * (c["switch_delay_slope"] > 0))


def test_gate9_rule_keeps_steps_in_order_with_the_pinned_tie_break():
    """Gate rule keeps C/K/S, drops T, breaks ties to the pinned values."""
    from tuning import gate9

    rule = gate9.apply_rule(_synthetic_tables(_planted)) # run rule on fakes
    assert rule["C"]["keep"] and rule["K"]["keep"] and rule["S"]["keep"] # C, K, S kept
    assert not rule["T"]["keep"]                  # T dropped
    bundle = rule["bundle"]                       # chosen bundle
    assert bundle["kin_w_dead"] == 0.1 and bundle["gate_kin_support"] is True   # 0.3 is a tie
    assert bundle["kin_w_team"] == 0.0            # team term stays off
    assert bundle["switch_delay_s"] == 0.96 and bundle["switch_delay_slope"] == 0.0 # pinned delay, no slope
    # planted primary gain rediscovered
    assert rule["adopt"] and abs(rule["bundle_primary"] - 0.036) < 1e-9
    # K alone leaves coalescing off
    alone = gate9.step_alone(rule, "K")
    assert alone["kin_w_dead"] == 0.1 and alone["dwell_coalesce"] is False # K alone, coalescing off


def test_gate9_hold_switches_a_component_off_and_re_reads():
    """Holding K switches it off and re-reads the rest."""
    from tuning import gate9

    tables = _synthetic_tables(_planted)          # fake tables again
    held = gate9.apply_rule(tables, frozenset({"K"})) # run rule holding K
    assert held["K"]["held"] and held["bundle"]["kin_w_dead"] == 0.0 # K held and off
    assert held["bundle"]["gate_kin_support"] is False # kin tier off too
    assert not held["adopt"]                                      # 0.016 is below the bar
    # slope held at reference: only delays vary
    no_slope = gate9.candidates("S", dict(gate9.REFERENCE), frozenset({"slope"}))
    assert all(c["switch_delay_slope"] == 0.0 for c in no_slope) and len(no_slope) == 3 # only delays vary
    # full delay x slope set, reference cell skipped
    assert len(gate9.candidates("S", dict(gate9.REFERENCE), frozenset())) == 11


def test_committed_stage9_read():
    """The recorded read: C, K, T kept, S dropped, bundle passes the bar on all 8 tables —
    and the shipped config is exactly that bundle (hold released by the operator)."""
    from pathlib import Path

    import pytest

    from tuning import gate9

    if not (Path(gate9.RES) / "stage9_uniform.csv").exists(): # matrices are optional
        pytest.skip("stage-9 matrices not present") # skip without matrices
    tables = gate9.load_tables()                  # recorded stage-9 tables
    assert gate9.harness_pin(tables) <= gate9.PIN_TOL # harness pin holds
    rule = gate9.apply_rule(tables)               # run rule on real tables
    assert rule["C"]["keep"] and rule["K"]["keep"] and rule["T"]["keep"] # C, K, T kept
    assert not rule["S"]["keep"]                  # S dropped
    bundle = rule["bundle"]                       # chosen bundle
    assert (bundle["dwell_coalesce"], bundle["kin_w_dead"], bundle["kin_w_team"], # bundle equals recorded read
            bundle["gate_kin_support"]) == (True, 0.3, 0.03, True)
    assert rule["adopt"] and rule["bundle_primary"] > 0.02 and rule["bundle_wins"] == 7 # adopted with clear win
    assert all(v[0] > 0.08 and v[1] == 7 for v in rule["bundle_deltas"].values()) # every table improves
    for key in gate9.P_PARAMS:                    # every pinned parameter
        assert CFG[key] == bundle[key], key       # shipped config matches
