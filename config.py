"""Single source of truth for every tunable parameter.

Rules (see dossier.md):
  * Every CFG key MUST have an entry in PROVENANCE with an origin of
    `prior`, `fixed_default` or `external_fit`. `provided_*` is never valid:
    nothing may be fitted or selected on the provided match.
  * `prior` values are starting points justified in one line here; they are
    replaced by `external_fit` (IDSSE evidence) during P5, never by
    in-match numbers.
  * tests/test_config.py fails the build if a key lacks provenance.
"""

CFG = dict(
    # --- frame / geometry (fixed) ---
    fps=25,                       # ATD metadata FrameRate=25.0
    pitch_length_m=105.3,         # games.py GAMES["BAR-ATH"]["pitch"] (measured)
    pitch_width_m=68.0,

    # --- geometry / evidence ---
    topk=3,                       # nearest slots used for team evidence
    sigma_m=4.0,                  # evidence falloff scale, metres
    pitch_margin=0.02,            # out-of-pitch margin in normalized units (~2 m)

    # --- decoder ---
    switch_pen=8.0,               # log-cost of H<->A switch (nats)
    dead_pen=5.0,                 # log-cost of entering/leaving D (nats)
    min_dwell_s=1.0,              # minimum segment length, seconds
    switch_support_s=0.6,         # non-ATD evidence must persist this long to switch

    # --- ball fill tiers ---
    fill_long_gap_s=10.0,         # kinematic fill covers bracketed gaps <= this
    fill_tau_s=1.0,               # boundary-velocity damping time
    fill_vel_window=5,            # known frames used for boundary velocity
    fill_pad=0.02,                # overshoot guard, normalized units
    fill_pad_frac=0.15,           # overshoot guard, fraction of span

    # --- confidence / tempering ---
    c0_atd=1.0,                   # confidence of raw tracking observations
    c0_kinematic=0.8,             # confidence of kinematic fills (before decay)
    tau_kinematic_s=3.0,          # confidence decay time for fills, seconds
    err_kinematic_m=1.5,          # position-error prior that widens sigma for fills

    # --- dead-ball term (sigmoid: p_d = sigmoid(d_bias + sum w_i f_i)) ---
    d_bias=-0.70,                 # logit(≈1/3): dead ball is about a third of football
    d_w_out_of_pitch=1.0,
    d_w_stationary=1.0,
    d_w_boundary=0.5,
    d_w_long_missing=0.5,
    d_w_low_activity=0.5,
    ball_stationary_thr_ms=1.0,   # m/s below which the ball counts as at rest
    d_missing_run_s=10.0,         # no-ball run length (s) that saturates the missing feature
    d_activity_thr_ms=2.0,        # mean player speed (m/s) below which play looks stopped

    # --- abstention policy (docs-derived, a priori) ---
    x_stable_s=5.0,               # only X when decoder this stable and confidence low
    x_conf_max=0.3,               # posterior margin below this counts as "low"
)

# key -> (origin, one-line justification)
PROVENANCE = {
    "fps": ("fixed_default", "ATD metadata FrameRate=25.0"),
    "pitch_length_m": ("fixed_default", "games.py pitch, measured from corrected tracking"),
    "pitch_width_m": ("fixed_default", "games.py pitch, measured from corrected tracking"),
    "topk": ("prior", "capped nearest-neighbour evidence; slots swap identity, so few and recent"),
    "sigma_m": ("prior", "ball-to-nearest-player distances in football are a few metres"),
    "pitch_margin": ("prior", "~2 m outside the touchline counts as out of play"),
    "switch_pen": ("prior", "H<->A must clear several seconds of contrary evidence (Jose plan init)"),
    "dead_pen": ("prior", "D transitions are restarts; cheaper than team switches (Jose plan init)"),
    "min_dwell_s": ("prior", "sub-second possessions are tracker flicker, not football"),
    "switch_support_s": ("prior", "weak (filled) evidence must persist before a switch (Jose plan init)"),
    "fill_long_gap_s": ("prior", "Hermite fill is trustworthy only over short bracketed gaps"),
    "fill_tau_s": ("prior", "damp boundary velocity so short gaps follow momentum (Jose design)"),
    "fill_vel_window": ("prior", "5 known frames = 0.2 s of velocity estimate at 25 fps"),
    "fill_pad": ("prior", "2 m overshoot guard on filled paths"),
    "fill_pad_frac": ("prior", "15% of gap span allowed beyond anchors"),
    "c0_atd": ("fixed_default", "raw tracking is full evidence"),
    "c0_kinematic": ("prior", "fills are estimates: start 20% below raw tracking"),
    "tau_kinematic_s": ("prior", "trust decays over a few seconds (Jose plan init)"),
    "err_kinematic_m": ("prior", "filled positions are a couple of metres off (Jose plan init)"),
    "d_bias": ("prior", "logit of ~1/3 dead ball; IDSSE sample showed 33% dead frames (P5 re-derives on all 7 matches)"),
    "d_w_out_of_pitch": ("prior", "ball beyond margin => dead (Jose WP3 design)"),
    "d_w_stationary": ("prior", "ball at rest => dead (restarts)"),
    "d_w_boundary": ("prior", "ball on the line => dead (throw-ins, corners)"),
    "d_w_long_missing": ("prior", "long no-ball runs correlate with stoppages (measured 75% of gaps are 5-30 s)"),
    "d_w_low_activity": ("prior", "low team activity during stoppages (Jose WP3 design)"),
    "ball_stationary_thr_ms": ("prior", "~1 m/s is a still ball; measured p50 pass speed is ~7 m/s (deep-dive 2)"),
    "d_missing_run_s": ("prior", "10 s without a ball: matches the median missing run (~10.7 s)"),
    "d_activity_thr_ms": ("prior", "players walking (<~2 m/s) is the stoppage signature"),
    "x_stable_s": ("prior", "EVALUATION.md: abstain inside a stable possession, guess at changes"),
    "x_conf_max": ("prior", "abstain only when the decoder is genuinely unsure"),
}

VALID_ORIGINS = ("prior", "fixed_default", "external_fit")


def provenance_problems():
    """Empty list = every CFG key is documented with a valid origin."""
    problems = []
    for key in CFG:
        if key not in PROVENANCE:
            problems.append(f"{key}: missing provenance")
            continue
        origin, note = PROVENANCE[key]
        if origin not in VALID_ORIGINS:
            problems.append(f"{key}: invalid origin {origin!r}")
        if origin == "external_fit" and not note:
            problems.append(f"{key}: external_fit requires IDSSE evidence in the note")
        if str(origin).startswith("provided"):
            problems.append(f"{key}: provided_* origins are forbidden")
    for key in PROVENANCE:
        if key not in CFG:
            problems.append(f"{key}: provenance for unknown CFG key")
    return problems
