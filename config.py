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
    sigma_m=1.5,                  # evidence falloff scale, metres (was 4.0 prior)
    pitch_margin=0.02,            # out-of-pitch margin in normalized units (~2 m)

    # --- decoder ---
    switch_pen=2.0,               # log-cost of H<->A switch (nats; was 8.0 prior)
    dead_pen=5.0,                 # log-cost of entering/leaving D (nats)
    min_dwell_s=2.0,              # minimum segment length, seconds (was 1.0 prior)
    switch_support_s=1.5,         # non-ATD evidence must persist this long to switch
    gate_switches=True,           # revert switches unsupported by nearby ATD frames (P5 ablates)
    no_play_rule=False,           # E10: D over sustained low-activity runs — gate PASSED but
                                  # ADOPTION RETRACTED 2026-11-09: measured effect 17x the
                                  # plan's envelope (30.9% of graded frames flip, D share
                                  # 26.6% -> 57.5%; IDSSE gate was vacuous — rule fires 0-1
                                  # matches externally). Operator decision 2026-11-09:
                                  # redesign probed and closed as infeasible (dossier §7).
    no_play_s=10.0,               # sustained inactivity (s) that marks a stoppage (a priori)
    kickoff_rule=True,            # K1: D from a graded half's start up to its `Kick Off (Start)`
                                  # tag (dossier §7 P7); ON by operator decision 2026-11-14 with
                                  # the amended first-half check (tuning/kickoff_check.py)

    # --- ball fill tiers ---
    # fill_long_gap_s = 0 DISABLES the kinematic tier: the IDSSE layer ablation
    # (kinematic - none = -0.0112 spellF1, CI [-0.017, -0.007], 7/7 matches negative;
    # all-cell mean -0.0089, positive in only 211/1701 cells) failed the keep-rule.
    # The other fill_* priors stay documented for re-enable/ablation; they are
    # inactive while this is 0. The sweep's `none` scenario uses the identical value.
    fill_long_gap_s=0.0,          # 0 = tier off; was 10.0 (prior) before the ablation
    fill_tau_s=1.0,               # boundary-velocity damping time (inactive)
    fill_vel_window=5,            # known frames used for boundary velocity (inactive)
    fill_pad=0.02,                # overshoot guard, normalized units (inactive)
    fill_pad_frac=0.15,           # overshoot guard, fraction of span (inactive)

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
    d_w_low_activity=0.0,         # OFF (was 0.5 prior): the IDSSE bed ran this feature on km/h, so it
                                  # was never validated; on the units-corrected bed weight 0 beats
                                  # 0.5 on 8/8 tables, 7/7 matches (dossier §7 P7 U1, 2026-11-14)
    d_w_set_piece=0.0,             # E7: OFF; ON value 1.0 a priori (dossier §7); dead-logit only
    sp_pre_s=2.0,                  # E7 window: seconds before a set-piece tag start (plan §2)
    sp_post_s=1.0,                 # E7 window: seconds after a set-piece tag start (plan §2)
    ball_stationary_thr_ms=1.0,   # m/s below which the ball counts as at rest
    d_missing_run_s=10.0,         # no-ball run length (s) that saturates the missing feature
    d_activity_thr_ms=2.0,        # mean player speed (m/s) below which play looks stopped

    # --- abstention policy (docs-derived, a priori) ---
    x_use=False,                  # X on graded frames OFF by default: for a Viterbi-smoothed
                                  # path, emit weakly dominates X (analysis in x_policy docstring);
                                  # P5 ablates on IDSSE with x_stable_s/x_conf_max
    x_stable_s=5.0,               # only X when decoder this stable and confidence low
    x_conf_max=0.3,               # posterior margin below this counts as "low"
)

# key -> (origin, one-line justification)
PROVENANCE = {
    "fps": ("fixed_default", "ATD metadata FrameRate=25.0"),
    "pitch_length_m": ("fixed_default", "games.py pitch, measured from corrected tracking"),
    "pitch_width_m": ("fixed_default", "games.py pitch, measured from corrected tracking"),
    "topk": ("prior", "capped nearest-neighbour evidence; slots swap identity, so few and recent"),
    "sigma_m": ("external_fit", "IDSSE 930-cell union: 1.5 (prior 4.0); interior peak found "
                 "in stage 3 — marginals 1.0:0.2627, 1.5:0.2644, 2.0:0.2417, 3.0:0.2245"),
    "pitch_margin": ("prior", "~2 m outside the touchline counts as out of play"),
    "switch_pen": ("external_fit", "IDSSE: 2.0 (prior 8.0); 2-vs-4 flat at noise, both above "
                   "8 (0.2263/0.2251/0.2196/0.2042 union marginals)"),
    "dead_pen": ("prior", "IDSSE surface flat (0.2195-0.2198 across 3-8): value immaterial; "
                 "prior from Jose plan init retained (center of range)"),
    "min_dwell_s": ("external_fit", "IDSSE: 2.0 s (prior 1.0); interior peak — "
                    "1.0:0.2343, 2.0:0.2394, 3.0:0.2167"),
    "switch_support_s": ("external_fit", "IDSSE: 1.5 s (prior 0.6); flat 1.0-1.5 at the top, "
                         "both well above 0.3"),
    "gate_switches": ("prior", "tier-2/3 must not switch alone; ablated +/- on IDSSE (scenario "
                      "results embed gating on)"),
    "no_play_rule": ("prior", "E10 compliance fix (FAQ: pre-kickoff frames are D): D over "
                     "sustained low-activity runs. No-harm gate PASSED 2026-11-09 (IDSSE "
                     "mean -0.0001, worst -0.0005) but adoption RETRACTED same day: the rule "
                     "fires on 30.9% of our graded frames (D share 26.6% -> 57.5%, 17x the "
                     "plan's 1.8% envelope) while firing on 0-1 IDSSE matches — the gate was "
                     "vacuous. Flag stays False — the redesign was probed and closed as "
                     "infeasible 2026-11-09 (dossier §7); full record "
                     "dossier §7 + tuning/results/no_harm_no_play.csv"),
    "no_play_s": ("prior", "10 s of unbroken low activity = stoppage, same scale as "
                   "d_missing_run_s; a priori (dossier §7 E10 pre-declaration)"),
    "kickoff_rule": ("prior", "K1 documents-only rule (FAQ: the second half is graded before "
                     "its kick-off and those frames are all D; DATA.md: Smart Tagging "
                     "`Kick Off (Start)`, time base checked against both kickoffs). No "
                     "external gate is possible (IDSSE halves start at the kick-off); "
                     "pre-declared dossier §7 P7 2026-11-14 with a categorical check; "
                     "switched ON by operator decision 2026-11-14 under the amended "
                     "first-half clause (no tag inside, or span <= 250 frames)"),
    "fill_long_gap_s": ("external_fit", "kinematic fill tier DISABLED (was prior 10.0): "
                         "IDSSE layer ablation kinematic-none = -0.0112 spellF1, "
                         "CI [-0.017, -0.007], 7/7 matches negative, all-cell mean -0.0089"),
    "fill_tau_s": ("prior", "damp boundary velocity so short gaps follow momentum (Jose design)"),
    "fill_vel_window": ("prior", "5 known frames = 0.2 s of velocity estimate at 25 fps"),
    "fill_pad": ("prior", "2 m overshoot guard on filled paths"),
    "fill_pad_frac": ("prior", "15% of gap span allowed beyond anchors"),
    "c0_atd": ("fixed_default", "raw tracking is full evidence"),
    "c0_kinematic": ("prior", "fills are estimates: start 20% below raw tracking"),
    "tau_kinematic_s": ("prior", "trust decays over a few seconds (Jose plan init)"),
    "err_kinematic_m": ("prior", "filled positions a couple of metres off (Jose plan init); "
                        "inactive while the fill tier is off"),
    "d_bias": ("prior", "logit of ~1/3 dead ball; external re-fit to the IDSSE 45% D share "
               "was TRIED and REJECTED (-0.177 spellF1, post_checks) — prior retained with evidence"),
    "d_w_out_of_pitch": ("prior", "ball beyond margin => dead (Jose WP3 design)"),
    "d_w_stationary": ("prior", "ball at rest => dead (restarts)"),
    "d_w_boundary": ("prior", "ball on the line => dead (throw-ins, corners)"),
    "d_w_long_missing": ("prior", "long no-ball runs correlate with stoppages (measured 75% of gaps are 5-30 s)"),
    "d_w_low_activity": ("external_fit", "OFF (was prior 0.5): units-corrected IDSSE stage 5, "
                         "weight 0 minus 0.5 = +0.0492 spellF1 on the primary table (phase "
                         "masks x aligned labels), 7/7 matches, positive on all 8 tables "
                         "(+0.025 ... +0.061); dossier §7 P7 U1, tuning/results/report_stage5.md"),
    "d_w_set_piece": ("prior", "E7 set-piece prior: OFF by default, ON value 1.0 a priori "
                     "(plan §2 E7); gated on IDSSE > 0.02 paired, dossier §7 E7 "
                     "pre-declaration 2026-11-09; timing only, no team/type"),
    "sp_pre_s": ("prior", "E7 window [start - 2 s, start + 1 s] a priori (plan §2 E7)"),
    "sp_post_s": ("prior", "E7 window [start - 2 s, start + 1 s] a priori (plan §2 E7)"),
    "ball_stationary_thr_ms": ("prior", "~1 m/s is a still ball; measured p50 pass speed is ~7 m/s (deep-dive 2)"),
    "d_missing_run_s": ("prior", "10 s without a ball: matches the median missing run (~10.7 s)"),
    "d_activity_thr_ms": ("prior", "players walking (<~2 m/s); inactive while d_w_low_activity is 0 "
                          "and no_play_rule is False. In m/s this fires on 36-42% of live IDSSE "
                          "frames - too high to be a stoppage signature (dossier §7 P7 U0)"),
    "x_use": ("prior", "off by default: emit>=X for a dwell/gated path; IDSSE ablation under the "
              "official scoring: spellF1 identical (the fill restores the label), macroF1 -0.11"),
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
