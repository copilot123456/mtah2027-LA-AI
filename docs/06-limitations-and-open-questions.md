# 06 · Limitations & open questions

The honest page. Every limitation below is real, documented in the record, and none of it
is hidden in a footnote. If you extend this project, start here.

## What the decoder cannot do

### 1. Dead-ball recall where the ball is unobserved

**The flag `f1_D` read `low` in all four guarded runs** — before and after the dead-ball
share of the file roughly tripled (P9). The decoder has almost no evidence for `D` when
the ball is missing: our own audit showed missed-`D` frames are overwhelmingly
ball-absent (documented in `dossier.md` §7, E7 audit).

- What we did about it: the P9 kinematic dead-ball term (K) works exactly on ball-less
  frames and pushed the external evidence up sharply (+0.05 … +0.095 where ball loss is
  tied to stoppages).
- What would actually fix it: **telling dead gaps from live gaps** — Smart Tagging
  set-piece/restart semantics or a broadcast ball detector. Both are blocked (below).
- Caveat on the flag itself: it is a bucket over 17 possessions. It is not evidence for or
  against any mechanism — it is a reminder of where the evidence is thin.

### 2. The external bed cannot judge ball-visible dead-ball features

On IDSSE the ball is *frozen exactly at rest* while dead — so any "ball at rest" feature
reads the dead-ball label there and scores artificially well under some masks. Recorded
verdict: **do not tune ball-visible dead-ball features on this bed** (`report_stage8.md`,
dossier §7 P8). The record-only rows of stages 9/10 (`R_kin_all`, `R_K05_all`) exist to
document exactly this boundary.

### 3. The kinematic models are visitors

The P9 kinematic models are fitted on seven Bundesliga matches tracked by a *different*
system. The parity checks show the provided match looks alike on six coarse statistics —
likeness, **not transfer**. The gains are real on the external bed; how they transfer is
only observable through the guarded runs (which came back A).

### 4. Smart Tagging semantics are unverified

The event tags are semi-automatic (timing noise) and their semantics for restarts could
not be verified without fitting on the provided match (forbidden). Consequences:

- the set-piece prior (E7) was rejected on a bed whose set-piece rows are **misaligned on
  most external matches** — a *weak* rejection (see the "rejected feature ≠ rejected
  signal" lesson);
- the team-phase prior is **not decidable** on IDSSE (no tagging there);
- the kick-off rule (the one tag rule we did ship) is a documents-only rule with an
  operator-approved amended check.

### 5. Switch timing on ball-less boundaries

With P9's T term the switch gate can place team boundaries using player kinematics when
the ball is gone, but the timing evidence is weaker there: latency flags read `none`, yet
the external bed shows the cost of T depends heavily on *when* ball loss happens (≈ 0.01
spellF1 if ball loss is almost only stoppages).

## Process limitations (what the evidence cannot tell us)

| Limitation | Consequence |
|---|---|
| 7 external matches | Paired deltas below ~±0.01 are noise; the +0.02 bar exists because of this. |
| 17-possession graded window | Band letters are correctness gates, not forecasts. |
| One match to submit | No measure of per-match variance on the target; LOMO on IDSSE is the best proxy. |
| Mask kinds are simulations | The real missingness pattern of the broadcast/tracking feed is unknown; the four kinds bracket plausible regimes but don't reproduce it. |

## Open questions (organizers)

The five questions drafted for the organizers (Q1–Q5) live in
`results/mtah2027/audit-2026-11-14/organizer-email-draft.md` and in `dossier.md` §6.
The two that gate future work:

- **Q1** — may the provided *tracking* be used for fitting (not just evaluation)?
  Everything calibration-shaped is blocked until this is answered.
- **Q5** — is ball loss in the graded window tied to dead ball? This decides the
  "no ball → dead" variant (a coin-flip by our own analysis; superseded in practice by K).

## If you continue this work (ranked, per the P9 record)

1. **Smart Tagging restart analogue** — needs a *new* external analogue (restart = end of
   a dead run + timing noise, misses, spurious) and a label-free timing check against the
   decoder's own restart boundaries. E7's rejection is weak evidence to redo.
2. **Team-phase prior** — documents-only adoption is an operator call; the T plumbing
   would carry it. Not decidable on IDSSE.
3. **Kinematic dead-ball term on ball-visible frames** — record-only until there is a bed
   where the ball moves while dead. The ball-detector track (a separate branch) is the
   route to that evidence.
4. **K weight beyond 0.3** — closed: the gain is at its knee (stage 10). Reopen only with
   genuinely new evidence.

Each item needs its own §7 pre-declaration and one guarded run after — see
[05 · the evidence protocol](05-the-evidence-protocol.md).
