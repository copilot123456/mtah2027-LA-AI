# 04 · Why these results — the decision story

How a decoder with no training on the target match ended at band **A**: the timeline of
every idea, what the evidence said, and why each yes/no was the right call. Read
[05 · the evidence protocol](05-the-evidence-protocol.md) first if you haven't — this
page is that protocol applied, twenty times.

**The shape of the story:** five phases (P6 → P10). Twenty pre-declared experiments.
Three adoptions, five clean rejections, one retraction and a closed redesign. The
submission was regenerated three times, each regeneration caused by an adopted change and
committed with it.

## The phases at a glance

| Phase | Question | Outcome | Submission after |
|---|---|---|---|
| **P6** (11-07 → 11-09) | Can external evidence improve the frozen decoder at all? | Gate 1 **REJECT** (frozen cell stays); E7 set-piece prior **REJECT**; E10 no-play rule adopted → **RETRACTED** → redesign **closed infeasible**; E1 label bed shipped | `aeef0058…` (unchanged) |
| **P7** (11-14) | The audit: does our evidence itself lie? | Units bug found and fixed; low-activity feature **OFF**; kick-off rule **ON** (operator); dead re-sweep **REJECT** | `1b0cf9e4…` (B) |
| **P8** (11-14) | First operator idea batch | Switch-gate move mode **adopt** + team-switch delay 0.64 s **adopt**; min-dwell variant drop; gap placement reject | `de14eb1c…` (A) |
| **P9** (11-14) | Second idea batch — player kinematics | Coalesce **adopt** + kinematic dead term 0.3 **adopt** + kinematic team term 0.03 **adopt**; speed-scaled delay drop | `c1217158…` (A) |
| **P10** (11-15) | Is the kinematic gain still rising past its grid edge? | **REJECT** — 0.3 is the knee | unchanged |

(Bands in parentheses = the guarded run that followed, operator-only.)

## The adoptions — why each cleared the bar

### P7 U1 — low-activity dead-ball feature: OFF

The feature marked frames dead when player movement was low. The audit found the external
bed had been scoring it in **km/h while the code assumed m/s** — so it had never really
been validated. On the units-corrected bed, weight 0 beats 0.5 on **8/8 tables, 7/7
matches (+0.025 … +0.061)**. Turning it *off* was the adoption. Side effect: > 20 % of
graded labels changed and the dead share dropped sharply — proof the feature had been
producing most of the dead-ball frames on unvalidated grounds.

### P7 K1 — kick-off tag rule: ON

A documents-only rule: from a half's start to its `Kick Off (Start)` tag, the frame is
`D` (the FAQ says the pre-kickoff span is graded and dead). Its first pre-declared check
failed on a clause we had mis-specified (the first-half tag sits just before the graded
span, where the rule is a no-op by design); the amended check passed and the **operator**
made the call. Effect: under 2 % of graded labels.

### P8 G + S — switch-gate move mode + 0.64 s team-switch delay

Hypothesis: when the ball is unobserved exactly at a possession change, the gate (which
demands ball support) *reverts* true switches and the decoder under-segments. The fix:
`move` the unsupported boundary to the nearest supported frame instead of deleting it, and
place ball-timed boundaries slightly later (0.64 s, derived externally). External
evidence **+0.031 (7/7)**. The stronger variant (place the boundary in the *middle* of a
ball gap) failed the pinned worst-case rule and stayed off — rejected honestly.

### P9 C + K + T — the bundle that made the jump

Three mechanisms, one pass (744 cells × 7 matches × 4 masks):

- **C** coalescing min-dwell (+0.014): the min-dwell merge left fragments of equal
  neighbours; joining them is worth a small, universal gain.
- **K** player-only dead-ball term on ball-less frames (weight 0.3, +0.017 primary, up to
  **+0.095** where ball loss is tied to dead ball): movement patterns say "stoppage"
  without the ball. Fitted LOMO on the external matches.
- **T** player-only team term on ball-less frames (weight 0.03, **+0.091** primary): which
  team is moving where says who has possession even when the ball is gone.

Bundle vs the pre-P9 config: **+0.122 primary (7/7), positive on all eight tables** — the
largest single jump of the project. Parity checks Q1–Q6 all PASS. The own-output bucket
came back **> 20 %**, so per the pinned rule the adoption was **held for the operator**,
who released it. The guarded run that followed: band **A**.

## The rejections — why each stopped

| Idea | Hypothesis | Why it failed | Lesson recorded |
|---|---|---|---|
| **Gate 1** grid closure (P6) | fine-tuning the frozen decoder cell | plateau candidate +0.0032 (3/7), CI straddles 0 | below bar → frozen cell ships; matrices kept as robustness evidence |
| **E7** set-piece prior (P6) | tag timing sharpens D boundaries | −0.0075 (2/7), all tables negative | rejected *feature*, weak evidence for the *signal* (external set-piece rows misaligned — see 06) |
| **E10** no-play rule (P6) | sustained inactivity = stoppage | gate passed but effect was **17× the envelope** on the target while firing on 0–1 external matches | gate was **vacuous**; adopted → retracted; redesign probed and closed infeasible |
| **P8 M** D-minimum-dwell | dead segments need their own dwell | no gain at the bar | — |
| **P8** gap placement | ball gaps should place H↔A boundaries | +0.026…+0.033 on three masks but +0.002 on the dead-tied one | pinned worst-case rule keeps it off |
| **P10** K weight > 0.3 | the K gain was rising at its grid edge | +0.011 at 0.4 decaying to +0.002 at 1.0 — all below +0.02 | the edge flattened; 0.3 is the knee; a fresh pin beat an extension |
| kinematic **fill** tier (P5) | estimated ball positions help | −0.0112 (7/7 negative) | estimated evidence hurts; ships off |
| **X** policy variants | abstention improves macroF1 | provably cannot gain spellF1 under the official forward-fill | discarded by construction |

## The audit twist (2026-11-14)

An independent design/process audit (record: `AUDIT.md`) asked a different question:
*is our evidence trustworthy?* Findings:

1. **The units bug** — the external bed's speed columns were km/h read as m/s. Fixed
   behind a self-check; every downstream number from before the fix is a km/h-bed record
   and is labelled as such. The fix invalidated the low-activity feature (U1 above) and
   made the kinematic work (P9) possible in a form we could trust.
2. **A scoreboard-incident disclosure** — recorded in `dossier.md` §3/§4. No number from
   it entered any decision.
3. **Re-sweep under the corrected bed** — the dead-ball term was re-tested and its
   re-sweep rejected (regime-agnostic rule: no cell gains under every ball-loss regime) —
   which is what set up P9's targeted kinematic answer later.

## The result

- Four guarded runs: **C → B → A → A**, every change adopted on external evidence before
  its run.
- The file regenerates **byte-identically** from a clean clone (cold cache) — md5
  `c121715836ed603fc5792b85daf6f279`.
- 182 tests; every shipped config value carries provenance; every rejection is on record.

The remaining weakness is known and documented —
[06 · limitations](06-limitations-and-open-questions.md).
