# 05 · The evidence protocol — how this project decides

This page is the operating system of the project. Every result in
[04 · why these results](04-why-these-results.md) was produced by these rules. If you
change the decoder, you follow them too — they are the reason the numbers mean anything.

## The one rule everything follows

> **A behaviour change ships only if a test written *before* seeing the answer says it
> should, and that test runs on data we are allowed to tune on.**

Everything below is machinery for making that rule real.

## 1. The forbidden data (the strict stance)

| Data | May we fit on it? | May we evaluate on it? | May we read it at all? |
|---|---|---|---|
| Provided match tracking | **No** | No | Coarse aggregates only, through approved diagnostics |
| Ground truth / scoreboard / result boards | **No** | Only via the guarded run (operator) | **No** — never opened |
| Broadcast video | **No** | No | **No** |
| Public window (graded span) figures | **No** — not even the possession count | No | Yes (it's published) |
| External IDSSE matches (7, CC BY 4.0) | Yes | **Yes — our only bed** | Yes |
| Challenge documents (FAQ, rules) | Rules may be encoded as *a-priori* rules | — | Yes |

Why so strict? Because the leaderboard is public: tuning against it (even indirectly, even
"just once") turns the score into a leak. The project treats the provided match as a
*test*, not as *training data*.

## 2. Pre-declaration — the anti-post-hoc device

Before any compute that could influence a decision, the full experiment is written into
[`dossier.md`](../dossier.md) §7:

- the **grid** (every cell that will be scored),
- the **decision rule** (what counts as adopt / keep / reject — thresholds, tie-breaks),
- the **no-harm tables** (what must not get worse),
- the **hold rules** (when the operator must decide instead of the rule).

The pre-declaration gets its own commit *before* the run. Then the matrices are computed,
the gate applies the pinned rule mechanically, and the read lands in the dossier. The
grid is never extended "because the trend looks promising" after the read (P10 was
pre-declared precisely *because* P9's K trend hit its grid edge — a fresh pin, not an
extension).

**Why:** the E10 episode. A rule passed its gate, shipped, and its measured effect was 17×
the approved envelope — the gate was *vacuous* (the rule fired on 0–1 of 7 external
matches, so the gate could not see an over-fire). Two new rules came out of it: gates must
report **external firing rates**, and a post-hoc grid change is forbidden.

## 3. The external bed — IDSSE

All tuning runs on 7 external matches with a pipeline identical to the shipping one:

- **paired deltas**: same matches, same masks, same seeds — candidate minus reference, per
  match, then averaged. A mean without its per-match wins (n/7) is not evidence.
- **4 mask kinds** (uniform / phase_a2 / cover_q50 / cover_q90): how the ball is hidden
  from the decoder, simulating the provided match's missing ball evidence. The phase
  masks concentrate ball loss around possession changes and crowds — the hard case.
- **2 label sets**: `orig` (per-touch) and `aligned` (challenge-definition labels,
  `derive_letters_controlled`) — because the raw external labels under-count long
  possessions and would bias selection toward short spells.
- **8 tables** = 4 masks × 2 label sets. Primary table: `phase_a2` × `aligned`
  (the hardest realistic regime on the faithful labels).
- **LOMO** where a model is fitted: fit on 6 matches, test on the 7th. The kinematic
  models are never fitted on the match they score.
- **Harness pin:** every stage's harness must reproduce the previous stage's shipped row
  *exactly* (max |Δ| = 0) before its own matrices are read — otherwise the pass is void.
  And the harness scorer itself is pinned to the official scorer's entry point (all of its
  post-processing included) — the X-fill artifact taught that one.
- **Target parity checks** (`tuning/kin_parity.py`): categorical PASS/FAIL tests that the
  provided match looks enough like the external data for a fitted model to transfer
  (speed scale, dead share, team balance, visible players…). A FAIL holds the adoption
  for the operator.

## 4. The adoption bar

> **Adopt only if:** primary-table paired delta > **+0.02**, positive on **≥ 5/7**
> matches, **and** ≥ **−0.02** on every other table (no-harm).

Below the bar: the current value ships and the matrices stay as robustness evidence. This
happened five times (Gate 1, E7, P8 M, P10, gap placement) — a rejected idea recorded
honestly is a successful output of the process.

## 5. The guarded evaluator — correctness, never selection

Once per code path, the **operator** runs `python -m tuning.guarded_eval`. It scores the
submission on the public window behind a redacting wrapper and returns only:

- a **band letter** (A/B/C/D), and
- four **categorical flags** (possession count, f1_D, H/A balance, transition latency).

No score ever enters an AI channel. The run answers exactly one question: *"did the file
break?"* — it is never used to choose between two files. Every adoption in this project
was decided on IDSSE **before** its guarded run, and the run confirmed the file was not
broken.

## 6. Holds — when the rule steps back

The pinned rule can trigger a hold; then the **operator** decides:

- own-output change > **20 %** of graded labels (P9 bundle: held → released → adopted),
- a parity check FAIL,
- any situation the pre-declaration named as operator-only.

## 7. The five recorded lessons (feedback rules)

| Lesson | In one line |
|---|---|
| Vacuous gates | A gate on a rule that never fires externally cannot see an over-fire (E10: it passed and the effect was 17× the envelope) — always report external firing rates, and root-cause "no firing opportunity" (units? scale?) before accepting it as football. |
| Rejected feature ≠ rejected signal | A feature failing its gate doesn't mean the underlying signal is worthless — the dead threshold lost on 8/8 tables, yet the same motion signal as a smoothed multi-feature model gave P9's biggest gains. Record *what form* was rejected; probe separability before closing an idea. |
| Evaluator-derived ideas are hypotheses | Anything the band/flags suggest is a *hypothesis* to pre-declare and test externally — never tuning input; ideas whose only possible evidence is the public window are rejected outright. |
| Test-bed target parity | Before trusting external gains: assert unit plausibility, compare feature firing rates bed-vs-target (a gap is a bug until proven otherwise), score baselines on the bed, and check the bed for label leaks (the frozen-ball trap) and event-table misalignment (E7). |
| Pin the harness to the official scorer | The tuning scorer must call — or prove equality with — the official `evaluate()` with *all* its post-processing. The X-fill artifact (a missing `fill_abstentions`) once reached three documents. |

## 8. Writing it all down

- [`dossier.md`](../dossier.md) — every pre-declaration, every read, every evaluator look.
  Append-only history.
- [`tuning/results/`](../tuning/results) — immutable matrices + reports. Never re-run
  selection with a different rule over old matrices.
- [`config.py`](../config.py) — every shipped value carries `origin` + `evidence`
  provenance (test-enforced).
- [AI-SOP-SKILL.md](../AI-SOP-SKILL.md) — how a branch reports its work (human + AI pair).

## Next

See the protocol produce results:
**[04 · why these results](04-why-these-results.md)**.
