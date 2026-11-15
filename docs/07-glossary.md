# 07 · Glossary

Every word the project uses, in one place. Return here whenever a document speaks a
language you haven't met yet.

## The challenge

| Term | Meaning |
|---|---|
| **MTAH 2027** | More Than A Hack 2027 Initial Challenge (Mobile World Capital / Talent Arena). |
| **BAR-ATH** | FC Barcelona vs Athletic Club — the one match the submission labels. |
| **graded span** | The frame ranges the scorer grades. Frames outside it are ours to fill but they don't count. |
| **frame** | One tracking timestamp. Our match has 199,319 of them (rows 0…199318). |

## The labels

| Term | Meaning |
|---|---|
| **H** | Home team (FC Barcelona) possession. |
| **A** | Away team (Athletic Club) possession. |
| **D** | Dead ball / stoppage — no team has possession (kick-off, throw-in, foul, half-time…). |
| **X** | Abstain. The scorer forward-fills through abstentions; our decoder emits `X` only outside the graded spans. |
| **spell** | One contiguous run of the same label — a "possession spell". `spellF1` compares predicted spells to true spells. |
| **possession** | A spell of `H` or `A` (with `D`/`X` gaps bridged by the scorer where the truth says the possession continued). |

## The metric

| Term | Meaning |
|---|---|
| **spellF1** | The ranking metric: F1 over possession spells (segments), not over frames. A boundary that is 2 seconds late can still be a wrong spell — the metric punishes broken segments more than pixel-perfect timing. Details: [reference-docs/EVALUATION.md](../reference-docs/EVALUATION.md). |
| **macroF1** | Per-class F1 averaged over classes — the tiebreak metric. |
| **band A/B/C/D** | The categorical score band our guarded evaluator run returns on the public window. A correctness gate only, never a selector. |

## The data

| Term | Meaning |
|---|---|
| **tracking (ATD)** | The provided tracking feed: per-frame ball and player positions. |
| **Smart Tagging** | The provided event tags (kick-off, throw-in, free kick, corner…). Semi-automatic: carries timing noise. |
| **homography** | The camera-to-pitch mapping (used by the starter kit; our pipeline works in pitch coordinates). |
| **IDSSE** | The external public tracking dataset (7 matches, CC BY 4.0) — our *only* tuning/evaluation bed. Never the provided match. |

## The decoder

| Term | Meaning |
|---|---|
| **emission** | Per-frame log-scores for each label — how well "this frame is H/A/D" fits the evidence (`src/emission.py`). |
| **Viterbi** | The dynamic program that finds the best label *sequence* (not the best per-frame label) (`src/decoder.py`). |
| **switch** | An H↔A boundary. The decoder has a switch gate: a boundary must have ball support (or kinematic support) or it moves/reverts. |
| **gate_mode** | What the switch gate does to an unsupported boundary: `revert` (drop it) or `move` (slide it to the nearest supported frame — P8 G). |
| **min-dwell** | Minimum seconds a label must last. Prevents one-frame flicker. |
| **coalesce** | After min-dwell merges, join equal neighbours again (`dwell_coalesce`, P9 C). |
| **D-share** | Share of frames labeled `D` in our own output — a sanity statistic, never a tuning target. |
| **fill** | Filling unobserved ball positions. Our kinematic fill tier ships **off** (it hurt on the external bed). |
| **kinematic terms** | Player-only movement features (dead-ball likelihood, team likelihood) used on frames where the ball is unobserved — the P9 K and T terms, fitted on the external IDSSE matches. |

## The evidence protocol

| Term | Meaning |
|---|---|
| **pre-declaration** | Writing the grid, the decision rule and the no-harm tables into `dossier.md` §7 *before* running any compute. The defense against post-hoc selection. |
| **adoption bar** | Adopt a change only if it beats the current config by > +0.02 paired spellF1 on ≥ 5/7 external matches and is not worse than −0.02 on any other table. |
| **no-harm table** | One of the 8 evaluation tables (4 mask kinds × 2 label sets). A change must not hurt any of them. |
| **mask kinds** | How the external ball is hidden from the decoder (uniform / phase_a2 / cover_q50 / cover_q90) — simulating missing ball evidence. |
| **label sets** | `orig` (per-touch labels) vs `aligned` (challenge-definition labels from `derive_letters_controlled`). |
| **paired delta** | Per-match score difference candidate − reference, then averaged over 7 matches. Paired = same matches, same masks, same seeds. |
| **LOMO** | Leave-one-match-out: fit on 6 external matches, test on the 7th. |
| **guarded evaluator run** | Operator-only run of the official evaluator on the public window. Only the band letter and categorical flags ever reach the AI. Correctness gate, never a selector. |
| **vacuous gate** | A no-harm check on a rule that never fires externally — it cannot see an over-fire. The E10 lesson. |
| **parity checks** | Categorical PASS/FAIL checks (`tuning/kin_parity.py`) that the provided match looks enough like the external data for a kinematic model to transfer. |
| **own-output check** | How many of our own labels change with an adoption, reported as a bucket only (> 20 % ⇒ held for the operator). |

## The repository

| Term | Meaning |
|---|---|
| **dossier** | [`dossier.md`](../dossier.md) — the compliance and decision record. History: never rewritten, only appended. |
| **frozen cell** | The decoder configuration selected at P6 and defended since (switch 2 / dead 5 / σ 1.5 / dwell 2.0 / support 1.5, fill off). |
| **P6…P10** | Improvement phases (P6 = the external-gate phase, P7 = audit follow-up, P8/P9 = idea batches, P10 = K-weight edge check). |
| **U1 / K1 / D1 / G / S / M / C / K / T** | Component codenames in the decision log (e.g. K = kinematic dead-ball term). See [04 · why these results](04-why-these-results.md). |
| **branch report pair** | `reports/<slug>.html` + `reports/<slug>_AI.md` — the shareable story of one branch ([AI-SOP-SKILL.md](../AI-SOP-SKILL.md)). |
| **validator** | The official format checker that runs inside `src.submit`. "Valid." means format only — not correctness. |
