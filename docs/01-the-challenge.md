# 01 · The challenge — what we were asked to do

Start here. Ten minutes and you know the problem, the score, and the traps. Every fact
below is from the official documents in [`reference-docs/`](../reference-docs) — the
refs point at the exact lines.

## The problem in one paragraph

One football match — **FC Barcelona vs Athletic Club**, LaLiga 2024-25 — is recorded by
two cameras and annotated by machines: tracking (ball + player positions every frame),
tactical tags (phases, set pieces) and a homography per frame. Your job: **label every one
of 199,319 frames** with who is in possession — `H` (Barcelona), `A` (Athletic), `D` (ball
dead / not in play) — or `X` if you refuse to guess. Submit a CSV of 199,319 rows, a
public repository that regenerates it byte-for-byte, and a short technical document.
Deadline: **2026-11-17** ([SUBMISSION.md](../reference-docs/SUBMISSION.md)).

## The labels

| Label | Means | Watch out |
|---|---|---|
| `H` | home (Barcelona) in possession, ball alive | possession = *controlled* possession — a bare touch is not a change |
| `A` | away (Athletic) in possession, ball alive | a contested/loose ball stays with the last team that had it |
| `D` | ball dead / not in play | dead ball is a **stoppage, never a possession** — ownership is not asked |
| `X` | "no answer" | the scorer **forward-fills** it with your previous label in the half — so `X` = "repeat my last answer" |

Ground truth has exactly three states on graded frames (`H`/`A`/`D`) —
[CHALLENGE.md:91](../reference-docs/CHALLENGE.md#L91). The possession definition and its
edge cases (what counts as "control", what does not) are in
[CHALLENGE.md:55](../reference-docs/CHALLENGE.md#L55); when your reading disagrees with
the published 5-minute ground truth, the ground truth is right.

## Which frames count

- Graded: **frames 11,185–90,651** (first half) and **110,808–190,197** (second half) —
  158,857 frames ([DATA.md:249](../reference-docs/DATA.md#L249)).
- Everything else (pre-match, half-time, post-whistle) is ungraded: anything you put
  there is free.
- **Trap:** the second half is graded *from 110,808* but does not kick off until 113,742 —
  those 2,934 frames are graded and are **all `D`** ([FAQ.md:38](../reference-docs/FAQ.md#L38)).
  Our decoder has a rule for exactly this ([03 · the kick-off rule](03-how-the-decoder-works.md)).
- **The file is always full-length:** exactly 199,319 rows, `frame,label`, frames 0…199318
  in order, no blank cells (write `X` to abstain). Short files are the classic way to
  score zero ([FAQ.md:16](../reference-docs/FAQ.md#L16)).

## The score

**Ranking metric: `spellF1`. Tiebreak: `macroF1`** ([EVALUATION.md:48](../reference-docs/EVALUATION.md#L48)).

- A **possession** is a maximal run of `H` or of `A`. `H H D D H H` = two possessions with
  a stoppage between ([EVALUATION.md:50](../reference-docs/EVALUATION.md#L50)).
- Predicted possessions are matched **one-to-one against true possessions by temporal
  IoU**, and F1 of that matching is averaged over IoU thresholds 0.3–0.7 — the COCO mAP
  trick applied to time ([EVALUATION.md:53](../reference-docs/EVALUATION.md#L53)).
- Consequences that shaped our decoder:
  - the metric punishes **broken/extra segments** far more than small timing errors
    (1 frame late = free; 1 second late = 0.65 — [EVALUATION.md:64](../reference-docs/EVALUATION.md#L64));
  - per-frame accuracy is a trap: most frames sit mid-possession where nothing happens
    ([EVALUATION.md:57](../reference-docs/EVALUATION.md#L57));
  - `X` inside a possession is nearly free; across a hand-over it is expensive —
    **guess at transitions** ([FAQ.md:27](../reference-docs/FAQ.md#L27));
  - differences below **~0.04** spellF1 are noise on a full match
    ([EVALUATION.md:72](../reference-docs/EVALUATION.md#L72)).
- `macroF1`: frame-exact per-class F1 for H/A/D, unweighted mean — an always-`D` guess
  scores near the floor ([EVALUATION.md:33](../reference-docs/EVALUATION.md#L33)).
- Published reference: the `smooth50` baseline scores spellF1 **0.2433**
  ([EVALUATION.md:105](../reference-docs/EVALUATION.md#L105)).

**The score you see on the leaderboard comes from the private full-match ground truth.**
The published 5-minute window (17 possessions) is a worked example for pipeline checking —
tuning on it picks the wrong variant ([EVALUATION.md:127](../reference-docs/EVALUATION.md#L127)).

## The data you are given

| Source | What | Facts that matter |
|---|---|---|
| **Tracking (ATD)** | position + speed of every visible object per frame, **including the ball** | 46 slots for 22 players — no identities, slots swap/vanish ([DATA.md:99](../reference-docs/DATA.md#L99)); **ball missing on 41 % of graded frames**, worst exactly on crowded/duel frames ([FAQ.md:43](../reference-docs/FAQ.md#L43)); coordinates normalised 0–1 |
| **Smart Tagging** | 638 machine-made phase/set-piece intervals with tags | semi-automatic (noise); **intervals are phases, not possessions** — copying them as labels imports errors ([DATA.md:168](../reference-docs/DATA.md#L168)) |
| **Homography** | pitch→pixel matrix per frame | tactical is valid everywhere; the broadcast one is valid 65.8 % |
| **Broadcast video** | second camera view | optional input, **not synchronised** — aligning it is your problem ([DATA.md:223](../reference-docs/DATA.md#L223)) |

All of it may be used as model input, in any combination — "there is no intended
solution" ([DATA.md:18](../reference-docs/DATA.md#L18)).

## The rules (the short version)

| Forbidden | Ref |
|---|---|
| Hand annotation of any kind — including hand-correcting tracking or algorithm output | [RULES.md:41](../reference-docs/RULES.md#L41) |
| Fitting/validating on the provided game (it is the test set). Unsupervised learning and algorithmic pseudo-labels on it are fine | [RULES.md:54](../reference-docs/RULES.md#L54) |
| Any human-produced data about this match (event feeds, corrected tracking, annotations) | [RULES.md:73](../reference-docs/RULES.md#L73) |
| Submitting a file your repository cannot regenerate byte-for-byte | [RULES.md:86](../reference-docs/RULES.md#L86) |
| Private/unshareable dependencies; committing data to the repo | [RULES.md:7](../reference-docs/RULES.md#L7), [RULES.md:107](../reference-docs/RULES.md#L107) |
| Uploading provided videos/datasets to third-party cloud LLMs | [FAQ.md:109](../reference-docs/FAQ.md#L109) |

Everything else is legitimate: pre-trained models, LLMs, external data from **other**
matches, classical methods. Teams are 3–5 members. Unclear rule? **Ask the organizers
before building** ([RULES.md:120](../reference-docs/RULES.md#L120)).

## Why this is hard (the three real problems)

1. **The ball is missing 41 % of the time** — and missingness concentrates exactly where
   possession changes hands. Most of the decoder's error lives here.
2. **Possession is an interval, not a frame** — a per-frame classifier that flickers for
   one frame creates a false possession and pays dearly on spellF1.
3. **Dead ball has no owner** — `D` is a state you must detect, and the evidence for it
   (ball at rest, restart tags) disappears with the ball.

## How this project answered

A classical, fully inspectable decoder (emission + Viterbi + post-rules) tuned **only** on
external public matches under a written evidence protocol — never on the provided match.
The result: band A on the guarded evaluator runs, byte-identical regeneration, and every
decision on record.

**Next:** [02 · quickstart](02-quickstart.md) — reproduce the submission in five steps.
