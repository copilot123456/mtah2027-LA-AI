# 08 · Mastery checklist — prove it to yourself

Interactive self-check. Work top to bottom; every box you can tick honestly means one
skill. The quiz answers are hidden — try first, then open. Nothing here touches the
forbidden files.

## Part 1 — hands-on verification (run and confirm)

Each command runs from the repo root with the environment of
[02 · quickstart](02-quickstart.md).

- [ ] **Reproduce the artifact.** `md5sum LA+AI.csv` → `c121715836ed603fc5792b85daf6f279`
- [ ] **Read a gate.** `python -m tuning.gate10` → the harness pin prints `0.0e+00` and
      the verdict is REJECT. Explain to yourself *why a REJECT can be good output*.
- [ ] **Break and fix.** Change one config value (e.g. `min_dwell_s` 2.0 → 1.0), run
      `src.submit`, and watch the md5 change. Then `git checkout config.py` and verify the
      md5 returns. You have just seen byte-identity as a contract.
- [ ] **Read the provenance.** Open [`config.py`](../config.py), find `kin_w_dead`, and
      state its `origin` and its `evidence` without looking at the glossary.
- [ ] **Follow one decision.** Pick any row of
      [04 · why these results](04-why-these-results.md), find its §7 block in
      [`dossier.md`](../dossier.md), and find its matrices under
      [`tuning/results/`](../tuning/results).

## Part 2 — the quiz

**Q1.** Why does `X` inside a possession cost almost nothing, but `X` at a hand-over cost
a lot?

<details><summary>answer</summary>

The scorer forward-fills `X` with your previous label in the half. Inside a possession the
previous label is already the right one; across a hand-over it is the old team's label,
which both misses the new possession and adds a false transition.
Source: [reference-docs/EVALUATION.md:84](../reference-docs/EVALUATION.md#L84).

</details>

**Q2.** Our decoder scores a per-frame accuracy of ~0.97 yet could still lose spellF1
badly. Why?

<details><summary>answer</summary>

spellF1 matches *possessions* (intervals) one-to-one. Per-frame accuracy is dominated by
long mid-possession stretches; a decoder that flickers at boundaries can keep high frame
accuracy while inventing many short possessions — each extra or broken segment is a
missed match in the IoU pairing.
Source: [reference-docs/EVALUATION.md:57](../reference-docs/EVALUATION.md#L57).

</details>

**Q3.** Why can't the switch gate simply *believe* a boundary when the ball is missing?

<details><summary>answer</summary>

Because unobserved frames are exactly where false boundaries are cheapest to invent and
true ones are easiest to lose. The gate demands support (ball, or since P9 kinematics);
unsupported boundaries move to the nearest supported frame. The alternatives were all
measured: reverting them loses true switches (under-segmentation), placing them mid-gap
failed the worst-case rule.

</details>

**Q4.** P9's bundle changed > 20 % of graded labels. The rule said "hold". Why is that a
feature of the protocol, not a bug?

<details><summary>answer</summary>

A change that large is beyond what the external bar alone should authorize — the operator
must see it and take responsibility. The hold produced exactly that: evidence shown,
decision explicit, adoption recorded. Silent large changes are how pipelines rot.

</details>

**Q5.** The K gain was "still rising at the grid edge" after P9. P10 extended the grid and
found +0.011 at weight 0.4 decaying to +0.002 at 1.0. What is the lesson?

<details><summary>answer</summary>

"Rising at the edge" is a hypothesis, not evidence — the fresh pre-declared pin (not a
post-hoc extension) is what converts it into knowledge. The trend flattened right past
the edge; the bar (> +0.02) never cleared. The shipped value 0.3 is the knee. See
[04](04-why-these-results.md) and `tuning/results/report_stage10.md`.

</details>

**Q6.** Name two things this project refused to do even though they might have raised the
score.

<details><summary>answer</summary>

Any two of: fitting on the provided match or its public window; hand-correcting tags or
tracking; adopting the "no ball → dead" variant at near break-even; extending the K grid
after reading; choosing between submission files with the evaluator; frame-specific rules
like "D before frame 113,742".

</details>

**Q7.** What stops a future contributor from re-running the stage-9 matrices with a
friendlier rule and "finding" a better cell?

<details><summary>answer</summary>

Three things: the pre-declaration (the rule was pinned before compute), the tests
(`tests/test_tuning.py` re-applies the pinned rule and must reproduce the shipped cell
exactly), and the record rule in [05](05-the-evidence-protocol.md): matrices are
immutable; a different question needs a new pin and new compute.

</details>

## Part 3 — explain it (the real test)

Write these out loud, without opening anything. If you can, you have mastered the project.

1. **The pipeline in six sentences** — data → track → emission → Viterbi → post-rules → CSV.
2. **Why the +0.02 bar** — what would happen to the selection with a +0.005 bar.
3. **The story of the dead-ball problem** — what we knew, what we tried, what is still open
   ([06](06-limitations-and-open-questions.md)).
4. **The E10 story** — the one time the protocol failed and what was rebuilt because of it.
5. **What you would do next** — pick one item from [06](06-limitations-and-open-questions.md)
   and write its §7 pre-declaration skeleton (grid, rule, no-harm, hold).

## Part 4 — reading path (the full tour)

- [ ] [01 · the challenge](01-the-challenge.md) — what and why
- [ ] [02 · quickstart](02-quickstart.md) — reproduce it
- [ ] [03 · how the decoder works](03-how-the-decoder-works.md) — the mechanism
- [ ] [04 · why these results](04-why-these-results.md) — the decision story
- [ ] [05 · the evidence protocol](05-the-evidence-protocol.md) — the operating system
- [ ] [06 · limitations](06-limitations-and-open-questions.md) — the honest page
- [ ] [07 · glossary](07-glossary.md) — as needed
- [ ] [`TECHNICAL_REPORT.md`](../TECHNICAL_REPORT.md) — the submission document
- [ ] [`dossier.md`](../dossier.md) §7 — one full pre-declaration + read (P9 is the richest)
- [ ] [`tuning/results/report_stage9.md`](../tuning/results/report_stage9.md) — read a real matrix read-out
- [ ] [`AUDIT.md`](../AUDIT.md) — the outside view
- [ ] [PIPELINE_STANDARD_AI.md](../PIPELINE_STANDARD_AI.md) + [AI-SOP-SKILL.md](../AI-SOP-SKILL.md) — the team standards

**Time to mastery:** ~4 hours of reading + ~30 minutes of hands-on.
