# 09 — Cost under-estimated ~6× ($5–6 vs real ~$25–30)

**Symptom.** I told the user the full build would cost **~$5–6**. The **$5 balance depleted after
generation alone** (no judge stage run) — the estimate was low by roughly 5–6×, and the run
stopped before its most important quality step.

> This is an **estimation/process error**, not a code bug. It is included because it had real
> consequences (a depleted balance mid-build) and the mistake is worth recording.

## Why it arose

The smoke test measured **token counts precisely** (~533 tokens per kept pair, from
`usage_metadata`). But to turn tokens into dollars I applied **assumed price rates** —
`$0.10 / 1M input`, `$0.40 / 1M output` — that I had not verified against the billing console. The
real Gemini 3.1 Flash-Lite price is **~$1 / 1M tokens blended** (derived after the fact: the $5
balance bought ~5M tokens). My assumed rate was ~6× too low.

Two compounding mistakes:
1. I treated a **guessed rate as known** because the token measurement around it was exact — the
   precision of one input created false confidence in the other.
2. I let the new low figure **override my own earlier $20–30 estimate** (which had used a higher,
   more realistic blended rate) instead of treating the discrepancy as a flag to verify.

## How it was fixed

Corrected the guide (`DATASET-BUILD-GUIDE.md §9.2`) and project memory to the **measured ~$1/1M**
and a **~$25–30 full-build** figure, with an explicit instruction to verify current pricing in the
billing console before budgeting. The datasets were assembled **without** the judge (the stage the
budget could not reach) and flagged as such; a top-up (~$8–12) re-runs `build.py` to add the
correctness gate, resuming from cached verdicts.

## Alternatives considered (that would have prevented it)

- **Verify pricing** in the billing console / pricing page before quoting a number.
- **Quote a range** spanning the rate uncertainty rather than a point estimate.
- **Read the actual dollar delta** from billing after the ~$0.05 smoke test, instead of computing
  it from assumed rates.

## Why they were not done at the time

There was no good reason — this is the lesson. I over-anchored on the smoke test's exact token
counts and presented a single low number as if the rate were established. The correct move was to
either verify the rate first, or present the estimate explicitly as *tokens × unverified rate* and
carry the uncertainty forward (a range). The rule going forward:

> **Measured tokens × an unverified rate = an unverified cost.** State the assumption, quote a
> range, and confirm the rate against billing before treating a dollar figure as reliable.
