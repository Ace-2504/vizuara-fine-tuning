# 04 — Summaries and rewrites deleted as "ungrounded"

**Symptom.** The first full assembly produced a badly skewed task mix: `qa 3834, summarize 254,
rewrite 26, extract 1`. The aux tasks had almost disappeared, despite ~11,700 aux pairs in the
raw data.

## Why it arose

`build.py::grounded_ok` required at least **0.55 content-word overlap between the answer and the
passage**, and applied it to **every** task:

```python
def grounded_ok(r):
    ...
    return len(answer_words & passage_words) / len(answer_words) >= 0.55
```

That test is correct for **extractive QA** (the answer is a span from the passage). It is *wrong*
for the aux tasks:

- a faithful **summary** compresses and rephrases — it deliberately shares far less than 55% of
  its words with the source;
- a plain-English **rewrite** changes the register on purpose;
- **extraction** emits JSON field names, not passage prose.

So legitimate, faithful aux examples were scored as hallucinations and dropped.

## How it was fixed

Restrict the overlap test to `task == "qa"`; everything else passes this stage (its faithfulness
is a *different* property, verified by the LLM judge when budget allows):

```python
def grounded_ok(r):
    if r["task"] != "qa":
        return True
    ...
    return len(aw & pw) / len(aw) >= C.GROUNDING_MIN
```

## Alternatives considered

- **Lower the 0.55 threshold globally** so aux survives.
- **NLI / entailment model** to check aux faithfulness against the passage.
- **Embedding similarity** (answer vs passage) instead of word overlap.

## Why they were not chosen

- **Lowering the threshold** weakens the QA grounding guarantee — the whole reason the check
  exists is to reject QA answers that drift from the passage. Loosening it to admit summaries
  would let ungrounded QA through too.
- **NLI** means loading and running another model on every pair — real cost and latency for what
  is meant to be a cheap rule filter, and redundant with the LLM judge that already checks
  faithfulness.
- **Embedding similarity** thresholds are hard to calibrate (a good summary is semantically close
  but lexically distant) and would still penalize legitimate compression.

The task type already carries the right signal: **only QA is extractive, so only QA gets the
extractive check.** Faithfulness for the paraphrastic tasks is the judge's job.
