# 06 — Aux tasks collapsed to a handful after dedup

**Symptom.** Even after the grounding fix (04) and the quarantine fix (05), aux tasks stayed tiny:
`summarize 542, extract 30, rewrite 71` — from ~11,700 aux pairs in the raw data.

## Why it arose

Deduplication (both the exact-hash pass and the 0.90-cosine embedding pass) used the **`question`**
field:

```python
emb = model.encode([r["question"] for r in out], ...)
```

For QA that is right — questions vary. But for the aux tasks the `"question"` field is a
**boilerplate instruction** the teacher was told to produce: "Summarize the following text",
"Extract the key fields into JSON", "Rewrite in plain English". Those instructions are
near-identical across thousands of examples, so the near-dup filter treated almost all
summaries as duplicates of each other and collapsed each aux task to a few survivors — even
though their **answers** (the actual training content) were all different.

## How it was fixed

Make the dedup signature **task-aware**: dedup aux tasks by their **answer** (which varies per
passage), and QA / unanswerable by their question. Applied to both the exact and embedding passes.

```python
def _dedup_text(r):
    if r["task"] in ("summarize", "extract", "rewrite"):
        return r["answer"]
    return r["question"]
```

Aux survival jumped from ~640 to the thousands, and the final task mix became
`qa 7702 / summarize 3283 / extract 2554 / rewrite 1461`.

## Alternatives considered

- **Dedup aux by `chunk_id` only** — each chunk yields one aux of a given task, and chunks are
  unique, so aux are inherently non-duplicate at the source.
- **Dedup by instruction + passage concatenated.**
- **Exempt aux from dedup entirely.**

## Why they were not chosen

- **`chunk_id`-only** relies on chunk uniqueness but would miss the real case it should catch:
  two *different* chunks that happen to yield near-identical summaries. Deduping by answer catches
  those; chunk-id does not.
- **Instruction + passage** lets the long passage text dominate the embedding, so near-dup
  detection keys on passage similarity rather than on the output — the thing we actually want to
  deduplicate.
- **Exempting aux** discards genuine near-duplicate removal on aux answers.

Deduping by **the field that actually varies (the answer)** is the correct signature — it removes
real near-duplicates without punishing examples for sharing a fixed instruction.
