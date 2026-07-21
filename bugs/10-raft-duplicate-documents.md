# 10 — Duplicate documents inside RAFT prompts (15.3%)

**Symptom.** Inspecting the generated RAFT set, **1,531 of 10,000 examples (15.3%)** contained two
identical document texts in the same prompt — e.g. `[Document 2]` and `[Document 3]` byte-for-byte
the same.

## Why it arose

`pick_distractors` selected the top-k most **question-embedding-similar** chunks as distractors,
excluding only the golden's `doc_id`:

```python
def pick_distractors(i, emb, pool, k, golden_doc):
    sims = emb @ emb[i]; sims[i] = -1
    for j in np.argsort(-sims):
        if pool[j]["doc_id"] == golden_doc: continue
        out.append(j) ...
```

This corpus contains near-duplicate passages that survived corpus-level dedup as **distinct
`chunk_id`s from distinct `doc_id`s** (e.g. boilerplate SEC language, repeated case citations).
"Most similar" therefore routinely surfaced two chunks with identical or near-identical text, and
nothing checked the document **text** — only the `doc_id`. A distractor that duplicates the golden
(or another distractor) is not a distractor: it teaches nothing and shrinks the effective context.

## How it was fixed

Deduplicate documents by **text** during selection. A shared `_distinct_take` helper skips any
candidate whose normalized passage already appears (seeded with the golden's text), used by both
the similarity-based and random distractor pickers:

```python
def _distinct_take(order, pool, k, exclude_doc, seen_texts):
    for j in order:
        if pool[int(j)]["doc_id"] == exclude_doc: continue
        t = norm(pool[int(j)]["passage"])
        if t in seen_texts: continue
        seen_texts.add(t); out.append(int(j)) ...
```

Applied to golden-present, distractors-only, and the eval distractor selection. Verified:
duplicate-document rate **15.3% → 0.0%**.

## Alternatives considered

- **Embedding-threshold dedup** — skip candidates with cosine ≥ 0.95 to the golden or a picked doc.
- **Fix it upstream** — a stricter corpus-level near-dup pass so no near-duplicate chunks exist.
- **Accept it** — 15% duplicates as tolerable noise.

## Why they were not chosen

- **Embedding-threshold** would also catch *near*-duplicates, but the observed 15.3% were **exact**
  text duplicates, which text-equality removes completely and deterministically; a cosine
  threshold adds a tunable magic number and false-positive risk for no measured gain here. (If a
  near-dup problem shows up later, the same `seen_texts` hook can carry an embedding check.)
- **Upstream corpus dedup** is the "right" long-term fix but re-tokenizes/rebuilds the 2.5B corpus
  — huge cost to fix a defect that belongs to *distractor selection*, not the corpus.
- **Accepting it** corrupts the core RAFT signal (distractors must differ from the golden), so no.
