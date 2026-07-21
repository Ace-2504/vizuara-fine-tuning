# 11 — "Golden-present-but-absent" RAFT case was topically mismatched

**Symptom.** In the RAFT "golden present but answer absent" examples, the provided golden passage
had nothing to do with the question — e.g. a passage about crucifixion art shown under the question
*"What reasons did BCM provide for resigning as advisor to the Trust?"* The model only has to
notice the topic is wrong, not that a specific fact is missing.

## Why it arose

This case is meant to teach the **subtle** abstention: the *right* document is present, is on the
*right topic*, but does not contain the specific answer — so the model must abstain rather than
guess. The assembly instead grafted an **unrelated** unanswerable question onto a **random QA
golden passage**:

```python
uq = unans[i % len(unans)]["question"]     # question from unanswerable pool
docs.insert(..., r["passage"])             # golden = an UNRELATED qa pair's passage
... Question: uq ... -> abstain
```

The unanswerable questions were written to be on-topic-but-unanswerable **for their own source
passage**, but were paired with a different passage entirely. Result: trivial, off-topic
abstention — a much weaker training signal than intended, and confusing examples in the set.

## How it was fixed

Build these examples from the unanswerable pool directly, using each unanswerable question's
**own** source passage as the golden. Now the golden is genuinely on-topic for the question, but
the specific answer is absent (which is exactly how the question was generated):

```python
for u in unans[:n_absent]:
    dd = random_distractors(C.RAFT_K, golden, u["doc_id"], u["passage"], RNG)
    docs = [golden[j]["passage"] for j in dd]
    docs.insert(RNG.randrange(len(docs) + 1), u["passage"])   # golden = the Q's own passage
    raft.append(chat(SYS_RAFT, f"{ctx}\n\nQuestion: {u['question']}", C.ABSTAIN_STRING,
                     {"golden_present": True, "answerable": False, ...}))
```

Verified: 800 answerable-absent examples, each with the question's own passage as golden.

## Alternatives considered

- **Drop the answerable-absent case entirely** — keep only golden-answerable + distractors-only.
- **Generate purpose-built pairs** — ask the teacher for a question answerable from *part* of a
  passage but not the shown span.
- **Keep the mismatch** — treat any absent-answer case as equivalent.

## Why they were not chosen

- **Dropping it** removes the most valuable abstention lesson: RAFT's whole point is refusing when
  a relevant-looking document does not actually contain the answer. Off-topic abstention
  (distractors-only) is the easy case; this is the hard one.
- **Purpose-built generation** is the ideal but costs teacher budget and a new recipe — and the
  `unanswerable` pool already contains exactly the right material (on-topic, answer-absent
  questions tied to a source passage). Reusing it is free and correct.
- **Keeping the mismatch** trains the wrong, trivial behavior (topic-matching instead of
  answer-presence), so no.
