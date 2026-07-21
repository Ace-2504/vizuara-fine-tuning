# 05 — Eval holdout ate ~9,400 training pairs

**Symptom.** After the gauntlet left ~15,138 clean pairs, holding out 500 eval items dropped the
**training pool to 5,698** — a ~9,400-pair loss to quarantine that made the target dataset
unreachable and starved RAFT of golden passages.

## Why it arose

`carve_eval` selected 500 eval pairs, each from a distinct **`doc_id`**, and then training
excluded every pair sharing those documents:

```python
picked.append(r); quar.add(r["doc_id"])
...
train = [r for r in r4 if r["doc_id"] not in quar]
```

On this corpus a single document yields many pairs: with `MAX_CHUNKS_PER_DOC = 12` and
`QA_PER_CHUNK = 5`, one long SEC filing can produce dozens of pairs. The 500 eval documents were
disproportionately the **pair-dense** ones, so quarantining them removed ~19 pairs each — ~9,400
in total — for 500 eval items.

## How it was fixed

Quarantine at **`chunk_id`** level instead of `doc_id`. The eval passage (the exact 256-token
chunk) is never trained on, but sibling chunks of the same document — different windows with
different content — remain valid training data. Collateral fell from ~9,400 to ~500.

```python
if r["chunk_id"] in quar: continue
picked.append(r); quar.add(r["chunk_id"])
...
train = [r for r in r4 if r["chunk_id"] not in quar]
```

Train pool recovered to ~11,981 → ~19,376 (after the companion fixes 06/07), enough to hit the
15,000 / 10,000 targets.

## Alternatives considered

- **Keep doc-level and generate more data** to absorb the loss.
- **Pick eval docs that have few pairs**, minimizing collateral while staying doc-level.
- **Pair-level quarantine** — remove only the exact eval pair from training.

## Why they were not chosen

- **Generating more data** spends teacher budget to buy a strictness we do not need — and budget
  was already the binding constraint.
- **Selecting only sparse docs** for eval biases the eval set toward short, atypical documents,
  hurting how representative the benchmark is of the real (long, dense) legal/financial data.
- **Pair-level quarantine is too weak**: sibling pairs from the *same chunk* (identical passage)
  would leak the eval answer straight into training.

**Chunk-level is the right grain** — strict enough that the eval passage is never seen in
training, cheap enough that collateral is negligible. This is a deliberate relaxation from the
RAFT guide's doc-level ideal, justified by this corpus's chunk density and documented as such in
`build.py::carve_eval`.
