# 14 — Few-shot base prompts overflowed the 1,024-token context

**Symptom.** During the set1 run, the un-tuned base models logged, on every generation:

```
This is a friendly reminder - the current text generation call will exceed the model's
predefined maximum length (1024).
```

The base models' floor scores would then be corrupted, not just noisy.

## Why it arose

A fairness fix (see the harness upgrade) gives un-tuned **custom bases** (125M / 500M) a few
**in-context QA exemplars** so they can follow the grounded-QA format at all. But each exemplar is a
full `document + question → answer` (~290 tokens). Three of them (~870 tokens) plus the actual
`clean` prompt (~300 tokens) ≈ **1,170 tokens > 1,024**. These bases have a 1,024-token context, so
positions past 1,024 use **undefined RoPE frequencies** — the model attends over a region it was
never trained on, degrading exactly the score the few-shot was meant to make fair.

(Gemma's "base" is `gemma-2-2b-it`, already instruction-tuned with an 8k context, so it stays
zero-shot and is unaffected.)

## How it was fixed

Make `gen()` **fit the few-shot to the context**: build the prompt, tokenize it, and drop exemplars
one at a time until it fits within `max_position_embeddings − max_new_tokens`:

```python
k = len(shots)
while True:
    text = render(system, user, shots[:k])
    ids = tok(text, ...).input_ids
    if ids.shape[1] <= GEN_BUDGET or k == 0:
        break
    k -= 1                       # base few-shot overflowed -> drop an exemplar
```

Tuned models have no shots, so their path is unchanged. Fewer-but-complete exemplars beats
truncated ones.

## Alternatives considered

- **Truncate each exemplar document** to keep k=3. Rejected: a chopped exemplar is a worse teaching
  signal than one fewer complete exemplar.
- **Skip few-shot for 1,024-ctx bases.** Rejected: that's the original problem (bases score at the
  floor for format reasons, not capability).
