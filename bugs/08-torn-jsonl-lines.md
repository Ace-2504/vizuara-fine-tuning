# 08 — Corrupt lines in raw.jsonl crashed the reader

**Symptom.** Counting the generated data crashed:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

A robust scan found **27 blank/partial lines out of 33,351** (0.08%).

## Why it arose

Generation writes from 8 worker threads under a lock and flushes per record. But the process was
**hard-killed by the budget wall** (a 429 "prepayment credits depleted" propagated as
`BudgetExhausted` while other threads were mid-write). A few lines were flushed partially or left
blank at the moment of shutdown. This is normal for an append-only log that can be truncated at
any instant — the failure was that the *reader* used a list comprehension that aborts on the first
bad line:

```python
raw = [json.loads(l) for l in open(C.RAW, encoding="utf-8")]   # dies on line 1 bad char
```

## How it was fixed

A tolerant loader that strips, skips blanks, and try/except-skips unparseable lines, reporting the
count:

```python
def load_raw():
    raw, bad = [], 0
    for l in open(C.RAW, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        try:
            raw.append(json.loads(l))
        except Exception:
            bad += 1
    if bad:
        print(f"  (skipped {bad} torn/blank lines)")
    return raw
```

27 lost pairs out of 33,324 valid ones is immaterial to the dataset.

## Alternatives considered

- **Prevent torn writes** with per-line `fsync` and stricter locking.
- **Per-worker output files**, merged at the end.
- **A separate validate-and-repair step** over `raw.jsonl`.

## Why they were not chosen

- **You cannot fully prevent a partial write on a hard kill** — the OS can truncate a buffered
  write at any point — and per-line `fsync` would badly slow a 30k+ call run for a guarantee it
  still could not make.
- **Per-worker files** complicate the resumability model (single append-only log + one state
  file) and the "appended the instant it is produced" guarantee that makes a mid-run credit-out
  safe.
- **A repair step** is extra machinery for a 0.08% loss.

**Tolerant reading is the standard contract for append-only logs that can be truncated
mid-write.** The writer stays simple and fast; the reader absorbs the rare torn line.
