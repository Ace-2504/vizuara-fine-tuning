# 18 — LRU eviction during generation freed no VRAM and over-committed the GPU

**Symptom.** With `MAX_RESIDENT = 2`, concurrent requests to different models could push VRAM well
past what the pool claimed to hold. `/health` would report 2 resident models while three sets of
weights actually sat on the card. Under enough overlap this ends in a CUDA OOM — while the
bookkeeping insists everything is fine.

## Why it arose

`/generate` is a **synchronous `def`**, so FastAPI runs it in a threadpool: requests genuinely
execute in parallel. The pool released its lock before the slow part:

```python
def get_model(mid):
    with _lock:
        ...
        while len(_resident) >= MAX_RESIDENT:
            _evict_one()
        ...
        return _resident[mid]        # lock released here

# caller, OUTSIDE the lock:
model, tok, family = get_model(r.model_id)
out = model.generate(...)            # seconds long
```

and eviction assumed dropping its own references was enough to free the weights:

```python
def _evict_one():
    mid, (model, tok, fam) = _resident.popitem(last=False)
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
```

The sequence that breaks it:

1. Thread A checks out `125m-qa` and starts generating — **holding a live reference**.
2. Thread B wants `gemma-dpo`, finds the pool full, and evicts `125m-qa`.
3. `del` drops only *B's* references. A's reference keeps the refcount above zero, so the
   tensors are never collected and `empty_cache()` reclaims **nothing**.
4. B loads Gemma on top of memory that was supposed to be free.

The eviction was a bookkeeping operation being trusted as a memory operation.

## How it was fixed

Make "in use" a first-class fact and make eviction respect it.

- **`_inuse: dict[str, int]`** counts in-flight requests per model, incremented under the lock at
  checkout and decremented in a `finally` via an `acquire()` context manager.
- **`_evict_idle_one()`** walks the pool in LRU order and evicts the first model with
  `_inuse == 0`. A model being generated on is structurally un-evictable.
- If **every** resident model is busy, the caller waits on a `threading.Condition` (signalled on
  each release) and returns **503** on timeout, rather than loading anyway.
- Generation is serialised by a `Semaphore(1)`: there is one GPU, so overlapping generations only
  interleave and double peak VRAM.
- Eviction now logs VRAM before/after, so the accounting can never quietly lie again.

Proof it works — evictions now genuinely reclaim memory:

```
[evict] gemma-base    vram 7.08 -> 2.21 GB
[evict] gemma-rlaif   vram 2.49 -> 0.32 GB
```

Stress test (10 threads × 2 requests over 13 models, `MAX_RESIDENT=2`): **20/20 OK, 0 failures,
no leaked in-use counters**, VRAM back to the 2 resident models afterwards.

## Alternatives considered

- **Hold the lock through generation.** Correct but serialises everything behind one mutex
  including model loads; the semaphore + refcount split keeps loading and queueing separate.
- **Make the endpoint `async` and rely on the event loop.** Would remove the parallelism, but
  `model.generate()` is blocking CPU/GPU work — it would stall the whole loop instead.
- **Deep-copy nothing / just raise `MAX_RESIDENT`.** Hides the race rather than fixing it; it
  reappears the moment the pool is full again (see bug 19 for why the ceiling is real).
