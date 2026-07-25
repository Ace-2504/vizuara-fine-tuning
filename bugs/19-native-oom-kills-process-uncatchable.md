# 19 — A 4-bit load OOM killed the server outright; `except OutOfMemoryError` never fired

**Symptom.** Pinning all 13 models (`MAX_RESIDENT=13`) walked VRAM up cleanly to **11.17 GB of
12 GB** with 10 models resident, then the whole server **vanished** while loading the 11th. The
client saw `ConnectionResetError: [WinError 10054]`. No Python traceback, no 500, no `[oom]` log
line — the process was simply gone, taking all 13 sites and the arena with it.

```
 9. gemma-base   -> 200  resident= 9  vram=8.99GB  / 12.0GB
10. gemma-qa     -> 200  resident=10  vram=11.17GB / 12.0GB
11. gemma-raft   -> [server killed]
```

## Why it arose

The OOM safety net was **reactive** — it assumed the failure would arrive as a catchable Python
exception:

```python
try:
    return _load_once(mid)
except torch.cuda.OutOfMemoryError:      # never reached
    while _evict_idle_one(): pass
    return _load_once(mid)
```

That holds for allocations made by PyTorch's caching allocator. It does **not** hold for loading
4-bit weights: quantisation happens inside **bitsandbytes' native CUDA code** during
`from_pretrained`. When that allocation fails, the extension aborts the process at the C level.
There is no Python frame to unwind into, so no `except` clause — however broad — can run.

The last log line confirms it died mid-load, not mid-Python:

```
[load] gemma-raft <- .../gemma-2-2b-raft
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
```

Compounding it, the estimate of what would fit was never checked against reality — the pool only
counted *models*, and a count says nothing about bytes.

## How it was fixed

Move the decision **before** the allocation. Admission control asks the driver how much memory is
actually free and compares it against an estimate for the incoming model plus headroom:

```python
need = _estimate_gb(mid) + HEADROOM_GB          # default 1.5 GB for activations / KV cache
if len(_resident) >= MAX_RESIDENT or _free_vram() < need:
    if _evict_idle_one():
        continue                                 # freed something, re-evaluate
    ... wait on the condition, else 503 ...
```

with `_free_vram()` from `torch.cuda.mem_get_info()` (driver-level, so it also accounts for other
processes on the card) and `_estimate_gb()` derived from on-disk weight size.

Re-running the exact sequence that had killed the server: **all 13 requests returned 200**, the
pool self-limited by evicting instead of dying. The reactive `except` was kept as a
belt-and-braces net for allocations that *do* raise.

## Alternatives considered

- **Keep only the reactive handler and lower `MAX_RESIDENT`.** Guesses at a safe number and still
  dies whenever the guess is wrong (e.g. another process takes VRAM).
- **`torch.cuda.set_per_process_memory_fraction()`.** Caps PyTorch's allocator, but bitsandbytes'
  native path is exactly what escapes it — the failure mode is unchanged.
- **Load on the CPU first and measure.** Doubles load time and RAM, and 4-bit quantisation is a
  GPU-side operation anyway.
- **Run each model in its own subprocess** so a crash is contained. Genuinely more robust, but
  13 processes × CUDA context (~300 MB each) wastes more VRAM than it protects, and it makes
  the shared-backbone optimisation (bug 20) impossible.

## Lesson

`try/except` protects you from *your* allocator, not from a native extension's. Where a library
can abort the process, admission must be decided in advance — a check after the fact is a check
that never runs.
