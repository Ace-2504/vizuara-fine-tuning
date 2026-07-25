# 12 — Gemma-2B eval kept blowing the 2-hour per-call timeout

**Symptom.** The set1/set2 evaluation ran fine for the small SLMs but the four Gemma-2B versions
repeatedly **hit the Modal function's 7,200 s (2 h) per-call timeout** and were cancelled:

```
[eval/gemma-2-2b-sft]  FAILED: Task's current input ... hit its timeout of 7200s
[eval/gemma-2-2b-raft] FAILED: ... hit its timeout of 7200s
```

## Why it arose

A cascade of compute problems, each fixed only to reveal the next:

1. **4-bit + eager is slow.** Gemma-2 requires `eager` attention (soft-capping), and 4-bit (QLoRA)
   inference adds dequant overhead. On an L4, ~500 sequential generations of 160 tokens already
   approach 2 h; the RAFT condition has **2,000** items — far over.
2. **A100 fallback → capacity shortage.** Routing Gemma to `A100-40GB` via `Function.with_options`
   fixed speed in principle, but the jobs then sat **queued**: *"Function 'evaluate' is waiting to be
   scheduled on a GPU_A100 worker … acquiring more capacity."* Only `base-gemma` got a slot.
3. **`use_cache=False`.** The harness forced `use_cache=False` for Gemma to dodge a hybrid-cache
   warning; that makes decoding recompute the whole context each step (O(n²)) — needless slowdown,
   since greedy output with the KV cache is identical.

## How it was fixed

Worked down to what actually had capacity and speed:

- Switched Gemma to **bf16** (≈5 GB, no dequant) — 2–3× faster than 4-bit.
- Moved Gemma to **L4 with a 6 h timeout** (L4 had capacity where A100 did not); bf16 makes L4 fast
  enough for the 2,000-item RAFT run.
- Later (after the account hit its spend limit — see bug 13) finished the four Gemma models
  **locally on an RTX 3060** with `use_cache=True` (identical greedy output, far faster than the
  forced-off path).

## Alternatives considered

- **Raise the L4 timeout and keep 4-bit.** Rejected: even at 6 h, 4-bit RAFT (2,000 × ~14 s) risks
  the wall; bf16 removes the risk.
- **Batched generation** (8–16/call) to cut wall-clock ~4×. Offered but declined in favour of the
  per-item-checkpointed sequential run, for maximum power-cut safety.
