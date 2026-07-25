# 13 — Modal workspace hit its spend limit mid-run

**Symptom.** With 9 of 13 model evaluations already committed, the next launch died immediately:

```
Error: Workspace ac-BKxFptO6bfepuaAEannJnt has exceeded its spend limit
SET1 EXIT=1
```

No more compute would run on the account — but the four Gemma evaluations still had to be produced.

## Why it arose

The account reached its configured Modal spend cap. This blocks **compute** (functions won't
schedule) but, crucially, **not storage reads** — the `ft-data` volume was still fully readable.

## How it was fixed

Migrated off Modal and finished **locally**:

1. Confirmed volume reads still worked, and **downloaded** the 9 finished results, the 4 Gemma
   adapters, and the eval/few-shot datasets from the (read-only) volume.
2. Stood up a **local CUDA env** (torch cu121 + transformers + peft) on the RTX 3060 and wrote
   `evaluations/eval_local.py` to run the 4 Gemma evals in bf16 — same output format as the Modal
   `eval.py`, so `judge_eval.py` / `eval_report.py` consumed it unchanged.
3. Ran the Gemini judge locally too (only needs the API key in `.env`, no GPU/Modal).

The whole downstream pipeline (judge → report → published site) then completed with **zero further
cloud cost**.

## Alternatives considered

- **Raise the spend limit / new Modal account.** The user could not add more accounts, and topping
  up wasn't chosen; local was free and already had the GPU.
- **Wait for the limit to reset.** Rejected: indefinite, and the local path also removed the future
  cloud dependency for this last stretch.

## Precautions added (so a local interruption is cheap)

`eval_local.py` checkpoints **every generation** to `_partial/<version>.jsonl` with `fsync`, skips
finished items/versions on restart, and writes finals atomically — so the multi-hour local run (and
two real power cuts during it) resumed with no lost work. See also the resumable judge cache.
