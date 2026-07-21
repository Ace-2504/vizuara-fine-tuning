# Fine-tuning — 500M & Gemma-2B, SFT + RAFT

Produces **two versions per model** — an SFT version and a RAFT version — as **independent
fine-tunes from the base checkpoint**. One script per model, run twice with `--method`:

```
500M  base ──┬─ modal run train_500m.py  --method sft   → /data/checkpoints/slm-500m-sft
             └─ modal run train_500m.py  --method raft  → /data/checkpoints/slm-500m-raft

Gemma base ──┬─ modal run train_gemma.py --method sft   → /data/checkpoints/gemma-2-2b-sft   (adapter)
             └─ modal run train_gemma.py --method raft  → /data/checkpoints/gemma-2-2b-raft  (adapter)
```

`--method sft` trains on `qa_sft.jsonl`; `--method raft` trains on `raft.jsonl`. Everything
else (base, tokenizer, template, hyperparameters) is identical, so the two versions are
directly comparable. **4 checkpoints from 2 scripts.**

## Files

| File | What |
| --- | --- |
| `ft_config.py` | all constants (bases, hyperparameters, GPUs, paths) |
| `ft_data.py` | chat rendering + **loss masking** (assistant-only); one renderer per model |
| `train_500m.py` | **full fine-tune** of `thesreedath/slm-500m-base` |
| `train_gemma.py` | **QLoRA** of `google/gemma-2-2b-it` (saves adapter only) |

## Model-specific handling (baked in)

| | 500M | Gemma-2B |
| --- | --- | --- |
| Training | full FT | QLoRA (4-bit nf4 + LoRA r16) |
| Tokenizer / template | own 32k BPE, `<\|role\|>` scheme, **no BOS** | Gemma 256k, `<start_of_turn>`, system→user, **BOS kept** |
| Special fix | `eos_token_id` corrected to `<\|eos\|>` (config.json is wrong) | `attn_implementation="eager"` (soft-capping) |
| Effective batch | 16 (8×2) | 16 (4×4) |

Validated locally on the real datasets: 500M SFT median 235 tok, RAFT median 861 / max 936 —
**RAFT fits in the 1024 context with zero truncation.**

## Prerequisites before running

1. **Upload the datasets to the Modal volume** `ft-data`:
   ```bash
   modal volume create ft-data
   modal volume put ft-data ../data/sft/qa_sft.jsonl /sft/qa_sft.jsonl --force
   modal volume put ft-data ../data/sft/raft.jsonl   /sft/raft.jsonl   --force
   modal volume put ft-data ../data/sft/eval.jsonl   /sft/eval.jsonl   --force
   ```
   (Windows: use absolute paths.)
2. **Create the Modal secret `hf-token`** with a valid `HF_TOKEN` (Gemma is gated; the licence
   must already be accepted on HuggingFace):
   ```bash
   modal secret create hf-token HF_TOKEN=hf_xxx
   ```
   The 500M is public and does not need it, but the scripts pass it through harmlessly.

## Run order (recommended)

1. **Smoke test** each script on a tiny slice first (add a `--limit` or run 1 epoch on a few
   hundred rows) to confirm it trains and saves before a full run.
2. 500M SFT → 500M RAFT → Gemma SFT → Gemma RAFT. Independent; can run in parallel.

## Not yet included

- **Post-training evaluation** (deterministic F1 / copy-rate / RAFT four-condition / forgetting)
  is described in the per-model guides and is the next module to add; these scripts currently
  train + save. Base-vs-fine-tuned scoring runs after the checkpoints exist.
