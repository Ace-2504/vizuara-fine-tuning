# 20 — Each Gemma variant loaded a private copy of the same 2.6B base (~15 GB instead of 2.4 GB)

**Symptom.** The five Gemma entries dominated the GPU: eight small models (5 × 125M + 3 × 500M)
together used **4.12 GB**, while Gemma alone needed roughly **15 GB** — more than the card has.
This is what made "all 13 resident" impossible and what drove the crash in bug 19.

Measured per-variant cost before the fix:

| Entry | VRAM |
| --- | --- |
| `gemma-base` (bf16, unquantised) | ~4.4 GB |
| each of `gemma-qa` / `-raft` / `-dpo` / `-rlaif` (4-bit + LoRA) | ~2.2 GB |

## Why it arose

All five entries are the **same 2.6B weights**: `google/gemma-2-2b-it`, plus four rank-16 LoRA
adapters of ~20.8M parameters each (~40 MB). But the loader treated every catalogue entry as an
independent model, so each adapter rebuilt its own 4-bit copy of the base:

```python
base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, ...)
model = PeftModel.from_pretrained(base, d).eval()      # a fresh base per variant
```

Two separate wastes:

1. **Four redundant backbones.** ~2.2 GB each to serve adapters worth 40 MB — a ~55× overhead.
2. **`gemma-base` was never quantised.** It took the non-adapter branch, which loads bf16, so the
   *unmodified* base cost 4.4 GB while the *fine-tuned* ones cost 2.2 GB.

## How it was fixed

One shared 4-bit backbone; adapters attached to it and switched per request. `gemma-base` is
served from the same object with adapters disabled, which also quantises it for free.

```python
_shared_gemma = AutoModelForCausalLM.from_pretrained(GEMMA_BASE_ID, quantization_config=bnb, ...)
# first fine-tune wraps it, later ones just attach:
_shared_gemma = PeftModel.from_pretrained(_shared_gemma, d, adapter_name=mid)
_shared_gemma.load_adapter(d, adapter_name=other)
```

Two constraints made this non-trivial:

- **Entries must not cache the model object.** PEFT *re-wraps* the base on first adapter attach
  and injects LoRA layers **into the base modules** — a stale reference to the raw base would
  silently apply whichever adapter happened to be active. Entries store
  `{"shared_gemma": True, "adapter": mid}` and resolve `_shared_gemma` at generation time.
- **`set_adapter()` mutates shared state.** Selection happens inside the generation semaphore, so
  it cannot race another request. `gemma-base` uses `with model.disable_adapter():`.

The backbone is released only when no Gemma variant remains resident.

**Result:** Gemma's five entries went from ~15 GB to **2.4 GB** — each extra adapter costs
~0.08 GB. All 13 models now sit resident in **6.55 GB of 12 GB**, with zero evictions.

```
 8. 500m-rlaif  resident= 8  vram=4.12GB
 9. gemma-base  resident= 9  vram=6.24GB      <- shared backbone
13. gemma-rlaif resident=13  vram=6.55GB      <- +4 adapters, 0.08 GB each
```

**Correctness was verified, not assumed** — a broken adapter switch would silently return
identical text from all five. Asking the same question to each gave **5/5 distinct outputs**, with
`gemma-raft` still emitting its trained `##begin_quote##` format and `gemma-dpo` reproducing the
same phrasing it gave as a standalone private copy.

## Alternatives considered

- **Merge each adapter into its own base** (`merge_and_unload`). Simplest to reason about, but
  that is exactly the ~2.2 GB-per-variant layout being replaced.
- **Keep private copies and just evict aggressively.** Works, but every Gemma request then pays a
  6–9 s reload, which is most of the arena's sweep time.
- **Serve Gemma from a separate process.** Isolates crashes (see bug 19) but forfeits sharing —
  the copies would still be private.
