# Fine-Tuning Guide — 500M model (QA SFT + RAFT)

**Audience:** a session fine-tuning `thesreedath/slm-500m-base` on the datasets in this
repository. Training on Modal; interactive serving on the local RTX 3060.

**Self-contained.** Two independent tracks — **Track A (QA SFT)** and **Track B (RAFT)**. A
thread told to do only one reads the preamble + that track and nothing else.

> **Datasets are already built** (`DATASET-BUILD-GUIDE.md`), shared unchanged with the 125M
> and Gemma guides. Do not regenerate them.

---

## Preamble — everything both tracks share

### P.1 The model

| | |
| --- | --- |
| Base | **`thesreedath/slm-500m-base`** (HF hub) |
| Params / layers / hidden | 517.8M · 24 · 1280 · 20 heads (head dim 64, MHA) |
| Context | **1,024** |
| Tokenizer | **its own, vocab 32,768** byte-level BPE — **NOT** `data/tokenizer/` (that's the 125M's 16k) |
| Special tokens | same 7-token scheme (`<\|bos\|>`…`<\|system\|>`); no chat template (base model) |
| Training mode | **Full fine-tune** (~8 GB weights+grads+Adam in bf16; fits L4 24 GB) |

**Load tokenizer + model from the hub**, not from `data/tokenizer/`:
```python
tok = AutoTokenizer.from_pretrained("thesreedath/slm-500m-base")
model = AutoModelForCausalLM.from_pretrained("thesreedath/slm-500m-base", torch_dtype="bfloat16")
```

⚠️ **Two model-specific gotchas — handle both before training:**

1. **`config.json` has `eos_token_id=2`, but the tokenizer puts `<|eos|>` at id 1** (`id 2 =
   <|pad|>`). This mismatch (an unchanged Llama default) means generation may stop on the wrong
   token or not stop. **Set it explicitly:**
   ```python
   eos_id = tok.convert_tokens_to_ids("<|eos|>")   # 1
   model.config.eos_token_id = eos_id
   model.generation_config.eos_token_id = eos_id
   ```
2. **`<|bos|>` is very likely untrained** (same from-scratch course convention as the 125M).
   **Verify, then decide:** encode a few training docs and check whether BOS was ever
   prepended in pretraining, or simply test generation with vs without a leading `<|bos|>`. If
   untrained, **omit BOS** from the template (default assumption below). Do not guess silently.

### P.2 The shared datasets

Text-level chat JSONL. Render with **this model's** 32k tokenizer + template.

| File | Track |
| --- | --- |
| `data/sft/qa_sft.jsonl` (15,000) | A |
| `data/sft/raft.jsonl` (10,000) | B |
| `data/sft/eval.jsonl` (500) | both |

The datasets are **text** — the 32k tokenizer will pack them into different token counts than
the 125M's 16k; that is expected and fine. Re-check context fit under this tokenizer (§Track B).

### P.3 Chat template — no new tokens; BOS per P.1.2

```jinja
{% for m in messages %}{{ '<|' + m['role'] + '|>\n' + m['content'] + '<|eos|>\n' }}{% endfor %}
{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}
```

The chat tokens exist in the vocab but (like the 125M) start untrained — they learn during SFT.

### P.4 Loss masking — mandatory
Loss **only on assistant tokens**; mask system+user with `-100`. Verify by decoding one batch.

### P.5 Hyperparameters

| | |
| --- | --- |
| LR | 2e-5, cosine → 2e-6 · warmup 5% (slightly lower than the 125M — larger model) |
| Epochs | 3 |
| Effective batch | 16–32 (**log it**); use grad-accum, VRAM is tighter than the 125M |
| Precision | bf16 · grad clip 1.0 · wd 0.0 · seed fixed & recorded |
| Init | from the base hub weights every run |

### P.6 Modal — training

```python
import modal
app = modal.App("sft-500m")
vol = modal.Volume.from_name("ft-500m", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.4.1","transformers==4.46.3","accelerate>=0.34","numpy>=1.26,<2.0"))

@app.function(image=image, gpu="L4", volumes={"/data": vol}, timeout=60*60*3)
def train(pairs, out_dir, **hp):
    ...   # standard HF Trainer or a plain loop; load base from the hub inside the function
```

L4 works; **A100-40GB roughly halves wall-clock** for ~2.5× the hourly rate — use it if time
matters. Upload `data/sft` to the volume. Windows: absolute paths. `vol.commit()` after
checkpoints. Pin versions. If VRAM is tight at batch 16, drop micro-batch and raise grad-accum.

### P.7 Sanity checks
Overfit 10 examples → loss ~0; decode a batch (template + masking); 10 real steps first. Log
the corrected loss under grad-accum.

### P.8 Evaluation on Modal, in the training job
Score the **untrained base** first (zero point), then the fine-tuned model, on the identical
instrument. **Persist per-item results.** Bootstrap 95% CI on every number; interval-includes-
zero ⇒ not resolved. Fix decoding and hold it constant.

**Forgetting — different from the 125M.** `data/tokens/val/` is the **125M's** tokenizer and is
**not valid here**. This model's tokenizer is different *and* it was pretrained on a different
corpus instance (`thesreedath/slm-pretraining-corpus`), so:

- Measure **bits-per-byte** (tokenizer-invariant — the model card recommends this) on a
  **held-out slice of this corpus's text**, quarantined from the SFT data, before vs after
  fine-tuning. Report per source.
- This is a **same-domain retention proxy**, not the model's exact pretraining val — state that
  caveat. (We do not have thesreedath's original val split.)

```python
def bits_per_byte(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        nll = model(ids, labels=ids).loss.item() * (ids.numel() - 1)   # nats total
    return nll / math.log(2) / len(text.encode("utf-8"))               # bits per UTF-8 byte
```

### P.9 Local serving — RTX 3060 12GB
517M in bf16 ≈ 1 GB — trivial on the 3060. Load tokenizer + model from the checkpoint, apply
the P.3 template, set `eos_token_id` per P.1.1, omit BOS per P.1.2.

### P.10 Expected time & cost. **1 GPU; no multi-GPU.**

| Track | GPU | Time | Cost |
| --- | --- | --- | --- |
| A · QA SFT | L4 | **~1.5 h** | ~$1.2 |
| B · RAFT (long seqs) | L4 | **~2.5 h** | ~$2.0 |
| (either) | A100-40GB | ~half the above | ~similar total |

---

## TRACK A — QA SFT

1. **Load** `data/sft/qa_sft.jsonl`; render with the P.3 template (this model's 32k
   tokenizer); mask per P.4; pad.
2. **Train** per P.5–P.7 on Modal.
3. **Evaluate** (P.8): token F1 / exact match, **fabrication rate**, format adherence, refusal
   correctness — base vs fine-tuned, each with a CI.
4. **Retention** per P.8 (bits-per-byte on held-out corpus text, per source).
5. **Report** with per-item JSON, sample generations incl. failures, hyperparameters, cost.

**Track-A done:** base scored → fine-tuned on same instrument → CIs → per-item persisted →
BPB retention → decoding constant → `eos_token_id` fix confirmed.

---

## TRACK B — RAFT

1. **Load** `data/sft/raft.jsonl` (assembled context + quote-first answer already inside each
   row); render with P.3; mask per P.4.
2. **Context fit under the 32k tokenizer:** `max_seq_len = 1024`. The 32k tokenizer packs the
   shared text into **fewer** tokens than the 125M's 16k, so fit is easier — but still **log
   truncation counts; require ≈ 0**. A dropped golden document silently poisons the example.
3. **Train** per P.5–P.7.
4. **Evaluate** (P.8) with the **four matched conditions** from `eval.jsonl` (clean / realistic
   / retrieval-failure / closed-book), each with a **paired** bootstrap CI:
   - **Grounding gap** = realistic − closed-book, **vs the base model's own gap**.
   - Distractor robustness gap = clean − realistic.
   - **Quote validity** and **quote precision** (golden vs distractor).
   - **Fabrication** and **false-abstention** rates, read together.
5. **Retention** per P.8 (bits-per-byte on held-out corpus text).

**Track-B done:** base scored in all four conditions → fine-tuned on same instrument → paired
CIs → grounding gap vs base → quote validity & precision → fabrication & false-abstention
together → truncation ≈ 0 → BPB retention → `eos_token_id` fix confirmed.
