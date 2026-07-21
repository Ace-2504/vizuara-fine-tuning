# Fine-Tuning Guide — 125M model (QA SFT + RAFT)

**Audience:** a session fine-tuning the 125M base model on the datasets in this repository.
Training on Modal; interactive serving on the local RTX 3060.

**Self-contained.** Two independent tracks — **Track A (QA SFT)** and **Track B (RAFT)**. A
thread told to do only one reads the preamble + that track and nothing else.

> **Datasets are already built** (see `DATASET-BUILD-GUIDE.md`) and are shared, unchanged,
> with the 500M and Gemma guides. Do not regenerate them.

---

## Preamble — everything both tracks share

### P.1 The model

| | |
| --- | --- |
| Base | the 125M Llama-style base (`Ace-2504/slm-125m-base`; **exact checkpoint TBD — confirm before running**) |
| Params / layers / hidden | 125.8M · 12 · 768 |
| Context | **1,024** |
| Tokenizer | `data/tokenizer/` — 16,384 byte-level BPE. **This is the corpus tokenizer** |
| Special tokens | `<\|bos\|> <\|eos\|> <\|pad\|> <\|unk\|> <\|user\|> <\|assistant\|> <\|system\|>` |
| Training mode | **Full fine-tune** (small enough; strongest behaviour change) |

⚠️ **`<|bos|>` and the chat tokens were never trained.** Pretraining used
`add_special_tokens=False` and appended only `<|eos|>`. So `<|bos|>`'s embedding is at random
init — **do not prepend it** — and `<|user|>/<|assistant|>/<|system|>` start from noise and
train during SFT. This model is the native match for `data/tokens/`.

### P.2 The shared datasets

Text-level chat JSONL (`messages` = system/user/assistant). Render with **this model's**
tokenizer + template at load time.

| File | Track |
| --- | --- |
| `data/sft/qa_sft.jsonl` (15,000) | A |
| `data/sft/raft.jsonl` (10,000) | B |
| `data/sft/eval.jsonl` (500) | both |

### P.3 Chat template — no new tokens, no BOS

```jinja
{% for m in messages %}{{ '<|' + m['role'] + '|>\n' + m['content'] + '<|eos|>\n' }}{% endfor %}
{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}
```

No `<|bos|>` (P.1). `<|eos|>` terminates every turn.

### P.4 Loss masking — mandatory
Compute loss **only on assistant tokens**; mask system+user with `-100`. Verify by decoding
one batch's unmasked positions — only answer text should appear. (Training on the prompt makes
the model generate questions instead of answering — a silent, common bug.)

### P.5 Hyperparameters (both tracks, unless a track overrides)

| | |
| --- | --- |
| LR | 3e-5, cosine → 3e-6 · warmup 5% |
| Epochs | 3 |
| Effective batch | 16–32 (micro-batch × grad-accum; **log it**) |
| Precision | bf16 · grad clip 1.0 · wd 0.0 · seed fixed & recorded |
| Init | from the **base** checkpoint every run — never from a previous fine-tune |

### P.6 Modal — training

```python
import modal
app = modal.App("sft-125m")
vol = modal.Volume.from_name("ft-125m", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.4.1","transformers==4.46.3","numpy>=1.26,<2.0")
         .add_local_python_source("train_core"))

@app.function(image=image, gpu="L4", volumes={"/data": vol}, timeout=60*60)
def train(pairs, out_dir, **hp):
    import train_core; return train_core.run(pairs_path=pairs, out_dir=out_dir, **hp)
```

Upload: `modal volume put ft-125m ./data/sft /sft --force` and the tokenizer + `data/tokens/val`.
⚠️ **Windows: absolute paths** (MSYS `/c/...` paths fail on the Windows Modal client).
Call `vol.commit()` after writing checkpoints. Pin image versions. Return metrics as a dict.

### P.7 Sanity checks before any full run
1. **Overfit 10 examples** → loss → ~0 in a few hundred steps, or masking/template/LR is wrong.
2. **Decode one batch** and read it — template correct, masking covers only the prompt.
3. **10 real steps** before the full job.
⚠️ Log the **corrected** loss under grad accumulation (divide by accum factor).

### P.8 Evaluation runs on Modal, in the training job
Score the **untrained base** on the identical instrument first — the zero point. Then the
fine-tuned model. **Persist per-item results** (id, output, reference, score) — not just
aggregates; without them you cannot compute intervals or re-analyse. Every number gets a
**bootstrap 95% CI**; a difference whose interval includes zero is "not resolved", not a trend.
Fix decoding (`no_repeat_ngram_size=3`, `repetition_penalty≈1.2`) and hold it constant across
base vs fine-tuned.

**Forgetting:** this model's benchmark is `data/tokens/val/` (same tokenizer). Report per
source (`sec`, `case-law`, `fineweb-edu`) vs the base — never an aggregate.

### P.9 Local serving — RTX 3060 12GB
125M is ~250 MB in fp16 — trivial locally.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("data/tokenizer")
m = AutoModelForCausalLM.from_pretrained("<ckpt>", torch_dtype=torch.bfloat16).to("cuda")
# generate with the P.3 template; do NOT prepend <|bos|>; stop on <|eos|>.
```

### P.10 Expected time & cost (L4)
Extrapolated from a measured run (4,000 pairs = 750 steps = 410 s). **1 GPU; no multi-GPU.**

| Track | Steps (≈) | Time | Cost |
| --- | --- | --- | --- |
| A · QA SFT (15k × 3ep) | ~2,800 | **~30 min** | ~$0.40 |
| B · RAFT (10k × 3ep, long seqs) | ~1,900 | **~50 min** | ~$0.70 |

---

## TRACK A — QA SFT

Teach instruction-following (answer / summarize / extract / rewrite) from `qa_sft.jsonl`.

1. **Load** `data/sft/qa_sft.jsonl`. Render each `messages` row with the P.3 template; mask
   per P.4. Pad to batch (prefer padding over packing for SFT).
2. **Train** per P.5–P.7 on Modal (P.6).
3. **Evaluate** (P.8), primary metrics:

| Metric | Why |
| --- | --- |
| Token F1 / exact match vs reference | Answer quality |
| **Fabrication rate** (number/entity absent from reference) | The predicted failure of a small model on document specifics — track from run 1 |
| Format adherence (bounded answer, not a continuation) | Did it learn to answer vs continue |
| Refusal correctness | If closed-book items are unanswerable |

4. **Forgetting** per P.8 on `data/tokens/val/`.
5. **Report** (per-item JSON attached): base vs fine-tuned with CIs, per-source forgetting,
   sample generations incl. failures, hyperparameters, cost.

**Track-A definition of done:** base scored → fine-tuned scored on the same instrument → every
metric with a CI → per-item persisted → per-source forgetting → decoding identical across both.

---

## TRACK B — RAFT

Teach grounded answering with distractors + abstention from `raft.jsonl`.

1. **Load** `data/sft/raft.jsonl`. Each row already contains the assembled context
   (documents + question) and a quote-first answer (or the abstention string). Render with the
   P.3 template; mask per P.4.
   ⚠️ **RAFT prompts are ~90% document tokens** — masking (P.4) matters even more here, or the
   model learns to continue documents.
2. **Sequence length:** `max_seq_len = 1024`. **Log truncation counts; they must be ≈ 0** —
   truncation that drops the golden document silently poisons an example. If non-zero, the
   dataset build's chunk size was too large for this context.
3. **Train** per P.5–P.7.
4. **Evaluate** (P.8) with the **four matched conditions** from `eval.jsonl`:

| Condition | Golden | Distractors | Expect |
| --- | --- | --- | --- |
| Clean | ✅ | none | highest |
| Realistic | ✅ | k | ≈ clean → distractor-robust |
| Retrieval failure | ❌ | k | high abstention, low fabrication |
| Closed-book | ❌ | none | low (control) |

Report, each with a **paired** bootstrap CI (matched questions):
- **Grounding gap** = realistic − closed-book, **vs the base model's own gap** (only movement
  above baseline is attributable to RAFT).
- Distractor robustness gap = clean − realistic.
- **Quote validity** (emitted quote is a verbatim substring of a provided doc) and **quote
  precision** (from the golden doc, not a distractor).
- **Fabrication** and **false-abstention** rates, read together.

5. **Forgetting** per P.8 on `data/tokens/val/`.

**Track-B definition of done:** base scored in all four conditions → fine-tuned scored on the
same instrument → paired CIs → grounding gap reported relative to base → quote validity &
precision → fabrication & false-abstention together → truncation ≈ 0 → per-source forgetting.
