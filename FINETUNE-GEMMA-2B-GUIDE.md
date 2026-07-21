# Fine-Tuning Guide — Gemma-2-2B-it with QLoRA (QA SFT + RAFT)

**Audience:** a session fine-tuning `google/gemma-2-2b-it` with QLoRA on the datasets in this
repository. Training on Modal; interactive serving on the local RTX 3060.

**Self-contained.** Two independent tracks — **Track A (QA SFT)** and **Track B (RAFT)**. A
thread told to do only one reads the preamble + that track and nothing else.

> **Datasets are already built** (`DATASET-BUILD-GUIDE.md`), shared unchanged with the 125M
> and 500M guides. Do not regenerate them.

---

## Preamble — everything both tracks share

### P.1 The model — read the differences first

`google/gemma-2-2b-it` is nothing like the two custom SLMs. Every following section reflects
these differences:

| | Gemma-2-2b-it | (vs the 125M/500M) |
| --- | --- | --- |
| Params / layers / hidden | 2.61B · 26 · 2304 | far larger |
| Context | **8,192** | vs 1,024 — not the binding constraint here |
| Tokenizer | Gemma SentencePiece, **256,128** vocab | vs custom 16k/32k BPE |
| Chat template | Gemma's own, `<start_of_turn>` — **no system role** | vs custom `<\|user\|>` |
| BOS | **`<bos>` IS trained — prepend it** | opposite of the custom models |
| Variant | **instruction-tuned (`-it`)** — already follows instructions | vs base completers |
| Training | **QLoRA** (4-bit base + adapters) | vs full fine-tune |

⚠️ **It is instruction-tuned, not a base model.** It already answers questions well, so its
**baseline scores will be high** — fine-tuning here is *domain adaptation*, not teaching
instruction-following. Two consequences: closed-book fabrication risk is lower, but you can
**degrade its general alignment**, so §P.8 measures that explicitly.

### P.2 Prerequisite — gated access
`google/gemma-2-2b-it` is **gated**. Before anything runs:
1. Accept the license at `https://huggingface.co/google/gemma-2-2b-it` (must be logged in).
2. Provide an `HF_TOKEN` (in `.env`, gitignored) with access, to both the local machine and
   the Modal image (`modal secret`), or the download 401s.

### P.3 The shared datasets — render for Gemma

Text-level chat JSONL (system/user/assistant). Use Gemma's **built-in chat template**, but
**fold the system message into the first user turn** — Gemma-2 has no system role.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")   # needs HF_TOKEN

def to_gemma(messages):
    sys = next((m["content"] for m in messages if m["role"] == "system"), "")
    conv = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user" and sys:
            conv.append({"role": "user", "content": f"{sys}\n\n{m['content']}"}); sys = ""
        else:
            conv.append({"role": m["role"].replace("assistant", "model"), "content": m["content"]})
    return conv   # -> tok.apply_chat_template(conv, tokenize=False)
```

| File | Track |
| --- | --- |
| `data/sft/qa_sft.jsonl` (15,000) | A |
| `data/sft/raft.jsonl` (10,000) | B |
| `data/sft/eval.jsonl` (500) | both |

Gemma's 256k vocab packs the shared text into **far fewer** tokens than the custom SLMs, so
context fit is never a problem here.

### P.4 Loss masking — only the model turn
Mask everything up to and including `<start_of_turn>model\n`; compute loss on the model
response + its closing `<end_of_turn>`. Use TRL's `DataCollatorForCompletionOnlyLM` with the
response template `<start_of_turn>model\n`, or mask manually. Verify by decoding one batch.

### P.5 QLoRA configuration

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it", quantization_config=bnb,
    attn_implementation="eager",        # Gemma-2 soft-capping — eager is the safe choice for training
    torch_dtype=torch.bfloat16)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()      # expect <1% trainable
```

⚠️ **`attn_implementation="eager"`** for Gemma-2 — its attention logit soft-capping is not
fully supported by flash-attention-2, and using it can silently hurt training.

### P.6 Hyperparameters (QLoRA)

| | |
| --- | --- |
| LR | **2e-4**, cosine → 2e-5 · warmup 3% (LoRA takes a higher LR than full FT) |
| Epochs | 2–3 (start 2 — an `-it` model adapts fast; watch for alignment regression) |
| Effective batch | 8–16 (grad-accum; **log it**) |
| Optimizer | `paged_adamw_8bit` |
| Precision | bf16 compute · grad clip 1.0 · seed fixed & recorded · gradient checkpointing on |
| Init | fresh adapters from the base every run |

### P.7 Modal — training

```python
import modal
app = modal.App("sft-gemma2b")
vol = modal.Volume.from_name("ft-gemma2b", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.4.1","transformers==4.46.3","peft>=0.13","trl>=0.11",
                      "bitsandbytes>=0.44","accelerate>=0.34","datasets>=2.20"))

@app.function(image=image, gpu="L4", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60*60*5)
def train(pairs, out_dir, **hp):
    ...   # save ONLY the adapter (model.save_pretrained) — a few hundred MB, not the 2B base
```

QLoRA-2B fits an **L4 (~8–10 GB used)**; A100-40GB is faster. Save **only the LoRA adapter**.
Windows: absolute paths. `vol.commit()` after saving. Pin versions.
*(Note: QLoRA-2B also fits your local RTX 3060 12GB, so local training is an option — but the
plan runs it on Modal for parity with the other two models.)*

### P.8 Evaluation on Modal, in the training job
Score **base `gemma-2-2b-it`** first (zero point — it will already be strong), then the
adapter-merged model, on the identical instrument. **Persist per-item results.** Bootstrap 95%
CI on every number. Fix decoding and hold it constant.

**"Forgetting" means something different here.** Gemma's pretraining data is proprietary and
QLoRA freezes the base, so this corpus's `data/tokens/val` is irrelevant. Instead measure
**general-capability regression**:
- **Held-out general-text bits-per-byte** (a small non-legal sample — general web / encyclopedic
  prose), base vs fine-tuned. QLoRA should barely move it; a large rise means the adapter is
  over-writing general ability.
- A handful of **general instruction probes** (unrelated to legal/financial) scored before vs
  after — catches alignment/format regression the BPB number misses.

State plainly: QLoRA's frozen base makes catastrophic forgetting unlikely by construction;
this check confirms it rather than expecting a big effect.

### P.9 Local serving — RTX 3060 12GB
Comfortable. Two options:
```python
# (a) 4-bit base + adapter (~2 GB VRAM):
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", quantization_config=bnb,
                                            attn_implementation="eager")
model = PeftModel.from_pretrained(base, "<adapter_dir>").eval()
# (b) merge to fp16 for speed (~5 GB VRAM):
#   model = PeftModel.from_pretrained(fp16_base, adapter).merge_and_unload()
```
Use `tok.apply_chat_template(..., add_generation_prompt=True)`; `<bos>` is added by the
template — keep it (unlike the custom models).

### P.10 Expected time & cost. **1 GPU; no multi-GPU.**

| Track | GPU | Time | Cost |
| --- | --- | --- | --- |
| A · QA SFT (QLoRA) | L4 | **~2.5 h** | ~$2 |
| B · RAFT (QLoRA, longer seqs) | L4 | **~3.5 h** | ~$3 |
| (either) | A100-40GB | ~40% faster | ~similar total |

---

## TRACK A — QA SFT

1. **Load** `data/sft/qa_sft.jsonl`; render each row via `to_gemma()` + Gemma chat template
   (P.3); mask to the model turn (P.4).
2. **Train** QLoRA per P.5–P.6 on Modal (P.7). Save the adapter only.
3. **Evaluate** (P.8): token F1 / exact match, **fabrication rate**, format adherence, refusal
   correctness — base vs fine-tuned, each with a CI. Expect a **high base**; the gain from
   fine-tuning will be smaller than for the custom models (it already answers well) — the
   interesting signal is domain grounding and reduced fabrication on legal/financial content.
4. **General-capability regression** per P.8.
5. **Report** with per-item JSON, sample generations, hyperparameters, adapter size, cost.

**Track-A done:** base scored → merged model scored on same instrument → CIs → per-item
persisted → general-capability regression checked → decoding constant → adapter saved.

---

## TRACK B — RAFT

1. **Load** `data/sft/raft.jsonl` (assembled context + quote-first answer already in each row);
   render via `to_gemma()` + template (P.3); mask to the model turn (P.4).
2. **Sequence length:** `max_seq_len = 1024` is ample (Gemma's tokenizer packs the shared
   1,024-budget text into far fewer tokens). Log truncation counts; they should be ~0.
3. **Train** QLoRA per P.5–P.6.
4. **Evaluate** (P.8) with the **four matched conditions** from `eval.jsonl` (clean / realistic
   / retrieval-failure / closed-book), each with a **paired** bootstrap CI:
   - **Grounding gap** = realistic − closed-book, **vs base Gemma's own gap**. An instruction-
     tuned model already grounds well, so the base gap is already sizeable — only movement
     *above* it is attributable to RAFT here.
   - Distractor robustness gap = clean − realistic.
   - **Quote validity** and **quote precision** (golden vs distractor). Note Gemma may answer
     correctly without emitting the `##begin_quote##` markers — score answer correctness and
     quote-emission separately.
   - **Fabrication** and **false-abstention** rates, read together.
5. **General-capability regression** per P.8.

**Track-B done:** base scored in all four conditions → merged model scored on same instrument →
paired CIs → grounding gap vs base → quote validity & precision → fabrication & false-
abstention together → truncation ~0 → general-capability regression → adapter saved.
