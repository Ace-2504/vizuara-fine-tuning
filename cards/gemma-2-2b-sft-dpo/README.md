---
license: gemma
base_model: google/gemma-2-2b-it
language:
  - en
library_name: peft
pipeline_tag: text-generation
tags:
  - legal
  - finance
  - question-answering
  - small-language-model
  - peft
  - lora
  - dpo
---

# gemma-2-2b-sft-dpo

**Gemma 2B · DPO** — the QA adapter aligned with Direct Preference Optimization.

One of 13 models in a controlled study of how far a small language model can be pushed on US legal and financial text. Every version was trained on the same data, evaluated on the same frozen held-out set, and scored by the same blind LLM judge, so the stages are directly comparable. Compare them side by side in the [SLM Arena](https://slm-arena-harman.vercel.app).

Trained and aligned by **Harman Sandhu** (Vizuara AI Labs).

> **This is a LoRA adapter, not a standalone model.** It must be loaded on top of [`google/gemma-2-2b-it`]( https://huggingface.co/google/gemma-2-2b-it ), which is **gated** — you need to accept Google's licence on that repo before this adapter can be used. Usage of the adapter is governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## At a glance

| | |
|---|---|
| Base model | [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) |
| Method | QLoRA-DPO — rank-16 |
| Parameters | 2.6B (frozen base) |
| Trainable parameters | 20.8M LoRA (0.8%) |
| Training data | 500 AI-judged preference triplets |
| Schedule | 750 |
| Compute | 6.7 min on L4 (Modal) |
| Cost | **$3.53** total lineage |
| Judge score | **9.53 / 10** |

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "google/gemma-2-2b-it"      # gated — accept the licence first
ADAPTER = "Ace-2504/gemma-2-2b-sft-dpo"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, attn_implementation='eager',
    torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER).eval()
tok = AutoTokenizer.from_pretrained(BASE)

SYS = ('You are a precise legal and financial assistant. Answer clearly using the '
       'provided context; do not invent facts.')
user = 'Context:\n<passage>\n\nQuestion: <your question>'
text = tok.apply_chat_template([{'role': 'user', 'content': f'{SYS}\n\n{user}'}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors='pt').to(model.device)
# Gemma-2's hybrid cache misbehaves here — use_cache=False is required.
out = model.generate(**ids, max_new_tokens=160, use_cache=False)
print(tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True))
```

## Evaluation

Scored on **500 decontaminated held-out questions** (chunk-level dedup against public legal benchmarks, so no evaluation passage was trained on). Every model in the study answered the identical questions with deterministic greedy decoding, and a **Gemini judge blind to the model** graded each answer with the gold answer in hand.

| Metric | Value |
|---|---|
| Judge score (0–10 rubric) | **9.53** |
| Judged correctness (0–1) | 0.924 |
| Groundedness | 90.6% |
| Fabrication ↓ | 4.2% |
| Token-F1 | 0.149 |
| n | 500 |

By source:

| US case law | SEC filings | Educational web |
|---|---|---|
| 9.32 | 9.46 | 9.85 |

The 0–10 figure is a four-dimension rubric (correctness 0–5 + completeness 0–2 + groundedness 0–2 + clarity 0–1). The 0–1 figure is the stricter correctness-only scale used in the experiment reports. Same questions, same answers, same judge — different scale, so the two numbers differ.

**Token-F1 is reported for completeness and should not be read as quality**: it punishes correct paraphrase heavily, which is exactly why the judge carries the headline.

## Architecture

| | |
|---|---|
| Class | Gemma2ForCausalLM |
| Layers | 26 |
| Hidden size | 2,304 |
| Attention | 8 heads / 4 KV · head dim 256 · GQA |
| Feed-forward | GeGLU · inner 9,216 |
| Attention window | sliding 4,096 (alternating) · logit soft-capping |
| Norm | RMSNorm |
| Context | 8,192 tokens |
| Vocabulary | 256,128 |
| Embeddings | tied input/output |

## Training

- **Initialised from** [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it)
- **Method** — QLoRA-DPO — rank-16
- **Data** — 500 AI-judged preference triplets, generated from a legal/financial corpus and gated by an LLM judge for faithfulness
- **Schedule** — 750, 6.7 min on L4
- DPO against a **frozen reference copy** of the SFT checkpoint, β = 0.1

**Cost — $3.53.** 
The base is Google's and free to build on, so only this project's own work is counted: $3.53.

## Limitations and intended use

- **Not legal or financial advice.** This is a research artefact for studying small-model training, not a professional tool. Do not rely on its output for real decisions.
- The judge behind these scores has **not been calibrated against human labels**, so treat small differences between models cautiously.
- English only; the corpus is US case law, SEC filings and educational web text.

## The rest of the family

| Size | Base | QA SFT | RAFT | DPO | RLAIF |
|---|---|---|---|---|---|
| 125M | [`slm-125m-e4`](https://huggingface.co/Ace-2504/slm-125m-e4) | `slm-125m-sft` | `slm-125m-raft` | `slm-125m-sft-dpo` | `slm-125m-sft-rlaif` |
| 500M | [`slm-500m-base`](https://huggingface.co/thesreedath/slm-500m-base) | `slm-500m-sft` | `slm-500m-raft` | `slm-500m-sft-dpo` | `slm-500m-sft-rlaif` |
| Gemma 2B | [`gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) | `gemma-2-2b-sft` | `gemma-2-2b-raft` | `gemma-2-2b-sft-dpo` | `gemma-2-2b-sft-rlaif` |

All under [`Ace-2504`](https://huggingface.co/Ace-2504) except the two imported bases.

This model has its own write-up — training details, cost breakdown and live demo — at [https://slm-gemma-dpo-harman.vercel.app](https://slm-gemma-dpo-harman.vercel.app).

## Citation

```bibtex
@misc{sandhu2026slm,
  title  = {Small Language Models for Legal and Financial Text: a controlled study of
            pretraining, instruction tuning, retrieval augmentation and alignment},
  author = {Harman Sandhu},
  year   = {2026},
  note   = {https://slm-arena-harman.vercel.app}
}
```