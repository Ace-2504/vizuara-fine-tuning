---
license: apache-2.0
base_model: thesreedath/slm-500m-base
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - legal
  - finance
  - question-answering
  - small-language-model
  - raft
  - retrieval-augmented
---

# slm-500m-raft

**500M · RAFT** — answers from the right document among distractors, and abstains when the answer is absent.

One of 13 models in a controlled study of how far a small language model can be pushed on US legal and financial text. Every version was trained on the same data, evaluated on the same frozen held-out set, and scored by the same blind LLM judge, so the stages are directly comparable. Compare them side by side in the [SLM Arena](https://slm-arena-harman.vercel.app).

Trained and aligned by **Harman Sandhu** (Vizuara AI Labs).

## At a glance

| | |
|---|---|
| Base model | [`thesreedath/slm-500m-base`](https://huggingface.co/thesreedath/slm-500m-base) |
| Method | full fine-tune |
| Parameters | 517.8M |
| Trainable parameters | 517.8M (all) |
| Training data | 10,000 retrieval-augmented examples (golden passage + distractors) |
| Schedule | 3 epochs · 1,875 |
| Compute | 59.6 min on L4 (Modal) |
| Cost | **$1.40** total lineage |

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Ace-2504/slm-500m-raft"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval()
tok = AutoTokenizer.from_pretrained(MODEL)

SYS = ('You are a precise legal and financial assistant. Answer clearly using the '
       'provided context; do not invent facts.')
user = 'Context:\n<passage>\n\nQuestion: <your question>'

# Custom chat scheme. NOTE: never prepend <|bos|> — it was left untrained.
prompt = f'<|system|>\n{SYS}<|eos|>\n<|user|>\n{user}<|eos|>\n<|assistant|>\n'
enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
eos = tok.convert_tokens_to_ids('<|eos|>')
out = model.generate(**enc, max_new_tokens=160, eos_token_id=eos,
                     pad_token_id=eos)   # pad and eos share an id here
print(tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True))
```

## Evaluation

**This checkpoint was not evaluated.** The published study evaluated the base, DPO and RLAIF legs of the 500M line; this intermediate checkpoint was trained and released for completeness but never scored, so no numbers are quoted here rather than borrowed from a sibling model.

## Architecture

| | |
|---|---|
| Class | LlamaForCausalLM |
| Layers | 24 |
| Hidden size | 1,280 |
| Attention | 20 heads · head dim 64 · full MHA |
| Feed-forward | SwiGLU · inner 3,456 |
| Positional | RoPE · θ 10,000 |
| Norm | RMSNorm · ε 1e-5 |
| Context | 1,024 tokens |
| Vocabulary | 32,768 |
| Embeddings | tied input/output |

## Training

- **Initialised from** [`thesreedath/slm-500m-base`](https://huggingface.co/thesreedath/slm-500m-base)
- **Method** — full fine-tune
- **Data** — 10,000 retrieval-augmented examples (golden passage + distractors), generated from a legal/financial corpus and gated by an LLM judge for faithfulness
- **Schedule** — 3 epochs · 1,875, 59.6 min on L4
- Loss is **masked to the answer tokens only**, so the model is never trained to reproduce the prompt

**Cost — $1.40.** 
The base was imported, so only this project's own work is counted: $1.40 of Modal GPU time and shared dataset generation.

## Limitations and intended use

- **Not legal or financial advice.** This is a research artefact for studying small-model training, not a professional tool. Do not rely on its output for real decisions.
- **At 517.8M it holds very little world knowledge.** It is built to read an answer out of a passage you supply, not to recall facts. Used closed-book it will produce fluent, confident and wrong text.
- Emits its evidence wrapped in `##begin_quote## … ##end_quote##` markers, and the sampler mangles these fairly often — strip or parse them before display.
- The judge behind these scores has **not been calibrated against human labels**, so treat small differences between models cautiously.
- English only; the corpus is US case law, SEC filings and educational web text.

## The rest of the family

| Size | Base | QA SFT | RAFT | DPO | RLAIF |
|---|---|---|---|---|---|
| 125M | [`slm-125m-e4`](https://huggingface.co/Ace-2504/slm-125m-e4) | `slm-125m-sft` | `slm-125m-raft` | `slm-125m-sft-dpo` | `slm-125m-sft-rlaif` |
| 500M | [`slm-500m-base`](https://huggingface.co/thesreedath/slm-500m-base) | `slm-500m-sft` | `slm-500m-raft` | `slm-500m-sft-dpo` | `slm-500m-sft-rlaif` |
| Gemma 2B | [`gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) | `gemma-2-2b-sft` | `gemma-2-2b-raft` | `gemma-2-2b-sft-dpo` | `gemma-2-2b-sft-rlaif` |

All under [`Ace-2504`](https://huggingface.co/Ace-2504) except the two imported bases.

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