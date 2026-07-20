# QA Supervised Fine-Tuning — Implementation Guide for This Corpus

**Audience:** an engineering session implementing QA SFT using the 2.5B-token legal/financial
corpus in this repository, on Modal.

**Self-contained.** Everything needed is here. No other document is required.

---

## 0. What this guide produces

A base model (a text *completer*) turned into a model that answers questions in a chat
format, trained on synthetic `question → answer` pairs distilled from a teacher LLM using
this repository's corpus as the source material.

**QA SFT is closed-book.** The training example contains a question and an answer; the
source passage is **not** in the prompt. The model must answer from knowledge in its weights.

```
System:    You are a precise legal and financial assistant.
User:      What does Regulation S-K require an issuer to disclose about executive compensation?
Assistant: Regulation S-K Item 402 requires ...
```

### 0.1 ⚠️ Read this before committing — it decides whether the method fits

QA SFT teaches **behaviour** (answer format, tone, instruction-following, refusal), not
**knowledge**. A single exposure to a fact during fine-tuning does not reliably install it in
the weights, particularly in smaller models.

**This corpus is document-heavy** — SEC filings and US case law. That is exactly the material
where closed-book QA SFT most reliably fails. A question like *"What were Cummins Engine's
net revenues in fiscal 1997?"* asks the model to recall a long-tail fact it saw **once**. It
cannot. What it learns instead is the *shape* of an answer, and it will produce a fluent,
confident, wrong number.

**Therefore, when generating from this corpus:**

| Do | Don't |
| --- | --- |
| Generate closed-book questions from **`fineweb-edu`** — general educational knowledge, likely reinforced across many documents | Generate closed-book questions about specific figures in a single SEC filing |
| Ask about **general legal/financial concepts** ("what is a motion to dismiss") | Ask about a specific case's holding, docket number, or date |
| Use QA SFT for **format, tone, and refusal behaviour** | Expect factual recall of document specifics |

If the goal is accuracy on specific source documents, closed-book SFT is the wrong method —
the passage needs to be in the prompt, which is a different technique.

---

## 1. The dataset in this repository

Everything below already exists on disk. **Do not re-clean, re-deduplicate, or
re-decontaminate it** — those phases are complete, and re-running them wastes hours and can
only degrade the data.

### 1.1 Layout

```
fine-tuning-dataset/
  data/
    corpus/                    # 11 GB — SOURCE PASSAGES for teacher generation
      case-law/    shard-000.txt … shard-009.txt
      sec/         shard-000.txt … shard-004.txt
      fineweb-edu/ shard-000.txt … shard-009.txt
    tokenizer/                 # 16,384-token byte-level BPE
    tokens/
      train/                   # 4.6 GB — pretraining bins (uint16), for REPLAY
      val/                     # held-out bins — the FORGETTING benchmark
      index.json               # token counts per shard
```

### 1.2 Corpus format — important

**One document per line, UTF-8, plain text.** Internal newlines were stripped during
cleaning. There is no JSON wrapper and no separator token.

```python
def iter_docs(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            text = line.rstrip("\n")
            if text:
                yield text
```

### 1.3 Contents

| Source | Shards | Documents | Size | Character |
| --- | --- | --- | --- | --- |
| `case-law` | 10 | 241,356 | 3.6 GB | US court opinions (HFforLegal/case-law) |
| `sec` | 5 | 45,035 | 4.1 GB | SEC filings (PleIAs/SEC) — long documents |
| `fineweb-edu` | 10 | 690,923 | 3.1 GB | Educational web text (HuggingFaceFW/fineweb-edu) |
| **Total** | **25** | **977,314** | **10.8 GB** | |

Note the asymmetry: SEC has the fewest documents but the most bytes — filings are very long.
Chunking equalizes this; document counts are not a proxy for passage counts.

### 1.4 Tokenized bins

| | Tokens | Windows |
| --- | --- | --- |
| Train (`data/tokens/train/`) | 2,475,090,944 | 2,417,081 |
| Val (`data/tokens/val/`) | 25,006,080 | 24,420 |
| **Total** | **2,500,097,024** | |

Realized composition (train): **sec 34.7%** (860,036,096) · **case-law 34.2%** (846,758,912)
· **fineweb-edu 31.0%** (768,295,936). Format: `uint16`, `seq_len = 1024`, 14 shards, 99/1
train/val split.

**The val bins are the forgetting benchmark** (§7.4). Nothing has trained on them.

### 1.5 Tokenizer

`data/tokenizer/` — 16,384-token byte-level BPE. Special tokens already defined:

```
<|bos|>  <|eos|>  <|pad|>  <|unk|>  <|user|>  <|assistant|>  <|system|>
```

⚠️ **The chat tokens exist in the vocabulary but were never seen during pretraining.** The
tokenization pipeline used `add_special_tokens=False` and appended only `<|eos|>`.

**Consequences, both of which matter:**

1. **`<|bos|>` was never trained.** Its embedding is still at random initialization.
   **Do not prepend it.** Prepending an untrained vector to every example injects noise into
   every forward pass. Use `<|eos|>` as the only structural token, matching pretraining.
2. **`<|user|>`, `<|assistant|>`, `<|system|>` embeddings are also at initialization.** They
   will train during SFT, but they start from noise — expect the first few hundred steps to
   be spent learning them.

### 1.6 What has already been done to this corpus

| Stage | Status | Detail |
| --- | --- | --- |
| Cleaning | ✅ Complete | Rule chain + document-level OCR gate (non-dictionary rate) |
| Exact dedup | ✅ Complete | sec −1,989 · fineweb-edu −120 |
| Near-dup (MinHash) | ✅ Complete | case-law −1,643 |
| Decontamination | ✅ Complete | 13-gram vs **LexGLUE** and **CaseHOLD**; case-law −28,274 · sec −175 |
| Tokenization | ✅ Complete | 1,024-token windows, 99/1 split |

⚠️ **Decontamination covers LexGLUE and CaseHOLD only.** If the fine-tuned model will be
evaluated on any other benchmark, decontaminate against it before training — this cannot be
fixed afterward.

⚠️ **Corpus-level dedup does not remove duplicate *questions*.** The teacher will generate
near-identical questions from different passages. QA-pair deduplication (§4.3) is still
required.

---

## 2. Questions to answer before you start

The corpus questions are answered above. These remain, and depend on the model:

| # | Question | Guidance |
| --- | --- | --- |
| 1 | **Which base model** — HF repo or checkpoint path, parameter count? | Must use the tokenizer in `data/tokenizer/` |
| 2 | **Was it pretrained on this tokenizer?** | If not, `data/tokens/` and the forgetting benchmark do not apply, and passage chunking must use the model's own tokenizer |
| 3 | **Context length?** | This corpus was packed at 1,024. A model with a larger context can use larger passages |
| 4 | **How many QA pairs?** | §2.1 |
| 5 | **Which Modal GPU?** | §2.2 |
| 6 | **Full fine-tune or LoRA?** | §2.3 |
| 7 | **Teacher model, and is billing enabled on the key?** | Rate limits dominate wall-clock |
| 8 | **Domain mix for the QA pairs?** | §3.2 — mirror the corpus, or weight to the target application |
| 9 | **Any benchmark beyond LexGLUE/CaseHOLD?** | Requires extra decontamination (§1.6) |
| 10 | **Is replay wanted?** | `data/tokens/train/` is present for this (§7.4) |
| 11 | **Budget and wall-clock ceilings?** | Teacher generation is the bottleneck, not GPU |

### 2.1 How many QA pairs

| Base model size | Starting point |
| --- | --- |
| ≤ 300M | 3,000 – 10,000 |
| 300M – 1B | 10,000 – 30,000 |
| 1B – 3B | 20,000 – 50,000 |
| 7B+ | 30,000 – 100,000+ |

Small models saturate early; beyond a point extra pairs add forgetting rather than
capability. Prefer fewer, more diverse, higher-quality pairs. **If a scaling study is
wanted, generate the full set once and subsample** — regenerating per condition makes
dataset composition a confound.

With 977,314 documents this corpus can support any of these; chunk supply is not a
constraint.

### 2.2 Modal GPU

Approximate rates (verify current pricing — these move):

| GPU | ~$/hr | Suits |
| --- | --- | --- |
| T4 | ~$0.60 | ≤150M full FT; inference |
| **L4** | **~$0.80** | **≤1B full FT — best value for small models** |
| A10G | ~$1.10 | ≤1.5B full FT |
| A100-40GB | ~$2.10 | 1–3B full FT; 7B LoRA |
| A100-80GB | ~$2.50–3.00 | 3–7B full FT; 13B LoRA |
| H100 | ~$4.00–5.00 | 7B+ full FT |

SFT jobs are short and **overhead-bound**, not compute-bound — container startup (30–90s)
and checkpoint I/O often dominate. Do not reach for a bigger GPU to save minutes of compute.

Full-FT VRAM in bf16 with AdamW ≈ **16 bytes/parameter** plus activations.

### 2.3 Full fine-tune vs LoRA

| | Full FT | LoRA / QLoRA |
| --- | --- | --- |
| Best for | ≤3B | 7B+, or limited VRAM |
| Forgetting | Higher | Lower (base frozen) |
| Behaviour change | Stronger | Weaker at equal data |

Above ~7B use LoRA (r=16–64, alpha=2r, dropout 0.05, attention + MLP projections).

### 2.4 Hyperparameter starting points

| Parameter | Value |
| --- | --- |
| Learning rate | **1e-5 – 3e-5** full FT; **1e-4 – 2e-4** LoRA |
| Epochs | **2–3** (more overfits synthetic data hard) |
| Effective batch size | 16–64 sequences |
| Schedule | Cosine to ~10% of peak |
| Warmup | 3–5% of total steps |
| Weight decay | 0.0 |
| Grad clip | 1.0 |
| Precision | bf16 |
| Seed | Fixed and recorded |

**Log the effective batch size explicitly** (micro-batch × grad-accum × world size). Mixing
this up silently changes LR-per-token and is a common cause of irreproducible runs.

---

## 3. Phase 1 — Chunking

**The corpus is already clean. This phase only splits documents into passages.**

### 3.1 Chunker

Chunk with **the tokenizer in `data/tokenizer/`** — token budgets are what matter downstream,
not character counts.

```python
import hashlib, os
from dataclasses import dataclass
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("data/tokenizer")
CHUNK_TOKENS = 500          # target passage size

@dataclass
class Chunk:
    chunk_id: str; text: str; token_len: int; source: str; shard: str; line_no: int

def chunks_from_shard(source: str, shard: str):
    path = os.path.join("data/corpus", source, shard)
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            text = line.rstrip("\n")
            if not text:
                continue
            ids = TOK(text, add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), CHUNK_TOKENS):
                piece = ids[i:i + CHUNK_TOKENS]
                if len(piece) < 100:            # drop runt tails
                    continue
                body = TOK.decode(piece)
                yield Chunk(
                    chunk_id=hashlib.sha256(body.encode()).hexdigest()[:16],
                    text=body, token_len=len(piece),
                    source=source, shard=shard, line_no=line_no,
                )
```

Notes:
- **`chunk_id` is a content hash, never an index.** It is what makes generation idempotent
  and eval quarantine possible; indices shift.
- **SEC documents are very long** and will produce many chunks each. Cap chunks per document
  (e.g. 20) or a handful of filings will dominate the dataset.
- Prefer sentence boundaries where cheap; never split mid-word.
- **Persist a `chunks_used` set.** A chunk must never be used twice — reuse silently
  multiplies its effective epoch count.

### 3.2 Domain mix

Two defensible choices:

- **Mirror the corpus** — sec 34.7 / case-law 34.2 / fineweb-edu 31.0. Keeps the fine-tune
  in-distribution.
- **Weight to the application.** Better task performance, more forgetting elsewhere.

⚠️ **Per §0.1, restrict closed-book QA to `fineweb-edu`.** Closed-book questions drawn from
`sec` or `case-law` ask for single-exposure long-tail recall and will train fabrication.

Record the realized mix in the run report and hold it constant across any comparison.

---

## 4. Phase 2 — Teacher generation

### 4.1 Teacher client

Wrap the provider behind one interface so it can be swapped:

```python
class TeacherClient:
    def generate_json(self, prompt: str, schema, *, temperature: float = 0.9) -> dict | list:
        """Returns parsed JSON. Retries on 429/5xx with exponential backoff."""
```

**Robustness requirements — a multi-hour run must survive any single bad response:**

- **Enforce structured output** (response schema / JSON mode).
- **Handle HTTP 200 with an empty body.** This happens. `json.loads(None)` raises and kills
  the run.
- **Handle unparseable JSON** even when a schema was requested.
- **Retry, then skip.** After N retries, log and continue. Never abort.
- **Rate-limit deliberately.** Confirm whether the key is on a paid tier before estimating
  wall-clock — free tiers are dramatically slower.
- **Track usage** for the cost report.

### 4.2 Resumability

- Append each record to JSONL **as produced** — never buffer.
- Persist a state file with counts per category.
- On restart, count what exists and generate only the remainder.
- Make resume the **default** path so it is exercised every run and cannot rot.

### 4.3 Prompt rules

Request ~5 pairs per passage to amortize cost. Put these rules in the teacher prompt — each
prevents a specific downstream failure:

- **Questions must be self-contained.** No "according to the passage", "the document",
  "above". In closed-book training the passage is absent; a question referring to it is
  unanswerable and teaches hallucination of a referent.
- **Answers must not reference the source either.**
- **Questions must carry their own context** — name the company, case, or topic.
- **Vary difficulty**; no invented facts; no near-duplicates within a call.

### 4.4 Record format

Keep the passage even though closed-book training discards it — needed for the grounding
filter (§5.2) and diagnostics.

```json
{"chunk_id": "...", "source": "sec", "question": "...", "answer": "...",
 "difficulty": "medium", "passage": "...", "teacher": "model-name", "batch": 1}
```

---

## 5. Phase 3 — Filtering and the frozen holdout

Expect to drop 20–50% of raw teacher output. Run cheapest filters first.

### 5.1 Rule filters
Answer length 8–1200 chars; question above a minimum; drop refusals, empty fields, template
echoes ("Here are 5 questions..."), and self-referential phrasing.

### 5.2 Grounding check
Fraction of the answer's content words present in the source passage. Below ~0.55 the
teacher likely invented content. For the borderline band (~0.40–0.55): drop, or batch back
to the teacher for a faithfulness check.

### 5.3 Deduplication — still required
The **corpus** is deduplicated; the **generated questions** are not.

1. **Exact** — hash of normalized question.
2. **Near-duplicate** — MinHash/LSH over question shingles (~0.8).
3. **Semantic** — sentence-embedding cosine (~0.90), run last on survivors.

Duplicates concentrate gradient on a few examples.

### 5.4 Decontamination
Already done against LexGLUE and CaseHOLD (§1.6). **If any other benchmark will be used,
decontaminate the QA pairs against it now** — 13-gram overlap. Not recoverable later.

### 5.5 Carve the frozen eval holdout ⚠️ DO THIS NOW

The one measurement decision that cannot be deferred: once a pair is trained on, it can never
be a valid eval item.

- Hold out **300–500 pairs** — not 50. Eval size costs no GPU time and determines whether
  results are resolvable at all. Undersized eval sets are the most common reason a
  fine-tuning study ends with uninterpretable numbers.
- **Stratify** across source, difficulty, and question type.
- **Quarantine at the chunk level.** If a chunk contributes an eval item, remove *all* its
  other pairs from training — sibling pairs leak the answer.
- **Freeze it.** Write once, never regenerate. A changing eval set makes runs incomparable.
- Reserve **10–20 hand-written qualitative probes**, held constant across runs.

### 5.6 Balance and format
Balance to the target mix, render to the chat template (§6), log realized composition.

---

## 6. Phase 4 — Chat template and masking

### 6.1 Template — no new tokens needed

The tokenizer already has `<|system|>`, `<|user|>`, `<|assistant|>`, `<|eos|>`. No vocabulary
extension or embedding resize is required.

```jinja
{% for message in messages %}
{{ '<|' + message['role'] + '|>\n' + message['content'] + eos_token + '\n' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}
```

⚠️ **No `bos_token` in this template — deliberate.** Per §1.5, `<|bos|>` was never seen in
pretraining and its embedding is untrained. Adding it injects noise into every example.

### 6.2 Loss masking — do not skip

**Compute loss only on assistant tokens.** Mask system and user turns with `-100`.

Training on prompt tokens teaches the model to generate questions, dilutes the signal, and is
a frequent silent bug. Verify by decoding one batch's unmasked positions — only answer text
should appear.

### 6.3 Padding vs packing
Prefer **padding** (one example per sequence) for SFT — simplest, keeps examples independent.
Packing needs block-diagonal attention masks to prevent cross-example attention; only do it
with those masks correct.

### 6.4 Context fit
Drop examples exceeding the context ceiling after rendering. **Log how many and which** — if
the filter systematically removes long or hard examples, composition has shifted without
anyone deciding it should.

---

## 7. Phase 5 — Training on Modal

### 7.1 Upload data

```bash
modal volume create ft-dataset          # if it does not exist
modal volume put ft-dataset ./data/sft/pairs.jsonl /sft/pairs.jsonl --force
modal volume put ft-dataset ./data/tokenizer /tokenizer --force
modal volume put ft-dataset ./data/tokens/val /tokens/val --force     # forgetting benchmark
```

⚠️ **On Windows, pass absolute paths.** MSYS/Git-Bash style paths (`/c/Users/...`) are not
resolvable by the Windows Modal client and fail confusingly.

Upload the corpus only if replay is planned — 11 GB is slow and usually unnecessary.

### 7.2 Modal wrapper

```python
import modal

app = modal.App("qa-sft")
volume = modal.Volume.from_name("ft-dataset", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "numpy>=1.26,<2.0")
    .add_local_python_source("config", "train_core")
)

@app.function(image=image, gpu="L4", volumes={"/data": volume}, timeout=60 * 60)
def train(pairs_path: str, out_dir: str, **hp) -> dict:
    import train_core
    return train_core.run(pairs_path=pairs_path, out_dir=out_dir, **hp)

@app.local_entrypoint()
def main():
    print(train.remote("/data/sft/pairs.jsonl", "/data/checkpoints/sft"))
```

- Call **`volume.commit()`** after writing checkpoints or they may not persist.
- **Return metrics as a dict**, not parsed stdout.
- **Pin image versions** — an unpinned `transformers` will eventually break the run.
- Set `timeout` generously; a kill mid-training wastes the whole job.

### 7.3 Training loop requirements

- **Initialize from the base checkpoint for every run** in a comparison — never from a
  previous fine-tune.
- Fixed, recorded seed. bf16 autocast. Cosine LR with warmup.
- Log per N steps: step, loss, LR, tokens seen, throughput, peak VRAM.

⚠️ **Log the corrected loss under gradient accumulation.** If loss is summed across
micro-batches, divide by the accumulation factor before logging — otherwise the curve is
inflated by that factor. This has fooled people into misreading healthy runs.

- Save the final checkpoint plus config; persist metrics as JSONL.

### 7.4 Replay (optional)

If forgetting proves unacceptable (§8.4), mix raw pretraining text into the SFT data at
10–30%. `data/tokens/train/` is present for this. Protection is roughly proportional to a
domain's share of the mix — a domain at 0% is the one that degrades most.

### 7.5 Sanity checks before the full run

1. **Overfit 10 examples.** Loss should approach zero within a few hundred steps. If it does
   not, masking, template, or LR is wrong. Catches most bugs in minutes.
2. **Decode one training batch and read it.** Confirm template rendering and that masking
   covers only the prompt.
3. **Run 10 steps** on real data before launching the full job.

---

## 8. Phase 6 — Post-training measurement

Run after training completes.

**Score the untrained base model on the identical instrument first.** It is the zero point
for every claim below; without it no number means anything.

### 8.1 Persist per-item results

For every metric, save item id, model output, reference, and score — **not just the
aggregate**. Aggregates cannot produce confidence intervals, cannot support paired tests, and
cannot be re-analysed. This loss is unrecoverable without re-running everything.

### 8.2 Confidence intervals on everything

```python
def bootstrap_ci(values, n=10000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    means = rng.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
```

**A difference whose interval includes zero is not a result** — report it as "not resolved".
When comparing two models on the same items, use a **paired** bootstrap on the per-item
differences: far more sensitive, and free.

### 8.3 Primary metrics

Deterministic, continuous, with real dynamic range:

| Metric | Definition |
| --- | --- |
| Exact match | Normalized answer equals reference |
| Token F1 | Token overlap with reference |
| Answer-span recall | Reference string appears in output |
| **Fabrication rate** | Answer contains a number/entity absent from the reference |
| Format adherence | Bounded answer, not a document continuation |
| Refusal correctness | Correct refusal vs false refusal |

**Fabrication rate matters most on this corpus.** Per §0.1, the predicted failure is fluent
invented figures from SEC and case-law questions. Count it from run one rather than
describing it in prose.

### 8.4 Forgetting — use `data/tokens/val/`

Evaluate the fine-tuned model on the held-out pretraining bins and compare with the base
model. **Report per source** — `sec`, `case-law`, `fineweb-edu` separately. An aggregate
hides one domain collapsing while another improves.

| Change vs base | Reading |
| --- | --- |
| < +5% | Mild |
| +5–10% | Notable |
| > +10% | Severe — consider replay (§7.4), lower LR, fewer epochs, or LoRA |

### 8.5 Perplexity
Report held-out SFT perplexity, but **do not headline it**. It measures fit to the answer
distribution and improves reliably even when answer quality does not. Treat it as a
training-health check.

### 8.6 On LLM judges
If used: **check the score histogram first.** If most answers land on one point of the scale,
it is a rare-event indicator, not a quality scale, and its variance will swamp any real
effect. Use ≥200 items, temperature 0, persist per-item scores, report CIs. Deterministic
metrics first.

### 8.7 Decoding
Fix decoding settings and hold them **constant across every model compared**. Greedy decoding
on small models loops; `no_repeat_ngram_size=3` and `repetition_penalty≈1.2` are reasonable
defaults. Changing decoding between conditions invalidates the comparison — easy to do by
accident, hard to notice afterward.

### 8.8 Run report
Hyperparameters, seed, effective batch size · dataset composition and filter drop rates ·
base-model scores (zero point) · fine-tuned scores with CIs · per-source forgetting · sample
generations including failures · wall-clock, GPU, cost · per-item results attached as JSON.

---

## 9. Failure modes

| Symptom | Likely cause |
| --- | --- |
| Loss near zero almost immediately | Loss masking wrong — training on prompt tokens |
| Model generates questions instead of answering | Same |
| **Fluent, confident, wrong figures** | **Closed-book on document specifics — §0.1. Restrict closed-book to fineweb-edu** |
| Repetition loops | Greedy decoding; add repetition penalty. Check EOS is trained |
| Never emits EOS | EOS missing from targets in the template |
| Garbled legal text in outputs | OCR noise in case-law; tighten the chunk filter |
| A few SEC filings dominate the data | No per-document chunk cap (§3.1) |
| Perplexity improves, quality flat | Expected — perplexity is not a quality metric |
| Severe forgetting | LR too high, too many epochs, or no replay |
| Eval scores implausibly high | Contamination (§5.4, §5.5) |
| Results move with no trend | Eval set too small (§5.5) |
| Nothing changes at all | LR too low, LoRA on wrong modules, or optimizer not stepping |
| Odd behaviour from step 0 | `<|bos|>` prepended despite §1.5 |

---

## 10. Definition of done

- [ ] Base model scored on the frozen eval set — the zero point exists
- [ ] Fine-tuned model scored on the identical instrument
- [ ] Every metric reported with a bootstrap 95% CI
- [ ] Per-item results persisted for every metric
- [ ] Per-source forgetting measured on `data/tokens/val/` vs the base model
- [ ] Fabrication rate reported (§8.3)
- [ ] Decoding settings identical across all compared models
- [ ] Qualitative probes run and recorded
- [ ] Run report written with full hyperparameters and costs
- [ ] Explicit statement of what the model can and cannot do, including negative results

**A negative result, reported with error bars, is a finding. A positive result without error
bars is not.**
