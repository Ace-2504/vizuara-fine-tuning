# RAFT — Retrieval-Augmented Fine-Tuning — Implementation Guide for This Corpus

**Audience:** an engineering session implementing RAFT using the 2.5B-token legal/financial
corpus in this repository, on Modal.

**Self-contained.** Everything needed is here. No other document is required.

---

## 0. What RAFT is

RAFT (Zhang et al., 2024, UC Berkeley — *RAFT: Adapting Language Model to Domain-Specific
RAG*) fine-tunes a model to answer questions **from documents placed in its prompt**, while
being robust to a retriever that also returns irrelevant documents.

Closed-book fine-tuning is a closed-book exam. Plain RAG is an open-book exam the model never
studied for. **RAFT is studying for an open-book exam.**

### 0.1 The three defining ingredients

A method missing these is not RAFT — it is open-book QA fine-tuning, which is a reasonable
thing to build but has different properties and different published results.

**1. Distractor documents.** Each example contains the golden document (which contains the
answer) *plus* several distractors — plausible, topically related documents that do not
contain the answer. Real retrieval is never clean.

**2. A fraction of examples with no golden document at all.** In a proportion of training
examples, *only* distractors are provided. The model must recognise the answer is absent
rather than confabulating. This produces robustness to retrieval failure, and it is the
ingredient most often dropped.

**3. Answers that quote the source verbatim, then answer.** The target begins by citing the
exact supporting span, marked explicitly:

```
##begin_quote## Net revenues for fiscal 1997 were $5.6 billion, an increase of 12%. ##end_quote##
Final answer: $5.6 billion
```

This is not cosmetic. Extraction is mechanically easier than free-form generation — most
pretrained transformers copy a span from context far more reliably than they compose a
faithful paraphrase. Quote-first routes the task through the capability the model already
has. Training on purely abstractive answers teaches "produce fluent text of roughly this
shape", which is exactly the behaviour that yields confident fabrication.

*(Parameter values below — distractor count, golden-free proportion — vary by dataset in the
paper. Treat them as starting points and verify against the paper if exact replication
matters.)*

### 0.2 Why RAFT suits this corpus

This corpus is SEC filings and US case law — document-heavy material full of specific figures,
dates, and holdings. Closed-book fine-tuning on that material asks a model to recall
long-tail facts it saw once, which it cannot do; it learns the shape of an answer and
fabricates the content.

**RAFT sidesteps this entirely.** The passage is in the prompt, so the model reads rather
than recalls. For this corpus RAFT is the better-matched method.

---

## 1. The dataset in this repository

Everything below already exists on disk. **Do not re-clean, re-deduplicate, or
re-decontaminate it** — those phases are complete.

### 1.1 Layout

```
fine-tuning-dataset/
  data/
    corpus/                    # 11 GB — golden passages AND the distractor pool
      case-law/    shard-000.txt … shard-009.txt
      sec/         shard-000.txt … shard-004.txt
      fineweb-edu/ shard-000.txt … shard-009.txt
    tokenizer/                 # 16,384-token byte-level BPE
    tokens/
      train/                   # 4.6 GB — pretraining bins (uint16), for REPLAY
      val/                     # held-out bins — the FORGETTING benchmark
      index.json
```

### 1.2 Corpus format — important

**One document per line, UTF-8, plain text.** Internal newlines were stripped during
cleaning. No JSON wrapper, no separator token.

```python
def iter_docs(path):
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            text = line.rstrip("\n")
            if text:
                yield line_no, text
```

**The line number is the document identity.** `(source, shard, line_no)` uniquely identifies
a document — this is what makes same-document distractor exclusion possible (§4.1), which
RAFT depends on.

### 1.3 Contents

| Source | Shards | Documents | Size | Character |
| --- | --- | --- | --- | --- |
| `case-law` | 10 | 241,356 | 3.6 GB | US court opinions — some OCR noise |
| `sec` | 5 | 45,035 | 4.1 GB | SEC filings — **very long documents** |
| `fineweb-edu` | 10 | 690,923 | 3.1 GB | Educational web text — cleanest |
| **Total** | **25** | **977,314** | **10.8 GB** | |

**SEC has the fewest documents but the most bytes.** A single filing yields dozens of chunks.
This makes same-document exclusion critical: two chunks from one filing frequently restate
the same fact, so an unfiltered "distractor" may contain the answer.

### 1.4 Tokenized bins

| | Tokens | Windows |
| --- | --- | --- |
| Train (`data/tokens/train/`) | 2,475,090,944 | 2,417,081 |
| Val (`data/tokens/val/`) | 25,006,080 | 24,420 |
| **Total** | **2,500,097,024** | |

Composition (train): **sec 34.7%** (860,036,096) · **case-law 34.2%** (846,758,912) ·
**fineweb-edu 31.0%** (768,295,936). Format `uint16`, `seq_len = 1024`, 14 shards, 99/1 split.

**The val bins are the forgetting benchmark** (§8.5). Nothing has trained on them.

### 1.5 Tokenizer

`data/tokenizer/` — 16,384-token byte-level BPE. Special tokens already defined:

```
<|bos|>  <|eos|>  <|pad|>  <|unk|>  <|user|>  <|assistant|>  <|system|>
```

⚠️ **The chat tokens exist in the vocabulary but were never seen during pretraining.** The
tokenization pipeline used `add_special_tokens=False` and appended only `<|eos|>`.

1. **`<|bos|>` was never trained** — its embedding is at random initialization. **Do not
   prepend it.** Use `<|eos|>` as the only structural token, matching pretraining.
2. **`<|user|>`, `<|assistant|>`, `<|system|>` embeddings are also at initialization.** They
   train during SFT but start from noise.

**Also check how `##begin_quote##` tokenizes.** This is a 16K byte-level BPE trained on legal
text; the marker will fragment into several tokens. That is acceptable, but confirm it round-
trips exactly, and consider a shorter marker (e.g. `>>` / `<<`) if fragmentation is severe —
fewer tokens spent on markers means more budget for documents, which is scarce here (§2.2).

### 1.6 What has already been done

| Stage | Status | Detail |
| --- | --- | --- |
| Cleaning | ✅ Complete | Rule chain + document-level OCR gate |
| Exact dedup | ✅ Complete | sec −1,989 · fineweb-edu −120 |
| Near-dup (MinHash) | ✅ Complete | case-law −1,643 |
| Decontamination | ✅ Complete | 13-gram vs **LexGLUE** and **CaseHOLD** |
| Tokenization | ✅ Complete | 1,024-token windows, 99/1 split |

⚠️ Decontamination covers **LexGLUE and CaseHOLD only**. Any other evaluation benchmark
requires its own decontamination before training — not fixable afterward.

⚠️ Corpus dedup does **not** remove duplicate *questions*. QA-pair dedup (§6.2) is still
required.

---

## 2. Questions to answer before you start

Corpus questions are answered above. These depend on the model:

| # | Question | Guidance |
| --- | --- | --- |
| 1 | **Which base model** — repo/path, parameters? | Must use `data/tokenizer/` |
| 2 | **Context length?** | ⚠️ **§2.2 — the binding constraint for RAFT** |
| 3 | **Was it pretrained on this tokenizer?** | If not, `data/tokens/` and the forgetting benchmark do not apply |
| 4 | **How many training examples?** | §2.1 |
| 5 | **Distractors per example (k)?** | §2.2 |
| 6 | **Golden-document proportion (P)?** | §2.3 |
| 7 | **Chain-of-thought or quote-only answers?** | §2.4 |
| 8 | **Modal GPU? Full FT or LoRA?** | §2.5 |
| 9 | **Teacher model, billing enabled?** | Rate limits dominate wall-clock |
| 10 | **What retriever at inference?** | Distractors should match its output |
| 11 | **Exact abstention wording?** | Must be fixed before generation |
| 12 | **Benchmarks beyond LexGLUE/CaseHOLD?** | Extra decontamination needed |
| 13 | **Replay wanted?** | `data/tokens/train/` is present |

### 2.1 How many training examples

| Base model size | Starting point |
| --- | --- |
| ≤ 300M | 3,000 – 10,000 |
| 300M – 1B | 10,000 – 30,000 |
| 1B – 3B | 20,000 – 50,000 |
| 7B+ | 30,000 – 100,000+ |

RAFT examples are far longer than closed-book pairs (several documents each), so **budget
training time on tokens, not example count**.

### 2.2 ⚠️ Context budget — do this arithmetic before generating anything

This is where RAFT implementations most often break, and this corpus makes it acute: it was
packed at **1,024 tokens**, so a model pretrained on it very likely has a 1,024-token context.

```
total = system + question + (k+1) × doc_tokens + answer + margin
```

**Worked budget for a 1,024-token context:**

| Component | Tokens |
| --- | --- |
| System prompt | ~45 |
| Question | ~30 |
| Answer (quote + final answer) | ~90 |
| Document markers, newlines, margin | ~40 |
| **Available for documents** | **~820** |

| k | Total docs | Tokens per doc |
| --- | --- | --- |
| 1 | 2 | ~410 |
| **2** | **3** | **~270** ← recommended at 1,024 ctx |
| 3 | 4 | ~205 |

**Consequence: chunk at ~250 tokens for RAFT**, not the ~500 you would use for closed-book
QA. This is the single most important corpus-specific parameter in this guide.

If the model has a larger context, scale up:

| Model context | Practical config |
| --- | --- |
| 1,024 | **k = 2, ~250-token docs** |
| 2,048 | k = 3, ~400-token docs |
| 4,096 | k = 4, ~600-token docs |
| 8,192+ | k = 4–6, ~1,000-token docs |

### 2.3 Golden-document proportion (P)

`P` = fraction of examples including the golden document; `1 − P` contain **only distractors**.

- `P ≈ 0.8` is a reasonable start.
- Lower `P` → more robust to retrieval failure, more prone to over-refusing.
- `P = 1.0` → **not RAFT.** The model never learns context can fail and will confabulate when
  retrieval misses.

Behaviour for golden-free examples — **abstention is recommended for this corpus**, because
the material is document-specific and parametric answering would be fabrication:

```
not stated in the context
```

Fix the exact wording before generation. Changing it later invalidates the data.

### 2.4 Chain-of-thought vs quote-only

| Model size | Answer format |
| --- | --- |
| ≤ 1B | **Quote-only** — `##begin_quote## … ##end_quote##` then a short direct answer |
| 1B – 7B | Quote + brief reasoning |
| 7B+ | Full CoT with citations, as published |

At a 1,024-token context (§2.2) the answer budget is ~90 tokens — **CoT does not fit**.
Quote-only is not merely preferred here, it is forced.

**Keep the quote in every configuration.** It is the ingredient that connects training to
span extraction.

### 2.5 Modal GPU and fine-tuning mode

Approximate rates (verify current pricing):

| GPU | ~$/hr | Suits |
| --- | --- | --- |
| T4 | ~$0.60 | ≤150M; inference |
| **L4** | **~$0.80** | ≤1B full FT — best value for small models |
| A10G | ~$1.10 | ≤1.5B full FT |
| A100-40GB | ~$2.10 | 1–3B full FT; 7B LoRA |
| A100-80GB | ~$2.50–3.00 | 3–7B full FT; 13B LoRA |
| H100 | ~$4.00–5.00 | 7B+ full FT |

RAFT sequences are long, so **VRAM is driven by activations**. On OOM, reduce micro-batch and
raise gradient accumulation before moving to a bigger GPU. Full FT ≤3B; LoRA above (r=16–64,
alpha=2r, attention + MLP projections).

### 2.6 Hyperparameter starting points

| Parameter | Value |
| --- | --- |
| Learning rate | 1e-5 – 3e-5 full FT; 1e-4 – 2e-4 LoRA |
| Epochs | 2–3 |
| Effective batch size | 8–32 (lower than closed-book — sequences are long) |
| Schedule | Cosine to ~10% of peak |
| Warmup | 3–5% of steps |
| Weight decay | 0.0 · Grad clip 1.0 · bf16 · fixed seed |

---

## 3. Phase 1 — Chunking and the retrieval index

**The corpus is already clean. This phase splits documents and builds the distractor index.**

### 3.1 Chunker — ~250 tokens for RAFT

```python
import hashlib, os
from dataclasses import dataclass
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("data/tokenizer")
CHUNK_TOKENS = 250              # from §2.2 — NOT 500
MAX_CHUNKS_PER_DOC = 12         # stop long SEC filings dominating

@dataclass
class Chunk:
    chunk_id: str; text: str; token_len: int
    source: str; doc_id: str          # doc_id = f"{source}/{shard}:{line_no}"

def chunks_from_shard(source: str, shard: str):
    path = os.path.join("data/corpus", source, shard)
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            text = line.rstrip("\n")
            if not text:
                continue
            ids = TOK(text, add_special_tokens=False)["input_ids"]
            doc_id = f"{source}/{shard}:{line_no}"
            for n, i in enumerate(range(0, len(ids), CHUNK_TOKENS)):
                if n >= MAX_CHUNKS_PER_DOC:
                    break
                piece = ids[i:i + CHUNK_TOKENS]
                if len(piece) < 80:
                    continue
                body = TOK.decode(piece)
                yield Chunk(hashlib.sha256(body.encode()).hexdigest()[:16],
                            body, len(piece), source, doc_id)
```

- **`chunk_id` is a content hash**, never an index.
- **`doc_id` is mandatory** — §4.1 depends on it.
- `MAX_CHUNKS_PER_DOC` matters here: without it, a handful of long SEC filings dominate.

### 3.2 Build the embedding index — on a sampled pool

The full corpus at 250 tokens/chunk yields **~10 million chunks**. Do not embed all of them.

**Sample a working pool of 300,000–500,000 chunks**, stratified by source. That is ample —
distractors only need to be plausible neighbours, not exhaustive.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode([c.text for c in pool], normalize_embeddings=True,
                   batch_size=256, show_progress_bar=True)
np.save("data/chunk_emb.npy", emb)          # cosine similarity = dot product
```

Cost check: 400k chunks ≈ 8–15 min on a GPU; the matrix is 400k × 384 × 4B ≈ **614 MB**, which
fits in RAM. Plain dot products are fine at this scale — no FAISS needed.

Persist the pool (`chunk_id`, `doc_id`, `source`, text offset) alongside the matrix so row
indices stay aligned across runs.

### 3.3 Domain mix

Mirror the corpus (sec 34.7 / case-law 34.2 / fineweb-edu 31.0) or weight toward the target
application. Record the realized mix and hold it constant across comparisons.

**Note on case-law:** it carries residual OCR noise that survived the document-level gate.
Quotes drawn from it may contain garbled text. §5.3's quote verification catches
mismatches; consider an additional non-dictionary-rate filter on candidate golden chunks.

---

## 4. Phase 2 — Distractor selection

The distinguishing step of RAFT. Getting it wrong reduces the method to ordinary open-book
fine-tuning.

### 4.1 Hard distractors, with same-document exclusion

| Strategy | Effect |
| --- | --- |
| Random chunks | Trivially separable — usually a different topic. Teaches almost nothing |
| **Nearest neighbours by embedding** ✅ | Same topic, no answer. Genuinely confusable. **Use this** |
| Actual retriever output ✅✅ | Best fidelity if the production retriever exists |

```python
def pick_distractors(golden_row, emb, k, pool):
    """Top-k most similar chunks that are NOT from the golden document."""
    sims = emb @ emb[golden_row]
    sims[golden_row] = -1.0
    golden_doc = pool[golden_row].doc_id
    out = []
    for i in np.argsort(-sims):
        if pool[i].doc_id == golden_doc:      # CRITICAL — see below
            continue
        out.append(int(i))
        if len(out) == k:
            break
    return out
```

⚠️ **The `doc_id` exclusion is not optional on this corpus.** SEC filings run to dozens of
chunks that restate the same figures, and case-law opinions repeat holdings across sections.
A "distractor" containing the answer teaches the model that quote markers are decorative and
silently corrupts training. This is the single most likely way to get RAFT subtly wrong here.

### 4.2 Shuffle position
Randomize where the golden document sits among the distractors. If it is always first, the
model learns position rather than relevance and breaks when the retriever reorders.

### 4.3 Golden-free examples
For the `1 − P` share, take the top-k distractors and omit the golden document. Target is the
abstention string from §2.3.

---

## 5. Phase 3 — Teacher generation

### 5.1 Teacher client

```python
class TeacherClient:
    def generate_json(self, prompt: str, schema, *, temperature: float = 0.9) -> dict | list:
        """Returns parsed JSON. Retries on 429/5xx with exponential backoff."""
```

**Robustness — a multi-hour run must survive any single bad response:**

- **Enforce structured output** (response schema / JSON mode).
- **Handle HTTP 200 with an empty body.** `json.loads(None)` raises and kills the run.
- **Handle unparseable JSON** even when a schema was requested.
- **Retry, then skip.** Never abort over one response.
- **Rate-limit deliberately**; confirm paid vs free tier before estimating wall-clock.
- **Track usage** for the cost report.

### 5.2 Resumability
Append to JSONL as records are produced. Persist a state file. On restart, count what exists
and generate only the remainder. Make resume the **default** path so it is always exercised.

### 5.3 Generate against the GOLDEN chunk only

The teacher sees one passage. Distractors are assembled later (§6.3) and are never shown to
the teacher.

```python
def raft_qa_prompt(passage: str, n: int) -> str:
    return (
        f"Read the PASSAGE and write {n} diverse question-answer pairs.\n"
        "For each, return JSON with fields:\n"
        '  "question" - self-contained; must NOT refer to "the passage", "the text", '
        '"the document", or "above". It must name the company, case, or topic.\n'
        '  "quote"    - the EXACT verbatim span from the passage supporting the answer, '
        "copied character-for-character. Do not paraphrase. Do not add ellipses.\n"
        '  "answer"   - a short, direct answer.\n'
        "Rules:\n"
        "- The answer must be fully supported by the quote.\n"
        "- Do NOT invent facts.\n"
        "- Vary difficulty: simple lookups and multi-step questions.\n"
        "- No duplicate or near-duplicate questions.\n\n"
        f"PASSAGE:\n{passage}"
    )
```

**Verify every quote programmatically. Teachers paraphrase even when told not to.**

```python
def verify_quote(quote: str, passage: str) -> bool:
    norm = lambda s: " ".join(s.split()).lower()
    return norm(quote) in norm(passage)
```

An unverified quote trains the model to fabricate citations — worse than no citation, because
it looks trustworthy. Drop or repair failures; do not pass them through.

⚠️ **Cap quote length** to ~60 tokens. At a 1,024-token context the whole answer budget is
~90 tokens (§2.2); a teacher that quotes half the passage produces examples that will not fit.

### 5.4 Unanswerable questions
Separately, ask the teacher for on-topic questions the passage does **not** answer. These are
distinct from §4.3's golden-free examples:

- **Golden present, answer absent** — "the right document is here but doesn't say"
- **Golden absent (§4.3)** — "the right document was never retrieved"

Both are needed. Target 5–15% of golden-present examples for the first kind.

### 5.5 Record format

```json
{"chunk_id": "...", "doc_id": "sec/shard-000.txt:12345", "source": "sec",
 "question": "...", "quote": "...", "answer": "...", "answerable": true,
 "teacher": "model-name", "batch": 1}
```

---

## 6. Phase 4 — Assembly and the frozen holdout

### 6.1 Filter
- **Quote verifies as exact substring** (§5.3) — hard requirement.
- Quote within the token cap.
- Answer supported by the quote (content-word overlap).
- Length bounds; drop self-referential questions, template echoes, empty fields.

### 6.2 Deduplicate — still required
The corpus is deduplicated; generated questions are not. Exact hash → MinHash/LSH (~0.8) →
embedding cosine (~0.90). Cheapest first.

### 6.3 Assemble

```python
def assemble(rec, distractor_rows, keep_golden, rng, pool, cfg):
    docs = [pool[i].text for i in distractor_rows]
    if keep_golden:
        docs.insert(rng.randrange(len(docs) + 1), rec["passage"])   # random position
        answer = f'##begin_quote## {rec["quote"]} ##end_quote##\n{rec["answer"]}'
    else:
        answer = cfg.ABSTAIN_STRING
    context = "\n\n".join(f"[Document {i+1}]\n{d}" for i, d in enumerate(docs))
    return {
        "messages": [
            {"role": "system", "content": cfg.RAFT_SYSTEM_PROMPT},
            {"role": "user",   "content": f"{context}\n\nQuestion: {rec['question']}"},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"golden_present": keep_golden, "n_distractors": len(docs) - int(keep_golden),
                 "chunk_id": rec["chunk_id"], "doc_id": rec["doc_id"], "source": rec["source"]},
    }
```

System prompt, fixed for the whole study (~45 tokens, per the §2.2 budget):

```
You are a grounded assistant. Answer using ONLY the provided documents.
Quote the exact supporting text, then give your answer.
If the answer is not in the documents, reply exactly: not stated in the context.
```

### 6.4 Composition to enforce

| Dimension | Target |
| --- | --- |
| Golden present | `P` (e.g. 0.80) |
| Golden absent (distractors only) | `1 − P` |
| Answerable-but-absent (golden present, no answer) | 5–15% of golden-present |
| Golden position | Uniform across slots |
| Domain mix | §3.3, recorded |

### 6.5 Carve the frozen eval holdout ⚠️ DO THIS NOW

The one measurement decision that cannot wait — once trained on, an item can never be a valid
eval item.

- **300–500 items.** Eval size costs no GPU time and determines whether results are
  resolvable. Undersized eval sets are the most common reason a study ends with
  uninterpretable numbers.
- **Quarantine at the `doc_id` level** — stricter than `chunk_id`. If a document contributes
  an eval item, bar **all** its chunks from training *and* from the distractor pool. On this
  corpus, sibling chunks of one SEC filing routinely restate the same fact.
- **Build matched sets** — the *same question* across all four conditions:

| Condition | Golden | Distractors | Tests |
| --- | --- | --- | --- |
| **Clean** | ✅ | none | Basic reading |
| **Realistic** | ✅ | k | The production setting |
| **Retrieval failure** | ❌ | k | Abstention / robustness |
| **Closed-book** | ❌ | none | Parametric knowledge (control) |

Paired comparison across matched conditions is far more powerful than four independent
samples, and it is free.

- **Freeze it.** Write once, never regenerate.
- Reserve **10–20 hand-written probes**, including passages unlike this corpus (§8.7).

---

## 7. Phase 5 — Template, masking, and training

### 7.1 Template — no new tokens needed

```jinja
{% for message in messages %}
{{ '<|' + message['role'] + '|>\n' + message['content'] + eos_token + '\n' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}
```

⚠️ **No `bos_token` — deliberate.** Per §1.5, `<|bos|>` was never seen in pretraining.

### 7.2 Loss masking — critical here

**Compute loss only on assistant tokens.** Mask system and user turns with `-100`.

This matters more in RAFT than in closed-book SFT: the prompt is ~820 tokens of documents and
the answer is ~90. **Without masking, roughly 90% of the loss comes from predicting document
text**, and the model learns to continue documents rather than answer questions. Verify by
decoding one batch's unmasked positions — only the answer should appear.

### 7.3 Sequence length and truncation

Set `max_seq_len` to the model's context (1,024 for a model pretrained on these bins).

⚠️ **Log truncation counts and fail loudly if non-trivial.** Truncation that removes the
golden document converts an answerable example into an unlabelled golden-free one, poisoning
the signal invisibly. If truncation occurs, reduce `k` or `CHUNK_TOKENS` (§2.2) — do not
accept it.

### 7.4 Upload and Modal wrapper

```bash
modal volume create ft-dataset
modal volume put ft-dataset ./data/raft/pairs.jsonl /raft/pairs.jsonl --force
modal volume put ft-dataset ./data/tokenizer /tokenizer --force
modal volume put ft-dataset ./data/tokens/val /tokens/val --force     # forgetting benchmark
```

⚠️ **On Windows, pass absolute paths.** MSYS/Git-Bash paths (`/c/Users/...`) fail confusingly.

```python
import modal

app = modal.App("raft-sft")
volume = modal.Volume.from_name("ft-dataset", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "numpy>=1.26,<2.0")
    .add_local_python_source("config", "train_core")
)

@app.function(image=image, gpu="L4", volumes={"/data": volume}, timeout=60 * 90)
def train(pairs_path: str, out_dir: str, **hp) -> dict:
    import train_core
    return train_core.run(pairs_path=pairs_path, out_dir=out_dir, **hp)

@app.local_entrypoint()
def main():
    print(train.remote("/data/raft/pairs.jsonl", "/data/checkpoints/raft"))
```

- Call **`volume.commit()`** after writing checkpoints.
- **Return metrics as a dict**, not parsed stdout. **Pin image versions.**

### 7.5 Training loop requirements
- Initialize from the **base** checkpoint every run in a comparison.
- Fixed recorded seed · bf16 · cosine LR with warmup.
- Log per N steps: step, loss, LR, tokens, throughput, peak VRAM.
- ⚠️ **Log the corrected loss under gradient accumulation** — divide by the accumulation
  factor, or the curve is inflated by it.

### 7.6 Replay (optional)
If forgetting is unacceptable (§8.5), mix pretraining text from `data/tokens/train/` at
10–30%. Protection is roughly proportional to a domain's share — the domain at 0% degrades
most.

### 7.7 Sanity checks before the full run
1. **Overfit 10 examples** — loss should approach zero. If not, masking/template/LR is wrong.
2. **Decode one batch and read it.** Confirm documents render, the golden document is present
   where expected, masking covers the whole prompt, and nothing is truncated.
3. **Confirm quote markers round-trip** through tokenization.
4. **Run 10 steps** on real data before the full job.

---

## 8. Phase 6 — Post-training measurement

**Score the untrained base model in all four conditions first** — the zero point.

### 8.1 Persist per-item results
Save item id, condition, model output, reference, and score — **not just aggregates**.
Aggregates cannot yield CIs or support paired tests, and the loss is unrecoverable without
re-running everything.

### 8.2 Confidence intervals

```python
def bootstrap_ci(values, n=10000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    means = rng.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
```

A difference whose interval includes zero is **not resolved**. Because §6.5 built matched
conditions, use a **paired** bootstrap on per-item differences.

### 8.3 The four-condition evaluation — the core of RAFT verification

| Condition | Expected if RAFT worked |
| --- | --- |
| **Clean** | Highest accuracy |
| **Realistic** | **Close to clean** — a small gap means distractor robustness |
| **Retrieval failure** | **High abstention**, low fabrication |
| **Closed-book** | Low — this is the control, and it *should* be low |

- **Distractor robustness gap** = clean − realistic. Large ⇒ distractors still confusing the
  model; use more/harder distractors or lower `P`.
- **Grounding gap** = realistic − closed-book. The headline. **Compare against the base
  model's own gap** — an untrained model already uses context to some degree, and only
  movement above that baseline is attributable to RAFT.

### 8.4 Primary metrics

| Metric | Definition |
| --- | --- |
| Answer exact match / token F1 | vs reference |
| **Quote validity rate** | Emitted quote is a verbatim substring of a provided document |
| **Quote precision** | Quote comes from the **golden** document, not a distractor |
| Copy rate | Answer contains a verbatim ≥5-token span from the documents |
| **Fabrication rate** | Answer contains a number/entity absent from all provided documents |
| **Correct abstention** | Abstains when golden is absent |
| **False abstention** | Abstains when the answer *is* present |
| Format adherence | Quote markers well-formed and parseable |

**Fabrication rate and false abstention must be read together.** A model that abstains
constantly scores perfectly on fabrication and is useless; one that never abstains fabricates
under retrieval failure.

### 8.5 Forgetting — use `data/tokens/val/`
Evaluate against the base model, **per source** (`sec`, `case-law`, `fineweb-edu`). An
aggregate hides one domain collapsing while another improves.

| Change vs base | Reading |
| --- | --- |
| < +5% | Mild |
| +5–10% | Notable |
| > +10% | Severe — replay (§7.6), lower LR, fewer epochs, or LoRA |

Report held-out perplexity but **never headline it**.

### 8.6 Retrieval-count sensitivity
Score at k = 0, 2, 4, 8. Degradation should be graceful. A cliff just past the training `k`
means the model overfit to one retrieval configuration and will break when the retriever's
output size changes.

### 8.7 Out-of-distribution grounding — do not skip
Evaluate on passages unlike this corpus (contemporary news, general reference, technical
prose). A model can learn to exploit legal/financial surface patterns rather than to read.
Without this check, "learned to use context" cannot be distinguished from "learned this
corpus's phrasing" — and only the first generalizes.

### 8.8 Decoding
Fix decoding and hold it constant across every model compared. `no_repeat_ngram_size=3`,
`repetition_penalty≈1.2` are reasonable defaults. Changing decoding between conditions
invalidates the comparison.

### 8.9 Run report
Hyperparameters, seed, effective batch size, `k`, `P`, distractor strategy · composition,
drop rates, **truncation counts** · base-model scores in all four conditions · fine-tuned
scores with CIs · robustness and grounding gaps vs base · quote validity and precision ·
fabrication and false-abstention rates · per-source forgetting · OOD results · sample
generations including failures · cost · per-item results as JSON.

---

## 9. Failure modes

| Symptom | Likely cause |
| --- | --- |
| Model continues documents instead of answering | Loss masking wrong — dominant here, prompts are ~90% of tokens |
| Quotes are paraphrases | Teacher quotes never verified (§5.3) |
| **Quotes come from distractors** | **`doc_id` exclusion missing (§4.1) — most likely subtle bug on this corpus** |
| Examples silently truncated | `CHUNK_TOKENS` or `k` too large for the context (§2.2, §7.3) |
| High accuracy clean, poor with distractors | Distractors too easy (random not hard), or `k` too low |
| Never abstains, fabricates on retrieval failure | `P` too high — likely `P = 1.0`, which is not RAFT |
| Abstains constantly | `P` too low, or too many unanswerable examples |
| Good in-distribution, poor OOD | Learned corpus surface patterns, not reading (§8.7) |
| Cliff past training `k` | Overfit to one retrieval configuration (§8.6) |
| Golden position matters at inference | Position not randomized (§4.2) |
| Garbled quotes | case-law OCR noise — add a non-dictionary-rate filter (§3.3) |
| A few SEC filings dominate | `MAX_CHUNKS_PER_DOC` not set (§3.1) |
| Odd behaviour from step 0 | `<|bos|>` prepended despite §1.5 |
| Perplexity improves, quality flat | Expected — not a quality metric |
| Eval scores implausibly high | Contamination (§1.6, §6.5) |

---

## 10. Definition of done

- [ ] Base model scored in **all four conditions** — the zero point exists
- [ ] Fine-tuned model scored on the identical instrument
- [ ] Every metric with a bootstrap 95% CI; paired where conditions are matched
- [ ] Per-item results persisted for every metric and condition
- [ ] Distractor robustness gap and grounding gap reported **relative to the base model**
- [ ] Quote validity and quote precision measured
- [ ] Fabrication and false-abstention rates reported together
- [ ] Per-source forgetting measured on `data/tokens/val/`
- [ ] OOD grounding measured
- [ ] Retrieval-count sensitivity measured
- [ ] Truncation count confirmed ≈ 0
- [ ] Decoding identical across all compared models
- [ ] Run report with `k`, `P`, distractor strategy, and full costs

**A negative result, reported with error bars, is a finding. A positive result without error
bars is not.**
