# Dataset Build Guide — QA SFT + RAFT, from the 2.5B corpus

**Audience:** an engineering session building the fine-tuning datasets, once, from the corpus
in this repository (`D:\vizuara-fine-tuning\`), using **Gemini 3.1 Flash** as the teacher.

**Self-contained.** Everything needed is here.

> **This is the shared upstream build. Run it ONCE.** It produces text-level datasets that
> all three models (125M, 500M, Gemma-2B) consume. Do **not** regenerate per model:
> regenerating gives each model a different dataset, which destroys the cross-model
> comparison *and* triples the teacher cost. Rendering into each model's tokenizer/template
> happens later, in the per-model fine-tuning guides — the artifacts here stay as **text**.

---

## 0. What this build produces

Three frozen, text-level artifacts (chat JSONL — `system`/`user`/`assistant`), plus their
per-item provenance:

| Artifact | What it is | Feeds |
| --- | --- | --- |
| `data/sft/qa_sft.jsonl` | Instruction-tuning pairs across 4 task types | QA-SFT track of every model |
| `data/sft/raft.jsonl` | Grounded QA with distractors + abstention, quote-first | RAFT track of every model |
| `data/sft/eval.jsonl` | Frozen held-out eval, matched conditions | Evaluation for every model |
| `data/sft/provenance/` | Per-pair judge scores, grounding, dedup keys, source chunk | Audit / re-analysis |

**RAFT reuses the QA generation pool** — a grounded `(question, quote, answer, passage)` is
rendered closed-book/grounded for QA SFT *and* assembled with distractors for RAFT. One
generation pass, two datasets. Only RAFT's abstention examples need extra generation.

---

## 1. The corpus (already prepared — do not re-clean)

```
D:\vizuara-fine-tuning\data\
  corpus\        11 GB — SOURCE PASSAGES. One document per line, UTF-8, plain text.
    case-law\    shard-000.txt … shard-009.txt   (241,356 docs)
    sec\         shard-000.txt … shard-004.txt   ( 45,035 docs, very long)
    fineweb-edu\ shard-000.txt … shard-009.txt   (690,923 docs)
  tokenizer\     16,384 byte-level BPE (the 125M's tokenizer)
  tokens\        pretraining bins — NOT used in this build (that's a training-phase asset)
```

Already cleaned, exact+near deduplicated, and decontaminated (13-gram vs **LexGLUE** and
**CaseHOLD**). ⚠️ If any model will be evaluated on another benchmark, decontaminate the
generated pairs against it in §6.5 — corpus-level decontamination does not cover it.

Read a document:

```python
def iter_docs(path):
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            t = line.rstrip("\n")
            if t:
                yield line_no, t
```

`(source, shard, line_no)` is the document identity — needed for RAFT same-document
distractor exclusion (§7.2).

---

## 2. Teacher — Gemini 3.1 Flash

```python
# pip install google-genai
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])   # .env, gitignored
GEN_MODEL   = "gemini-3.1-flash"      # generation
JUDGE_MODEL = "gemini-3.1-flash"      # LLM-as-judge (§6.4). A stronger model may be used here.
```

**Robustness — a multi-hour run must survive any single bad response:**

- **Structured output.** Use `response_schema` / JSON mode; never parse free text.
- **Handle HTTP 200 with an empty body** — `json.loads(None)` raises and kills the run.
- **Handle unparseable JSON** even when a schema was requested.
- **Retry (exp. backoff on 429/5xx), then skip.** Never abort over one response.
- **Rate-limit deliberately.** Confirm the key is on a paid tier before estimating wall-clock
  — free tier is dramatically slower. Set a min interval between requests.
- **Track token usage** (`response.usage_metadata`) — this is what the cost report is built
  from. Log input+output tokens per call.

**Resumable:** append each record to JSONL as produced; persist a state file with per-recipe
counts; on restart, generate only the remainder. Make resume the default path.

---

## 3. Phase 1 — Chunking

The corpus is clean; this only splits documents into passages, with the **125M tokenizer** so
token budgets are exact for the tightest model.

```python
import hashlib, os
from dataclasses import dataclass
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("data/tokenizer")
CHUNK_TOKENS = 256              # RAFT-compatible; serves BOTH datasets (see note)
MAX_CHUNKS_PER_DOC = 12         # stop long SEC filings dominating

@dataclass
class Chunk:
    chunk_id: str; text: str; token_len: int
    source: str; doc_id: str          # doc_id = f"{source}/{shard}:{line_no}"

def chunks_from_shard(source, shard):
    with open(f"data/corpus/{source}/{shard}", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            text = line.rstrip("\n")
            if not text: continue
            ids = TOK(text, add_special_tokens=False)["input_ids"]
            doc_id = f"{source}/{shard}:{line_no}"
            for n, i in enumerate(range(0, len(ids), CHUNK_TOKENS)):
                if n >= MAX_CHUNKS_PER_DOC: break
                piece = ids[i:i+CHUNK_TOKENS]
                if len(piece) < 80: continue
                body = TOK.decode(piece)
                yield Chunk(hashlib.sha256(body.encode()).hexdigest()[:16],
                            body, len(piece), source, doc_id)
```

**Why 256 tokens (not 500):** RAFT must fit `(k+1)` documents in a 1,024-token context
(§7.1). 256-token chunks serve RAFT *and* grounded QA. Closed-book QA (fineweb-edu only, §4.3)
may optionally use larger 512-token chunks for richer questions, since it carries no context
at train time — keep that as a separate small pass if desired.

`chunk_id` is a content hash, never an index. Persist a **`chunks_used`** set — a chunk is
never generated from twice.

---

## 4. Phase 2 — Generation recipes

Task mix (Group B) and difficulty (Group A) are configured here and enforced by the balancer
(§6.6). Domain mix mirrors the corpus: **sec 34.7% · case-law 34.2% · fineweb-edu 31.0%**.

### 4.1 Group B — task types

| Task | Share of QA SFT | Notes |
| --- | --- | --- |
| **Grounded QA** | 40% | Answer stated only in the passage. Also the RAFT source pool. |
| **Summarization** | 25% | Faithful short summary; passage in the user turn. |
| **Extraction** | 20% | Prose → JSON fields. |
| **Rewriting** | 15% | Restate in a target register without changing meaning. |

### 4.2 Grounded QA prompt (produces the shared pool)

```python
def qa_prompt(passage, n):
    return (
      f"Read the PASSAGE and write {n} diverse, high-quality question-answer pairs.\n"
      "Return JSON list; each item has:\n"
      '  "question" - self-contained; must NOT say "the passage/text/document/above"; '
      "name the company, case, or topic so it stands alone.\n"
      '  "quote"    - the EXACT verbatim span from the passage supporting the answer, '
      "copied character-for-character. No paraphrase, no ellipses.\n"
      '  "answer"   - short, direct, fully supported by the quote.\n'
      '  "difficulty" - "easy" | "medium" | "hard".\n'
      "Rules: vary difficulty; do NOT invent facts; no near-duplicate questions.\n\n"
      f"PASSAGE:\n{passage}")
```

Verify each quote is an exact substring **at generation time** (normalize whitespace); repair
or drop failures. An unverified quote is a fabricated citation — worse than none.

Summarize / extract / rewrite: batch ~5 passages per call to amortize cost; the passage lives
in the user turn (they are context-bearing by nature).

### 4.3 Closed-book slice — fineweb-edu ONLY

⚠️ A 125M/500M model cannot recall a single SEC filing's specific figure from one exposure —
closed-book training on document specifics teaches **fabrication**. Restrict closed-book QA to
**fineweb-edu** (general knowledge, reinforced across many docs). Generate it grounded, then
drop the passage at render time.

### 4.4 Group A — diversity and difficulty (the user's priority)

- **Self-Instruct** — seed the teacher with ~30 varied hand/sample questions per task and ask
  it to invent new, distinct ones in the same spirit. Drives breadth beyond what any single
  passage suggests.
- **Evol-Instruct** — evolve ~20% of QA into harder versions (add a constraint, an edge case,
  a multi-step reasoning requirement). Fills the hard tail so the model sees more than lookups.

```python
def evolve_prompt(passage, q):
    return ("Rewrite the QUESTION into ONE harder version — add a constraint, edge case, or "
            "multi-step reasoning — still answerable ONLY from the passage. Return JSON "
            '{"question","quote","answer"}.\n\n'
            f"PASSAGE:\n{passage}\n\nQUESTION: {q}")
```

### 4.5 RAFT abstention examples (extra generation)

For RAFT's golden-free and answerable-but-absent cases (§7.3), ask the teacher for **on-topic
questions the passage does NOT answer**; target string is exactly `not stated in the context`.

### 4.6 Record format

```json
{"chunk_id":"...","doc_id":"sec/shard-000.txt:12345","source":"sec","task":"qa",
 "question":"...","quote":"...","answer":"...","difficulty":"medium","answerable":true,
 "passage":"...","teacher":"gemini-3.1-flash","gen_tokens":{"in":612,"out":388}}
```

---

## 5. Phase 3 — Quality gauntlet (order matters: cheapest filter first)

The website's five filters, plus the **LLM-as-judge correctness gate** you specified. Expect
to keep **~50–60%** of raw output — that attrition is the quality, not waste.

### 5.1 Rule / format filters (cheapest)
Answer 8–1200 chars; question ≥ 8 chars; drop empty fields, template echoes ("Here are 5…"),
self-referential phrasing, malformed JSON. RAFT: cap quote ≤ 60 tokens (§7.1 budget).

### 5.2 Grounding / faithfulness
Fraction of answer content-words present in the passage. < 0.55 ⇒ likely invented; drop.
Borderline (0.40–0.55) ⇒ send to the judge (§5.4) rather than auto-dropping.

### 5.3 Deduplication — you asked for zero duplicates, so all three levels
1. **Exact** — hash of normalized question.
2. **Near-duplicate** — MinHash/LSH over question shingles, threshold 0.8.
3. **Semantic** — sentence-embedding cosine ≥ 0.90 (run last, on survivors).

```python
from sentence_transformers import SentenceTransformer
EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# encode questions, greedy drop any within 0.90 cosine of a kept one (or cluster + keep-1)
```

### 5.4 LLM-as-judge — the correctness gate (your requirement)

Gemini scores whether the answer **correctly and completely answers the question, supported by
the passage**. This is stronger than string overlap: it catches answers that are grounded but
wrong, or right but incomplete.

```python
JUDGE_SCHEMA = {"type":"object","properties":{
    "correct":{"type":"integer"},      # 1..5
    "grounded":{"type":"boolean"},
    "reason":{"type":"string"}}, "required":["correct","grounded","reason"]}

def judge_prompt(passage, q, a):
    return ("You are grading a synthetic training pair. Using ONLY the PASSAGE, decide:\n"
            "- correct (1-5): does the ANSWER correctly and completely answer the QUESTION?\n"
            "- grounded (bool): is every claim in the ANSWER supported by the PASSAGE?\n"
            "Be strict. A fluent but unsupported or incomplete answer scores low.\n\n"
            f"PASSAGE:\n{passage}\n\nQUESTION: {q}\nANSWER: {a}")
```

- **Temperature 0.** Batch to amortize where the API allows.
- **Keep rule:** `correct >= 4 AND grounded == True`.
- **Persist every score** (`data/sft/provenance/judge.jsonl`) — per-item, with `chunk_id`.
  This is what makes the dataset auditable and lets you tune the threshold without re-judging.
- Judge **after** dedup+grounding, not before — judging is the dominant token cost (§9), so
  never judge a pair you were going to drop anyway.

### 5.5 Difficulty & task balance
Confirm the realized easy/medium/hard (~40/40/20) and task mix (§4.1). Rebalance by dropping
over-represented cells; do not up-weight duplicates to fill under-represented ones.

### 5.6 Decontaminate against the eval set
After the eval set is carved (§8), remove any training pair within 13-gram or 0.90-embedding
overlap of an eval item. A leaked eval reports scores you did not earn.

---

## 6. Phase 4 — Diversity engineering & measurement

You asked for **very high diversity**. Do not assert it — **measure and report it**, and act
on the numbers.

### 6.1 Structural coverage
Stratify generation across **source × task × difficulty** = 3×4×3 = 36 cells. Track fill per
cell; steer generation toward thin cells rather than over-sampling easy ones.

### 6.2 Topic coverage
K-means (k≈50) over question embeddings. Report cluster sizes; a few giant clusters ⇒ the
teacher is repeating itself — inject Self-Instruct seeds for the sparse regions.

### 6.3 Diversity metrics (report these in the build report)

| Metric | Meaning | Target |
| --- | --- | --- |
| **distinct-2 / distinct-3** | unique bi/tri-gram ratio over questions | higher is better |
| **self-BLEU** | mean BLEU of each question vs the rest | **lower** is better |
| **embedding dispersion** | mean pairwise cosine *distance* of questions | higher is better |
| **cell entropy** | entropy over the 36 structural cells | near-uniform |

Report all four before and after filtering; filtering should *raise* diversity (dedup removes
the redundant mass).

---

## 7. Phase 5 — Assemble the two datasets (text-level)

### 7.1 RAFT context budget (1,024-token target — the tightest model)

```
system(~45) + question(~30) + (k+1)×doc + answer(quote+final ~90) + margin(~40) ≈ 1024
→ ~820 tokens for documents → k=2 distractors, ~256-token docs (the §3 `CHUNK_TOKENS`)
```

Measured in the 125M tokenizer (least efficient). Gemma's 256k vocab tokenizes the same text
into fewer tokens, so it fits with headroom. **k=2, 256-token chunks** — already the §3 chunk
size.

### 7.2 RAFT distractors — hard, with same-document exclusion
Build a 300k–500k-chunk embedding index (sampled, stratified). Distractors = top-k nearest
neighbours **from a different `doc_id`**.

⚠️ **The `doc_id` exclusion is mandatory on this corpus.** SEC filings and case-law opinions
restate the same facts across chunks; a "distractor" from the same document may contain the
answer and teaches the model that quote markers are decorative. This is the most likely subtle
RAFT bug here. Randomize the golden document's position among the distractors.

### 7.3 RAFT composition
`P = 0.8` golden present; `0.2` distractors-only (target `not stated in the context`); 5–15%
of golden-present are answerable-but-absent (golden there, but does not contain the answer).
`P = 1.0` is **not RAFT** — the model never learns retrieval can fail.

### 7.4 QA SFT assembly
Render the pool as chat pairs: grounded QA (passage in user turn), the fineweb-edu closed-book
slice (passage dropped), plus summarize/extract/rewrite. Quote-first answers are optional for
plain QA SFT but recommended (they engage span-copying).

### 7.5 Text-level output — no tokenization, no BOS decisions here
Write both datasets as chat JSONL with `messages`. **Do not** apply a chat template, add
special tokens, or tokenize — that is per-model and happens in the fine-tuning guides. Keeping
these as text is exactly what lets one build serve three different tokenizers.

```json
{"messages":[{"role":"system","content":"..."},
             {"role":"user","content":"..."},
             {"role":"assistant","content":"..."}],
 "meta":{"task":"qa","source":"sec","difficulty":"medium","mode":"raft",
         "golden_present":true,"chunk_id":"...","doc_id":"..."}}
```

---

## 8. Phase 6 — The frozen eval set (shared, build once)

⚠️ Carve this **before** decontamination (§5.6) and **before** finalizing training — a pair
trained on can never be a valid eval item.

- **500 items.** Eval size costs no GPU and decides whether results are resolvable; small eval
  sets are the top reason a study ends uninterpretable.
- **Quarantine at `doc_id` level** — if a document contributes an eval item, bar all its
  chunks from training **and** from the RAFT distractor pool.
- **Matched conditions** for RAFT — the *same question* rendered four ways:

| Condition | Golden | Distractors |
| --- | --- | --- |
| Clean | ✅ | none |
| Realistic | ✅ | k |
| Retrieval failure | ❌ | k |
| Closed-book | ❌ | none |

Matched conditions enable paired evaluation (far more powerful at fixed n). Also reserve
**10–20 hand-written probes**, including passages unlike this corpus, for OOD spot-checks.

- **Freeze it.** Write once, never regenerate. Shared across all three models unchanged.

---

## 9. Size decision & cost

### 9.1 Recommended sizes (shared across all three models)

Grounded in LIMA (1k curated can beat 100k noisy) and the SFT sweet spot, sized up for the
diversity target and because QLoRA-Gemma can absorb more; the 125M can subsample if it
over-forgets.

| Dataset | Curated size | Composition |
| --- | --- | --- |
| **QA SFT** | **15,000** | qa 6,000 · summarize 3,750 · extract 3,000 · rewrite 2,250; ~15% closed-book (fineweb-edu); difficulty 40/40/20 |
| **RAFT** | **10,000** | 8,000 golden-present (of which ~1,000 answerable-but-absent) · 2,000 distractors-only |
| **Eval** | **500** | matched-condition, `doc_id`-quarantined |

Same set for all three models — comparability is the point, and subsampling is free if needed.

### 9.2 Approximate Gemini 3.1 Flash cost (one-time, whole build)

To net ~25,000 kept pairs at ~55% keep rate ⇒ **~46,000 raw** generated + Evol-Instruct +
judging the survivors.

| Stage | Tokens (approx) |
| --- | --- |
| Generation (46k raw @ ~220 tok/pair, batched) | ~10 M |
| Evol-Instruct hard slice (~10k @ ~520 tok) | ~5 M |
| Abstention/unanswerable generation | ~2 M |
| **LLM-as-judge** (~32k survivors @ ~710 tok) | ~23 M |
| Embedding dedup / RAFT index | local, ~$0 (CPU/short GPU) |
| **Total** | **~40 M tokens** |

At Gemini 3.1 Flash rates (blended ~$0.40–0.70 / 1M tokens — **verify current pricing**):

> **≈ $20 – $30 total, one-time, shared across all three models.**

The judge is ~55% of the cost; dropping it to a 20% spot-audit would roughly halve the total,
at the price of your explicit correctness guarantee. I recommend keeping the full judge — $25
of teacher spend is trivial against the value of a verified-correct training set.

---

## 10. Definition of done

- [ ] `qa_sft.jsonl` (15,000) and `raft.jsonl` (10,000) written as chat JSONL — **text only**
- [ ] `eval.jsonl` (500) carved, `doc_id`-quarantined, matched conditions, frozen
- [ ] Zero duplicates: exact + MinHash + embedding (≥0.90) dedup passed
- [ ] Every kept pair judged `correct ≥ 4 AND grounded`; per-item scores persisted
- [ ] Diversity metrics (distinct-n, self-BLEU, dispersion, cell entropy) reported
- [ ] Decontaminated vs the eval set
- [ ] Realized composition (task/source/difficulty/mode) logged
- [ ] Total teacher token usage and cost recorded from `usage_metadata`
- [ ] Nothing tokenized or templated — rendering is deferred to the per-model guides

**Build once. Freeze. Hand the same three files to all three model tracks.**
