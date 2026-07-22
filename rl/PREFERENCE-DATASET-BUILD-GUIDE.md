# Preference Dataset Build Guide — RLAIF/DPO triplets

**Goal.** Build **500 preference triplets** `(prompt, chosen, rejected)` for aligning the
**QA-SFT** models (`slm-500m-sft`, `gemma-2-2b-sft`). One shared, off-policy dataset serves
both DPO and RLAIF, and both models.

**Teacher:** Gemini 3.1 Flash (falls back to `gemini-3.1-flash-lite` — full flash 404s on this
key). **Reuses** `teacher.py`, the corpus, and the embedding-dedup from the SFT build.

---

## 1. Why off-policy + shared

You chose **Gemini writes both responses** (off-policy). Consequence: the dataset is
model-agnostic, so one 500-triplet file trains DPO and the RLAIF reward model for *both*
models. Simplest and cheapest, and it matches "one dataset of 500". (Trade-off: it aligns to
Gemini's notion of quality, not each model's own failure modes — acceptable for a fast first
pass.)

---

## 2. Pipeline (step by step)

```
sample 500 prompts → generate (strong, weak) response pair → CLEAN → DEDUP → LLM-JUDGE verify
    → keep only judge-confirmed triplets → format as chat JSONL
```

### 2.1 Prompts (500)
Draw QA-style prompts from the corpus, **held out from the SFT training prompts** so alignment
isn't measured on trained data. Reuse the machinery in `chunker.py`/`build.py`:
- take grounded + closed-book question prompts (legal/financial), mixed difficulty;
- 500 distinct prompts after dedup (over-sample ~650 raw to survive dedup + judge).

### 2.2 Generate the pair (Gemini, 1 call/prompt)
Ask the teacher for a **strong** answer and a **plausibly weaker** answer in one JSON call:

```python
def pref_prompt(question, passage=None):
    ctx = f"CONTEXT:\n{passage}\n\n" if passage else ""
    return (ctx + f"QUESTION: {question}\n\n"
        "Return JSON with two fields:\n"
        '  "chosen":   a correct, complete, well-structured answer.\n'
        '  "rejected": a PLAUSIBLE but clearly worse answer — e.g. less accurate, missing a '
        "key point, vague, or over-hedged. It must still look like a real attempt, not gibberish.\n"
        "Both answer the same question; only quality differs.")
```
Schema: `{"chosen": str, "rejected": str}`.

### 2.3 Clean (instruction 7)
Drop empty/malformed, length-out-of-bounds, `chosen == rejected`, and pairs where `rejected`
is trivially broken (too short / non-answer). Enforce the chat schema.

### 2.4 Deduplicate (instruction 7)
- **Exact:** hash of the normalized prompt.
- **Semantic:** sentence-embedding cosine ≥ 0.90 over prompts (reuse `all-MiniLM-L6-v2`, the
  `HF_HUB_DISABLE_IMPLICIT_TOKEN` fix from the SFT build). Guarantees 500 distinct prompts.

### 2.5 LLM-judge verification (instruction 7 — the key quality gate)
A **separate** Gemini call re-reads each triplet **blind** (chosen/rejected order shuffled) and
must confirm the intended chosen is genuinely better:

```python
JUDGE_SCHEMA = {"type":"object","properties":{
  "better":{"type":"string"},        # "A" | "B"
  "margin":{"type":"integer"},       # 1..5 confidence
  "reason":{"type":"string"}}, "required":["better","margin","reason"]}
```
**Keep rule:** judge picks the intended `chosen` AND `margin >= 3`. Drop ties, reversals, and
low-margin pairs. This removes cases where Gemini's "weaker" answer wasn't actually worse —
without it, DPO/RM training on wrong-labeled pairs actively harms the model.

### 2.6 Format
Chat JSONL with a `prompt` and both completions (rendered per-model later by `ft_data.py`):

```json
{"prompt":[{"role":"system","content":"..."},{"role":"user","content":"..."}],
 "chosen":"...","rejected":"...",
 "meta":{"source":"sec","judge_margin":4,"prompt_id":"..."}}
```

Output: `rl/data/preferences.jsonl` (gitignored).

---

## 3. Cost & time

| Stage | Tokens | Cost | Time |
| --- | --- | --- | --- |
| Generate pairs (~650 raw @ ~600 tok) | ~0.4M | ~$0.4 | ~10 min |
| LLM-judge (~600 survivors @ ~750 tok) | ~0.45M | ~$0.45 | ~10 min |
| Dedup (local embeddings) | — | $0 | ~2 min |
| **Total** | ~0.85M | **~$1–2** | **~15–30 min** |

(At the measured ~$1/1M blended rate; concurrency can cut wall-clock further. Gemini key
confirmed funded.)

---

## 4. Definition of done
- [ ] 500 triplets, all judge-confirmed (`chosen` wins, margin ≥ 3)
- [ ] Zero duplicate prompts (exact + embedding)
- [ ] Per-item provenance (judge margin) persisted
- [ ] Written as `rl/data/preferences.jsonl`, ready for DPO ([DPO-PLAN.md](DPO-PLAN.md)) and the
      RLAIF reward model ([RLAIF-PLAN.md](RLAIF-PLAN.md))
