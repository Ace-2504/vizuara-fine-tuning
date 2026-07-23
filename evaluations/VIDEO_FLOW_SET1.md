# VIDEO FLOW — Set 1 (base vs SFT vs RAFT)  ·  ~5 minutes

**Experiment question:** *Does fine-tuning a tiny 125M model for grounded QA actually work,
how does SFT compare to RAFT, and how does the 125M stack up against a 20×-larger Gemma-2B?*

**Models compared (6):** SLM-125M base · SLM-125M SFT · SLM-125M RAFT · Gemma-2B base ·
Gemma-2B SFT · Gemma-2B RAFT. All scored on the **same frozen 2,000-item held-out set**.

**What to have on screen:** `evaluations/eval_results/REPORT.md` (the "Experiment SET1" section) —
the per-version table, the ranking + pairwise matrix, and the RAFT breakdowns. Fill every
`[FILL: …]` from your actual REPORT.md before recording.

---

### 0:00–0:40 — Frame the experiment (talking head)
- "This is one of two evaluation experiments. Set 1 asks a clean question: **on grounded
  question-answering, does fine-tuning help, and does the RAFT recipe beat plain SFT** — measured
  the same way for a 125M model and for Gemma-2B."
- One sentence on the task: "Every item gives the model documents and a question; the model must
  answer **using only those documents**, or say *not stated in the context* when the answer isn't
  there."

### 0:40–1:40 — Why the evaluation is trustworthy (this is what earns marks)
Show the top of the report + the "Reproducibility" line. Hit these four points fast:
1. **Held-out + decontaminated** — the exact eval passages were never trained on (chunk-level
   quarantine), so this measures generalization, not memorization.
2. **Deterministic decoding** — greedy, fixed settings recorded in the run manifest → reproducible.
3. **Fair headline metric** — an **independent LLM judge** scores correctness and groundedness,
   *blind* to which model produced each answer. This avoids token-F1's habit of punishing correct
   paraphrases, and it's comparable across the 125M and Gemma tokenizers.
4. **Real statistics** — every model-vs-model claim uses a **paired bootstrap** on the same items;
   I only call a difference real when the *difference's* 95% CI excludes zero. Overlapping
   per-model bars are never treated as a test.

### 1:40–2:50 — The headline result (show the per-version table + ranking)
- Read the ranking by judged correctness. Expected shape: **base models near the floor**
  (they can't follow the grounded-QA format), **SFT and RAFT far above them.**
  - "SLM-125M base scores `[FILL]` — essentially the floor, as expected for an un-tuned model."
  - "After SFT it jumps to `[FILL]`, a **significant** gain (Δ `[FILL]`, CI `[FILL]`)."
- The cross-family comparison: "Does a 125M model close the gap to Gemma-2B on *this* grounded
  task? On the shared eval, SLM-125M SFT vs Gemma-2B SFT is Δ `[FILL]`, CI `[FILL]` —
  `[significant / not resolved]`." Say the honest version of whatever the number is.
- Groundedness column: "Fine-tuning also raised faithfulness — groundedness went `[FILL]→[FILL]`."

### 2:50–4:00 — SFT vs RAFT: the real point of Set 1 (show the RAFT breakdown tables)
- "RAFT trains the model to handle **retrieved documents that may not contain the answer** — so
  the interesting question isn't just clean accuracy, it's **robustness and abstention.**"
- Walk the four RAFT conditions for the RAFT models:
  - **clean vs realistic** (distractor gap): "Adding distractor documents costs RAFT only
    Δ `[FILL]` F1 — `[small/large]`, showing `[good/poor]` distractor robustness."
  - **retrieval_failure** (correct abstention): "When the answer is genuinely absent, RAFT abstains
    `[FILL]%` of the time — this is the behavior SFT never learns."
  - **grounding gap** (realistic − closed_book): "`[FILL]`, i.e. having the right document
    `[does/doesn't] measurably help`."
- Head-to-head: "On clean correctness, SFT vs RAFT is Δ `[FILL]`, CI `[FILL]`. So RAFT buys
  `[abstention/robustness]` at `[no cost / a small cost]` to clean accuracy — that trade is the
  Set-1 takeaway."

### 4:00–4:40 — Honest caveats (say these out loud — mentors reward it)
- "The reference answers are teacher-generated, so token-F1 is a proxy; that's exactly why the
  **LLM judge is the headline** and F1 is shown alongside, not alone."
- "`closed_book` rewards answering from memory, so it's contamination-sensitive — I read it as a
  parametric-recall probe, not as grounding."
- "The LLM judge isn't yet calibrated against human labels — the honest next step."

### 4:40–5:00 — Close
- "So Set 1: fine-tuning takes a 125M model from the floor to `[FILL]` judged correctness; RAFT
  adds abstention and distractor-robustness at `[no/low]` cost; and against Gemma-2B the 125M is
  `[competitive on X / behind on Y]`. Every claim here is backed by a paired significance test on
  a decontaminated held-out set."

---
**Pre-record checklist:** ① run the pipeline (see EVAL-HARNESS-REVIEW.md) so REPORT.md exists ·
② replace every `[FILL]` with the real number + CI · ③ keep the report's SET1 tables on screen
during 1:40–4:00 · ④ if a gap is *not* significant, say "not resolved," don't oversell it.
