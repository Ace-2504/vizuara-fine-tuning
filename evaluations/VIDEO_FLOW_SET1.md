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
- **Lead with the size story.** "Judged correctness, clean condition. Gemma-2B is near-ceiling
  everywhere — base **0.954**, SFT **0.964**, RAFT **0.975** — while the 125M is far lower:
  base **0.000**, SFT **0.261**, RAFT **0.049**."
- The cross-family gap: "Gemma-2B SFT beats SLM-125M SFT by **+0.703** (95% CI [0.667, 0.738]),
  significant. A 20×-larger model is simply in a different league on grounded QA — the 125M
  does **not** close that gap."
- The 125M's own gain from SFT: "For the tiny model, SFT is what unlocks the format — base
  **0.000 → SFT 0.261** (significant), and groundedness rose **0.068 → 0.456**. Un-tuned, even
  few-shot, it scores zero — it can't follow grounded-QA instructions at all."

### 2:50–4:00 — The twist: RAFT *hurt* the 125M (show the RAFT breakdown tables)
- "Here's the surprise. RAFT trains the model to abstain when the answer isn't retrieved — but
  on the 125M it **backfired**: SFT **0.261 → RAFT 0.049**, a **−0.212** drop (CI [0.175, 0.248],
  significant). RAFT made the small model *worse*."
- Why — over-abstention: "On the RAFT breakdown, the 125M-RAFT abstains on **83.6%** of the
  *clean, answerable* questions — it learned to say 'not stated in the context' even when the
  answer is right there. It over-generalised the abstention lesson because it's too small to tell
  'absent' from 'present'."
- Contrast with Gemma: "The 2B model handles RAFT fine — Gemma-RAFT vs Gemma-SFT is only
  **+0.011, not significant** (a tie), and it abstains correctly **83.6%** of the time only when
  the answer is genuinely missing (retrieval_failure), while answering the clean ones. Its
  grounding gap is **+0.083** (significant) — having the right document measurably helps."
- Takeaway line: "So RAFT is a **capacity-dependent** recipe: safe abstention for a 2B model,
  but a net loss for a 125M that can't afford the caution."

### (optional) — token-F1 would have lied to you
- "One methodology point worth 15 seconds: if I'd ranked by token-F1, Gemma-RAFT looks *terrible*
  — F1 **0.145**. But the judge says **0.975** correct. F1 punished its verbose paraphrasing; the
  F1↔judge disagreement is **0.97**. This is why the headline metric is an independent LLM judge,
  not word-overlap."

### 4:00–4:40 — Honest caveats (say these out loud — mentors reward it)
- "The reference answers are teacher-generated, so token-F1 is a proxy; that's exactly why the
  **LLM judge is the headline** and F1 is shown alongside, not alone."
- "`closed_book` rewards answering from memory, so it's contamination-sensitive — I read it as a
  parametric-recall probe, not as grounding."
- "The LLM judge isn't yet calibrated against human labels — the honest next step."

### 4:40–5:00 — Close
- "So Set 1, three findings: **(1)** SFT takes the 125M from **0.00 to 0.26** — it's what makes a
  tiny model usable at all. **(2)** RAFT is capacity-dependent — a **−0.21** loss on the 125M
  (it over-abstains on 84% of answerable questions) but a harmless tie on Gemma. **(3)** Scale
  dominates everything: Gemma-2B beats the 125M by **+0.70**, significant, on every recipe. And
  token-F1 would have inverted the Gemma ranking — the independent judge is doing real work here.
  Every number is a paired significance test on a decontaminated held-out set."

---
**Pre-record checklist:** ① results are in `eval_results/REPORT.md` (SET1 section) · ② keep that
table on screen during 1:40–4:00 · ③ the RAFT-hurts-125M over-abstention number (83.6% abstain on
clean) is the most striking single stat — land it clearly · ④ base-gemma is `gemma-2-2b-it`
(already instruction-tuned), so its 0.954 "base" score is a ceiling, not a fair untuned floor —
say so if asked.
