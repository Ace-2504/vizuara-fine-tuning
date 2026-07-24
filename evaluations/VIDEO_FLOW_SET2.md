# VIDEO FLOW — Set 2 (alignment: DPO vs RLAIF)  ·  ~5 minutes

**Experiment question:** *After SFT, does preference alignment improve grounded-QA quality — and
which method wins, DPO or RLAIF — across the 125M, 500M, and Gemma-2B families?*

**Models compared (7):** SLM-500M base · SLM-125M SFT+DPO · SLM-125M SFT+RLAIF ·
SLM-500M SFT+DPO · SLM-500M SFT+RLAIF · Gemma-2B SFT+DPO · Gemma-2B SFT+RLAIF.
Scored on the **same frozen held-out set** as Set 1, but this is a **separate experiment** —
no numbers are compared across the two sets.

**What to have on screen:** `evaluations/eval_results/REPORT.md` (the "Experiment SET2" section) —
the per-version table and the pairwise significance matrix. Fill every `[FILL: …]` from your real
REPORT.md before recording.

---

### 0:00–0:45 — Frame the experiment (talking head)
- "Set 2 is the **alignment** experiment. Each model started from an SFT checkpoint and was then
  aligned on a preference dataset two different ways — **DPO** (direct preference optimization) and
  **RLAIF** (a reward model + PPO). The question is whether alignment helps grounded QA, and
  **DPO vs RLAIF head-to-head** across three model sizes."
- One line on the methods: "DPO trains directly on chosen-vs-rejected pairs; RLAIF trains a reward
  model on those preferences, then optimizes the policy against it with PPO."

### 0:45–2:00 — The one methodological subtlety that makes this rigorous (KEY MOMENT)
This is the strongest thing to say in the whole video — lead with it:
- "There's a trap here. RLAIF is **trained to maximize a reward model**. If I then *scored* these
  models with that same reward model, RLAIF would look better **by construction** — that's
  circular, not real improvement."
- "So the headline metric is an **independent LLM judge**, blind to which model produced each
  answer and completely decoupled from the RLAIF reward. In the report you'll see the reward model
  is explicitly **omitted for the RLAIF rows** — marked *circular — omitted* — and only shown as a
  weak secondary signal for DPO." (Point at that cell in the table.)
- "And because the preference set is small (500 pairs), I'm reporting **paired significance**, so
  we don't over-read tiny differences."

### 2:00–3:15 — Results: the winning method FLIPS with scale (show the table + matrix)
- The headline finding, stated up front: "There is **no universal winner** — the better alignment
  method depends on model size."
  - "**125M: RLAIF wins big**, Δ **+0.217** (CI [0.181, 0.252]), significant. DPO **collapsed** the
    125M — judged correctness **0.054**, degenerate ~2-word answers, fabrication 0.31. RLAIF
    reached **0.271**."
  - "**500M: RLAIF wins again**, Δ **+0.164** (CI [0.122, 0.207]), significant — RLAIF **0.443** vs
    DPO **0.279**."
  - "**Gemma-2B: DPO wins**, Δ **+0.025** (CI [0.001, 0.050]), significant — DPO **0.924** vs RLAIF
    **0.899**. It flips at the top end."
- The one-line takeaway: "**RLAIF for the small models, DPO for the big one.** On a thin 500-pair
  preference set, PPO's reward signal helped the tiny models where direct DPO destabilised them,
  but at 2B DPO's simplicity edges ahead."
- Faithfulness note: "Watch groundedness too — Gemma-RLAIF's fabrication jumped to **0.29** vs
  DPO's 0.04, so RLAIF made the 2B model less grounded even though its correctness stayed high."

### 3:15–4:05 — Size effect and the base floor (show base-500m row)
- "SLM-500M base is the un-aligned floor: judged correctness **0.000** (few-shot). Every aligned
  model sits above it, so alignment-on-top-of-SFT is clearly doing something."
- Cross-size read (within Set 2 only): "Scale dominates the method choice. **Gemma-2B leads at
  ~0.90–0.92**; the 500M models reach **0.28–0.44**; the 125M models **0.05–0.27**. A 2B model
  aligned either way beats every SLM by **+0.45 to +0.87** (all significant). The method matters,
  but size matters far more."

### 4:05–4:45 — Honest caveats (say them — this is what a mentor wants to hear)
- "**500 preference pairs is thin** — treat DPO-vs-RLAIF as directional. The 125M DPO collapse
  (2-word answers) shows DPO is fragile on tiny models with little preference data, not that DPO
  is bad in general."
- "The reward model is a 500M-backbone head — a **within-family, secondary** signal only; it's
  **omitted for the RLAIF rows** because they were trained to maximise it (that would be circular).
  The judge is the independent headline."
- "The judge itself isn't yet human-calibrated — the honest next step (the harness has the tool)."

### 4:45–5:00 — Close
- "Set 2: alignment helps, but the **winning method flips with scale — RLAIF for the 125M and 500M
  (both significant), DPO for Gemma-2B (significant)** — and DPO catastrophically destabilised the
  125M. Crucially the comparison is fair because RLAIF was judged by an **independent** model, not
  its own reward. Every claim is a paired significance test on a decontaminated held-out set."

---
**Pre-record checklist:** ① results are in `eval_results/REPORT.md` (SET2 section) · ② the RLAIF
rows show *circular — omitted* for reward — mention it on camera as the key fairness safeguard ·
③ the "method flips with scale" line is the memorable takeaway — lead and close with it ·
④ the 125M-DPO collapse (0.054, 2-word answers) is the striking cautionary stat.
