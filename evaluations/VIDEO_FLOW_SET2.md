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

### 2:00–3:15 — Results: does alignment help, and who wins? (show the table + matrix)
- Per family, read DPO vs RLAIF from the pairwise matrix:
  - "125M: RLAIF vs DPO is Δ `[FILL]`, CI `[FILL]` → `[significant / not resolved]`."
  - "500M: Δ `[FILL]`, CI `[FILL]` → `[…]`."
  - "Gemma-2B: Δ `[FILL]`, CI `[FILL]` → `[…]`."
- State the honest headline. The likely story (confirm against your numbers): **the DPO–RLAIF gap
  is small and mostly *not* statistically resolved** on 500 preferences — say that plainly rather
  than crowning a winner the data doesn't support.
- Groundedness / false-abstention columns: "Alignment `[did / didn't]` change faithfulness or make
  models over-cautious — false-abstention stayed at `[FILL]`."

### 3:15–4:05 — Size effect and the base floor (show base-500m row)
- "SLM-500M base is in this set as the un-aligned floor: judged correctness `[FILL]` — every
  aligned model sits far above it, so alignment-on-top-of-SFT is clearly doing *something*."
- Cross-size read (within Set 2 only): "Among aligned models, `[500M / Gemma]` leads at `[FILL]`;
  the 125M aligned models reach `[FILL]` — `[close to / behind]` the larger models."

### 4:05–4:45 — Honest caveats (say them — this is what a mentor wants to hear)
- "**500 preference pairs is thin.** The reward model's held-out pairwise accuracy was `[FILL]`,
  which caps how much RLAIF can help — a weak reward model means a weak PPO signal."
- "Treat this as a *'does alignment move the needle'* run, not a large alignment gain."
- "Reward is reported as secondary only; the judge is the headline; and the judge itself isn't yet
  human-calibrated — the honest next step."

### 4:45–5:00 — Close
- "Set 2: alignment on top of SFT `[helps / barely moves]` grounded-QA quality; on this data
  **DPO and RLAIF are `[statistically indistinguishable / DPO wins / RLAIF wins]`**; and crucially
  the comparison is fair because RLAIF was judged by an **independent** model, not its own reward.
  Every claim is a paired significance test on a decontaminated held-out set."

---
**Pre-record checklist:** ① run the pipeline (see EVAL-HARNESS-REVIEW.md) so REPORT.md exists ·
② confirm the RLAIF rows show *circular — omitted* for reward — mention it on camera ·
③ replace every `[FILL]` with the real number + CI · ④ when the DPO–RLAIF gap isn't significant,
say "not resolved" — do not declare a winner the CI doesn't support.
