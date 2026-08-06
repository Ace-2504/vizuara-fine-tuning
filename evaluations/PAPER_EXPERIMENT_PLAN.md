# SLM paper — experiment execution plan (parallel-heavy, deadline-free)

Companion to `slm-arena-15/experiment_playbook.html`. This plan reorders the playbook around
the **true dependency graph** and **who can actually do each piece**, given that (a) there is
no deadline — you command when things run — and (b) Claude has direct access to the eval data,
so most of what the playbook assigned to "You" (because the mentor was remote) can be done by
Claude here.

**Nothing in this plan runs until you give the go-signal for that specific item.**

---

## 0. The one scope decision to make first: 13 or 15 models?

The playbook was written for **13** models. The arena is now **15** (we added 500M QA-SFT and
500M RAFT, evaluated identically on the same 500 questions and 0–10 rubric). Recommendation:
**do the paper on all 15.** They were produced and scored the same way, they fill the 500M
lineage (QA-SFT and RAFT stages that were previously missing), and they strengthen the
fabrication story with two more alignment-relevant points. Cost of including them is ~15% more
rows everywhere and one cheap re-judge (below). Every number below assumes 15; scoping back to
13 is a one-line filter if you prefer.

---

## 1. Priority verdict — is the playbook's order right?

The playbook order (P0 → P1 → stats 3/4/7 → P3 second-judge → P4 repro → P5 seeds → P6) is
**correct for a deadline-driven run**, where value-per-hour is what matters and the long GPU
job (seeds) is the deadline risk to defer. **You removed the deadline and asked for heavy
parallelism, which changes two things:**

1. **Promote Exp 5 (multi-seed training) from "optional/last" to "launch FIRST."** Its only
   real downside in the playbook was deadline risk ("3–6 days GPU, skip if the clock gets
   tight"). With no clock, that downside vanishes and its long wall-clock makes it the natural
   thing to start on day 0 so it cooks in the background while everything else happens. It stays
   *low value-per-hour*, but *high value-per-elapsed-day* when run in parallel.
2. **Most of Tier-1 is not a user task here.** P0 (export), Exp 3 (CIs), Exp 4 (McNemar),
   Exp 7 (bias) are pure analysis of data Claude already has — Claude produces them directly.
   The playbook made them "you" only because the mentor couldn't see your files. So they are
   **not sequential user chores**; they run as a Claude track in parallel with your grading.

Everything else in the playbook's reasoning holds: Exp 1 (human calibration) is the true
critical path (only you can do it, can't be rushed), and P6 (submission) is genuinely last.

**Net verdict: keep the tiering (1 mandatory, 2 booster), but execute by dependency + parallel
track, not strictly top-to-bottom, and move seeds to the front as a background job.**

---

## 2. Dependency graph

```
                 ┌─────────────────────────────────────────────┐
   (independent) │ EXP 5  multi-seed DPO/RLAIF  → GPU, days     │  launch first, runs in bg
                 └─────────────────────────────────────────────┘
                 ┌─────────────────────────────────────────────┐
   (independent) │ EXP 6  repro repo (eval set, prompt, code)   │  assemble anytime
                 └─────────────────────────────────────────────┘

   P0  EXPORT  ── the 15×500 master table ──┐  (root; blocks the rest)
        │                                    │
        ├──────────────┬───────────────┬─────┴────────┐
        ▼              ▼               ▼              ▼
   EXP 1 human    EXP 3 CIs +     EXP 4 McNemar   EXP 7 judge-bias
   calibration    significance    fabrication     checks
   (YOU, long)    (Claude, mins)  (Claude, mins)  (Claude, mins)
        │              │               │              │
        └── EXP 2 second judge (API run + Claude analysis) ──┘
                                   │
                                   ▼
                          P6  submission hygiene   (last)
```

---

## 3. Prerequisite fix before P0 (cheap, Claude, needs go-signal)

**Persist rubric parts for the 2 new 500M models.** `judge_500m_new.py` stored only
`{score, grounded}`; a complete Step-0 table needs `{correctness, completeness, groundedness,
clarity}` too. Fix = re-run that judge with the parts persisted (same method, ~500 calls,
~10–15 min, a few cents of Gemini). One-time. After this, all 15 models have identical columns.

*(Alternative if you want zero extra API spend: export the 2 new models with the 4 parts blank
and only their total+grounded. Not recommended — it weakens Exp 1/7 for those two.)*

---

## 4. The experiments, in execution order

### WAVE A — launch on day 0, in parallel (three independent starts)

**A1 · EXP 5 — multi-seed re-training** · owner: **you command, Claude drives** · GPU, ~days
- **DECIDED SCOPE:** **3 seed runs of every non-base model** — all 12 (QA-SFT, RAFT, DPO, RLAIF
  across 125M / 500M / Gemma). Base models excluded (500M & Gemma imported; 125M base = a $70
  pretraining lineage, out of scope). This is broader than Claude's original "DPO+RLAIF only"
  suggestion — the user opted for the full non-base sweep.
- **Budget:** **$75.51** (3 × the $25.17 one-seed cost of all 12 non-base stages), funded by the
  **3 alternate Modal profiles (~$90 combined)**. Comfortably under budget. Note $75.51 is the
  *conservative* figure — the Gemma SFT/RAFT per-seed cost is anchored to an invoice that included
  retries, so clean seeded runs may come in ~$10 lower (~$65).
- **Parallel allocation:** run **one full seed per Modal profile** — profile 1 → seed A (all 12),
  profile 2 → seed B, profile 3 → seed C. Each profile carries ~$25.17 (< its ~$30 share), no
  single training run is split across profiles, and the three seeds run **concurrently**, cutting
  wall-clock ~3×. (Watch the Gemma runs specifically — $21.34 of each seed is Gemma, so keep one
  Gemma stage set per profile.)
- Recipes: `rl/train_dpo.py`, `rl/train_ppo.py`, `rl/train_reward.py`, and the SFT/RAFT trainers
  `rl/train_125m.py` / `train_500m.py` / `train_gemma.py` — add a `--seed` and vary it per profile.
- Re-score each seed's output with the judge; report **mean ± sd** across the 3 seeds, especially
  for the fabrication rate (ties Exp 5 → Exp 4) and the DPO/RLAIF deltas.
- Runs unattended on Modal. Launch first because it's the longest pole. **Real $ + GPU — waits
  for your explicit go.**
- Output: "robustness across seeds" section + error bars on the alignment/fabrication numbers.

**A2 · EXP 6 — reproducibility release** · owner: **Claude assembles, you approve** · no GPU
- Assemble an anonymised repo: the 500-question eval set, the judge prompt + rubric, the
  scoring/generation code (`eval.py`/`eval_local.py`/`gen_500m_new.py`/rubric judge), and a
  couple of checkpoints. Claude writes the README + dataset card.
- Kept **anonymous** for double-blind; real links revealed only after acceptance.
- Independent of P0 — can be built anytime. Output: repro section + anonymised URL.

**A3 · P0 prerequisite re-judge** (§3 above) — do this so P0 is complete.

### WAVE B — the root export (Claude, fast, after A3)

**B1 · P0 — master score table** · owner: **Claude** · ~automated
- One row per (model × question): `model, size, stage, question_id, source,
  correctness, completeness, groundedness, clarity, total(0–10), grounded(bool),
  fabricated(bool), answer_text, gold_text`. 15 × 500 ≈ **7,500 rows**, CSV + JSON.
- Built by joining `eval_results/<model>.judged.json` (resp, ref, question/context, source,
  fabrication) with `rubric10.jsonl` + the re-judged 2-new-model parts (rubric parts, grounded),
  keyed on the shared `pair_id` so every question lines up across all 15 models.
- This is the single artifact everything downstream reads. Output: the supplementary data file.

### WAVE C — three parallel tracks off P0

**C1 · EXP 1 — human calibration** · owner: **YOU (only you)** · 6–8 h · CRITICAL PATH
- Claude prepares a **blind grading sheet**: a stratified sample of **~150 answers (10 per
  model × 15)**, balanced across case-law / SEC / web, with the AI judge's score **hidden** —
  each row shows question + gold + model answer and blank fields for your 0–5/0–2/0–2/0–1 and a
  fabricated y/n.
- You grade blind. Optionally a friend grades ~40 for a human ceiling.
- Claude computes weighted **Cohen's κ**, correlation, and **% within 1 point** (`judge_calibration.py`).
  Target κ ≥ 0.6, r ≥ 0.7. Output: new **"Judge calibration"** section.
- Start as soon as the sheet exists — it's the one thing that can't be compressed at the end.

**C2 · EXP 3 + 4 + 7 — the free statistics** · owner: **Claude** · minutes, ₹0
- **Exp 3:** bootstrap 95% CIs per model + paired Wilcoxon signed-rank between neighbours with
  multiple-comparison correction (`stats.py` / `eval_report.py` already do bootstrap CIs).
- **Exp 4:** McNemar on the paired `fabricated` flag, base-vs-RLAIF and base-vs-DPO, per family;
  p-value + effect size. This is the paper's headline result made bulletproof.
- **Exp 7:** from P0 — answer-length-vs-score correlation (verbosity bias), note one-answer-at-a-
  time scoring (order bias), and self-preference check (does the Gemini judge over-reward
  Gemma/aligned outputs). One methods paragraph.
- All three read only P0 → run the moment P0 lands, in parallel with C1. Output: "Statistical
  significance" section + methods bias paragraph.

**C3 · EXP 2 — second judge** · owner: **you provide a key, Claude runs + analyses** · ~$4
- **DECIDED: the second judge is `gpt-5.6-luna`** (OpenAI) — the closest tier-match to the primary
  Gemini 3.1 Flash-Lite judge, and a different vendor (kills single-judge bias + the Gemma/Gemini
  same-vendor self-preference concern). Verified pricing: **$0.20 in / $1.20 out per 1M tokens**.
- ✅ **Key smoke-tested 2026-08-02** (OPENAI_API_KEY in `slm-arena-15/.env.local`): luna reachable,
  returns valid JSON. Note: reasoning model — set a generous `max_completion_tokens` in the runner.
- Re-score the same answers (all 500, or a 150 subset) with **identical rubric/prompt** — only the
  judge model changes. Claude writes the runner (`judge_pairwise.py` scaffold); you supply the key.
- **Billing:** OpenAI has **no reliable free API tier** (unlike Gemini), so you'll add a payment
  method and a **one-time ~$5 minimum credit** — which covers the whole run (full 15×500 ≈ $4;
  a 150-subset is cents). Your Claude Max plan does **not** cover any API usage (verified), so
  Anthropic wouldn't have been free either — no cost reason to prefer it here.
- Claude computes Spearman/Kendall rank correlation and whether the leaderboard order holds.
  Output: "ranking stable across two independent judges" line.
- Can overlap fully with C1/C2.

### WAVE D — finish (after Tier-1 lands, and seeds return)

**D1 · fold in Exp 5 seed results** (when the background runs finish) → error bars on alignment.
**D2 · P6 — submission hygiene** · owner: **you + Claude** · Springer LNNS template, page limit,
double-blind scrub (no name / no arena URL / no personal GitHub in the PDF), similarity check.
Genuinely last.

---

## 5. What runs in parallel (the heavy-parallelism answer)

- **Day 0, three things start at once:** Exp 5 (GPU, background), Exp 6 (repo assembly), and the
  P0 prerequisite re-judge → P0 export.
- **Once P0 exists:** your Exp 1 grading (days, human) runs *simultaneously* with Claude's entire
  Exp 3/4/7 stats bundle (minutes) and the Exp 2 second-judge run. None of these block each other.
- **Only two things are truly serial:** P0 must precede the analyses, and P6 must be last.
- **The critical path is Exp 1 (your grading)** — everything Claude does finishes long before you
  finish grading, so your grading time is the schedule. Seeds (Exp 5) run alongside and are folded
  in whenever they return.

---

## 6. Ownership summary (given Claude's data access)

| Item | Playbook said | Actually here |
|---|---|---|
| P0 export | You, 1 h | **Claude** (has the data) |
| Exp 1 grading | You, 6–8 h | **You** (only human judgment) — Claude preps the blind sheet + computes κ |
| Exp 2 second judge | You + Claude | **Claude runs**; you supply the second API key |
| Exp 3 / 4 / 7 | Claude | **Claude** (unchanged) |
| Exp 5 seeds | You launch | **You command** (real GPU $); Claude writes seed configs + drives Modal |
| Exp 6 repro | You + Claude | **Claude assembles**, you approve what's public |
| P6 submission | You + Claude | **You + Claude** |

Your real, irreducible workload: **grade ~150 answers (Exp 1)**, **say go on the seed runs and
supply a second-judge key**, and **make the final submission calls.** Everything else Claude does.

---

## 7. Decisions — LOCKED (2026-08-02)
1. **Models:** ✅ **15** (includes 500M QA-SFT + RAFT).
2. **Exp 5 seeding:** ✅ **3 seed runs of all 12 non-base models**, ≈ **$75.51**, on the **3 Modal
   profiles (~$90)**, one full seed per profile in parallel. Base models excluded.
3. **Exp 2 second judge:** ✅ **`gpt-5.6-luna`** (OpenAI, $0.20/$1.20 per 1M). Needs a ~$5 OpenAI
   credit top-up (no free tier; Max plan doesn't cover API).
4. Still open: nothing else — awaiting only your **go-signal per item** (they still don't run until
   you command each). Recommended first move: launch the 3 seed runs (longest pole) + let Claude
   build P0 and prep your blind grading sheet in parallel.
