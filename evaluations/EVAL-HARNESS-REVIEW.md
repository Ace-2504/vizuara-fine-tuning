# Evaluation harness — frontier-standard review & upgrade

Audit of the fine-tuning eval harness (`eval.py` + `eval_report.py`) against how top labs
(OpenAI / Anthropic) run model evals, and the concrete changes made to close the gaps.
**Two separate experiments** are supported end-to-end: **set1** (base vs SFT vs RAFT, on the
125M and Gemma families) and **set2** (the DPO vs RLAIF alignment comparison). There is no
cross-set comparison anywhere — each set gets its own independent report.

## Verdict before the upgrade: solid B− research harness

Genuinely good bones: a frozen held-out set with **chunk-level decontamination** (bugs/05),
**deterministic greedy decoding**, **bootstrap 95% CIs on every metric**, a **matched
four-condition RAFT design** (clean / realistic / retrieval_failure / closed_book), and
per-item outputs saved. Above typical hobby evals. It was let down by five things:

| # | Gap vs frontier standard | Severity |
|---|---|---|
| 1 | **Comparisons done by CI overlap**, not a paired test. Overlapping per-model CIs is not a significance test. | High |
| 2 | **Reward model is circular & family-biased.** It's a BT head on `slm-500m-sft`; RLAIF *optimises it*, so reporting it as quality is circular for RLAIF and favours the 500M family cross-model. | High |
| 3 | **Correctness was lexical only** (token-F1 punishes paraphrase; `fabrication` checked digit-strings only). | High |
| 4 | **set1/set2 not encoded** — a flat 15-model list grouped by base family. | Medium |
| 5 | No CI on RAFT gaps; no **false-abstention** metric; no **run manifest**; aggressive uniform decoding unaudited. | Medium |

## What changed (all in this repo, `evaluations/`)

New / rewritten modules:

- **`experiments.py`** *(new)* — `SET1` (6) and `SET2` (7) as first-class lists plus per-version
  `META` (family, is_base, is_raft, `reward_circular`, sft_parent, label). This is the single
  source of truth for both the runner and the report.
- **`stats.py`** *(new)* — `paired_delta_ci(A, B)`: bootstraps `mean(A) − mean(B)` on the items
  both models were scored on, with **shared resample indices**, so per-item difficulty cancels.
  A difference is called only when the delta's own 95% CI excludes 0. Also `paired_gap_ci` for
  the RAFT condition gaps (previously bare point estimates).
- **`judge_eval.py`** *(new)* — the fair headline metric. A **Gemini judge, blind to which model
  produced each answer**, scores every saved response for `correct` (1–5), `grounded` (bool),
  and `matches_ref` (bool), with abstention handled explicitly so the four RAFT conditions share
  one judge. Runs **locally** over the saved generations (no GPU, no key on Modal), **resumable
  and budget-aware** exactly like `build_prefs.py`. Being independent of the reward model, it
  **breaks the RLAIF circularity**.
- **`eval.py`** *(upgraded)* — now (a) persists **per-item scores + full context** so the judge
  and paired bootstrap can run offline; (b) writes a **run manifest** (eval-set sha256, decoding
  params, model source, torch/transformers versions, git commit, timestamp); (c) makes the
  **reward model optional** (`--no-reward`) and never a headline; (d) adds a **false-abstention**
  metric (penalises abstaining when the answer is present); (e) hardens the numeric fabrication
  check (normalises `1,000`/`5%`). Adds `--set set1|set2` to run an experiment.
- **`eval_report.py`** *(rewritten)* — emits **one report per experiment**: a per-version headline
  table (judged correctness, groundedness, token-F1, fabrication↓, false-abstain↓, reward as a
  clearly-labelled secondary that is **omitted for RLAIF**), a **full pairwise paired-delta matrix**
  with significance, and the RAFT four-condition breakdown with **paired gap CIs**. Also a
  reproducibility check that all versions share the same eval-set hash. Writes `REPORT.md` +
  `comparisons.json`.

## New run order (nothing executed yet — run on your go)

```bash
# 1) generate + deterministic metrics on Modal, per experiment
modal run evaluations/eval.py --set set1                 # set1's 6 versions
modal run evaluations/eval.py --set set2                 # set2's 7 versions
#    (set1 needs no reward metric: add --no-reward to skip loading the RM)

# 2) pull results to a local folder
modal volume get ft-data /eval ./eval_results

# 3) LLM-judge them locally (resumable; a credit-out just re-runs)
python evaluations/judge_eval.py ./eval_results --set all
#    cheaper judged estimate on a fixed identical subsample:
#    python evaluations/judge_eval.py ./eval_results --sample 200

# 4) build both experiment reports
python evaluations/eval_report.py ./eval_results         # -> REPORT.md, comparisons.json

# 5) (optional) calibrate the judge against human labels — do once
python evaluations/judge_calibration.py emit ./eval_results --n 50 --out calib.csv
#    ...fill human_correct / human_grounded in calib.csv, then:
python evaluations/judge_calibration.py score calib.csv

# 6) (optional) sharp head-to-head for the key comparison, position-bias-controlled
python evaluations/judge_pairwise.py ./eval_results slm-500m-sft-dpo slm-500m-sft-rlaif
```

## Judge cost (so there are no surprises)

Non-RAFT versions are judged on 500 clean items; RAFT versions on 2,000 (four conditions).
set1 ≈ 6,000 judge calls, set2 ≈ 3,500 — on `gemini-3.1-flash-lite` this is small but non-zero,
and fully resumable. Use `--sample N` (identical first-N pair_ids per condition, so the
comparison stays paired) to cut cost when iterating.

## Quality-caveat fixes (implemented)

Every caveat from the review now has code behind it:

| Caveat | Fix implemented |
|---|---|
| 1 · judge not calibrated | `judge_calibration.py` — emit a 50-item human-labelling sheet, then score exact/within-1/Pearson/kappa agreement (judge vs human). |
| 2 · pointwise only | `judge_pairwise.py` — direct A-vs-B judging run BOTH orderings; a win counts only if consistent across the swap (position bias surfaced as ties). Bootstrap CI on the net preference. |
| 4 · token-F1 is a proxy | Report now shows the judge's semantic `matches-ref`, plus a **F1↔judge disagreement** table exposing where lexical scoring misleads. |
| 5 · closed_book contamination | Relabelled as *parametric recall — contamination-sensitive*, kept out of any grounding headline, and auto-**flagged** when closed_book ≈ clean (likely memorisation). |
| 6 · reward not cross-family | Reward labelled *within-family only*, a **median-length** column sits beside it (reward models favour length), and it is never a headline; omitted for RLAIF. |
| 7 · uniform decoding | `--decoding plain\|constrained`; **plain greedy is now the default** (penalties removed), `constrained` kept as a one-command sensitivity ablation. Recorded in the manifest. |
| 8 · bases score at the floor | Un-tuned custom bases (125M/500M) are now **few-shot** prompted (exemplars from training data, never the eval set) so they show real capability, and are shown as a **floor line excluded from the significance matrix**. (base-gemma is instruction-tuned, so it stays zero-shot.) |

## Still open (honest)

- The calibration **requires you to actually label the 50 items** — the tool is built, the human
  pass is yours to run. Below ~0.8 within-1 agreement, treat judged numbers cautiously.
- A **larger/neutral reward model** would remove the family bias entirely; for now reward is a
  within-family diagnostic only and the judge carries the headline.
- An **HTML report** like the `slm-125m-eval` site, if you want to publish these.
