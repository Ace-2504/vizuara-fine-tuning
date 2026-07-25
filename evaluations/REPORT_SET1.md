# SET1 — base vs SFT vs RAFT (125M & Gemma-2B)

_Detailed evaluation report — this experiment stands alone; no numbers are compared against the other set._

_Headline metric = the correctness rating (1–5, rescaled to 0–1) of an independent LLM judge — Google's **gemini-3.1-flash-lite**, blind to which model wrote each answer — which is fair across the 125M / 500M / Gemma tokenizers and — unlike the reward model — is not the RLAIF training objective. token-F1 is reported alongside as a lexical proxy. Every model-vs-model claim uses a **paired bootstrap** on the same eval items (a difference counts only when the delta's 95% CI excludes 0); overlapping per-model CIs are never used as a test. All models scored on one decontaminated, held-out eval set (500 pair_ids × 4 RAFT conditions)._

## Key findings

- **Best model:** Gemma-2B RAFT — 0.975 [0.963,0.986] judged correctness (clean).
- **5 of 6** pairwise comparisons are statistically resolved (paired bootstrap, 95% CI excludes 0).
- **Largest gap:** Gemma-2B RAFT beats SLM-125M RAFT by +0.926 [+0.905,+0.946].
- **125M lift:** base 0.000 → best (SLM-125M SFT) 0.261.
- **GEMMA lift:** base 0.954 → best (Gemma-2B RAFT) 0.975.
- ⚠️ **SLM-125M RAFT over-abstains** — says 'not stated' on 84% of *answerable* clean questions.
- **token-F1 misleads for Gemma-2B SFT** (F1↔judge disagreement 0.45) — trust the judge, not word-overlap.
- **token-F1 misleads for Gemma-2B RAFT** (F1↔judge disagreement 0.97) — trust the judge, not word-overlap.

==============================================================================
## Experiment SET1 — 6/6 versions present

### Per-version (clean condition)

| model | judged correctness | groundedness | matches-ref | token-F1 | fabrication↓ | false-abstain↓ | median len | reward (within-family) |
|---|---|---|---|---|---|---|---|---|
| SLM-125M base _(base·few-shot)_ | 0.000 [0.000,0.000] | 0.068 [0.046,0.090] | 0.000 [0.000,0.000] | 0.019 [0.017,0.022] | 0.110 [0.084,0.140] | 0.000 [0.000,0.000] | 123w | n/a |
| SLM-125M SFT | 0.261 [0.228,0.293] | 0.456 [0.412,0.500] | 0.178 [0.144,0.212] | 0.229 [0.206,0.251] | 0.062 [0.042,0.084] | 0.000 [0.000,0.000] | 8w | n/a |
| SLM-125M RAFT | 0.049 [0.033,0.067] | 0.108 [0.082,0.134] | 0.038 [0.022,0.056] | 0.078 [0.070,0.086] | 0.002 [0.000,0.006] | 0.836 [0.802,0.868] | 5w | n/a |
| Gemma-2B base _(base·few-shot)_ | 0.954 [0.937,0.970] | 0.980 [0.966,0.992] | 0.938 [0.916,0.958] | 0.511 [0.488,0.534] | 0.002 [0.000,0.006] | 0.024 [0.012,0.038] | 20w | n/a |
| Gemma-2B SFT | 0.964 [0.950,0.976] | 0.976 [0.962,0.988] | 0.942 [0.920,0.962] | 0.569 [0.544,0.594] | 0.042 [0.026,0.060] | 0.000 [0.000,0.000] | 18w | n/a |
| Gemma-2B RAFT | 0.975 [0.963,0.986] | 0.994 [0.986,1.000] | 0.960 [0.942,0.976] | 0.145 [0.139,0.152] | 0.008 [0.002,0.016] | 0.008 [0.002,0.016] | 95w | n/a |

_Reward is a SECONDARY signal from a 500M-backbone reward model — meaningful only WITHIN a family, never across families, and shown next to median length because reward models favour longer answers. The judge is the headline (caveat 6)._

### Lexical-F1 vs judge disagreement (caveat 4)

_Fraction of clean items where token-F1 (>0.5) and the judge (correct>0.5) disagree — higher = token-F1 is a worse proxy for that model, usually because it punishes correct paraphrases. This is why the judge, not F1, is the headline._

| model | F1↔judge disagreement |
|---|---|
| SLM-125M base | 0.000 |
| SLM-125M SFT | 0.128 |
| SLM-125M RAFT | 0.038 |
| Gemma-2B base | 0.480 |
| Gemma-2B SFT | 0.454 |
| Gemma-2B RAFT | 0.970 |

### Ranking by judged correctness (clean) + paired significance

_Tuned models only; base models are the floor (table above), excluded here._

1. **Gemma-2B RAFT** — 0.975 [0.963,0.986]
2. **Gemma-2B SFT** — 0.964 [0.950,0.976]
3. **SLM-125M SFT** — 0.261 [0.228,0.293]
4. **SLM-125M RAFT** — 0.049 [0.033,0.067]

**Pairwise deltas** (A − B on the same items; ✓ = 95% CI excludes 0):

| A | B | Δ (A−B) | 95% CI | significant |
|---|---|---|---|---|
| Gemma-2B RAFT | Gemma-2B SFT | +0.011 | [+0.000,+0.024] | — |
| Gemma-2B RAFT | SLM-125M SFT | +0.714 | [+0.679,+0.748] | ✓ |
| Gemma-2B RAFT | SLM-125M RAFT | +0.926 | [+0.905,+0.946] | ✓ |
| Gemma-2B SFT | SLM-125M SFT | +0.703 | [+0.667,+0.738] | ✓ |
| Gemma-2B SFT | SLM-125M RAFT | +0.914 | [+0.892,+0.935] | ✓ |
| SLM-125M SFT | SLM-125M RAFT | +0.212 | [+0.175,+0.248] | ✓ |

_Floor (few-shot base models): SLM-125M base 0.000 [0.000,0.000], Gemma-2B base 0.954 [0.937,0.970]._

### RAFT breakdown — SLM-125M RAFT

| condition | token-F1 | judged correctness | abstain rate |
|---|---|---|---|
| clean | 0.078 [0.070,0.086] | 0.049 [0.033,0.067] | 0.836 [0.802,0.868] |
| realistic (with distractors) | 0.126 [0.116,0.138] | 0.222 [0.191,0.255] | 0.002 [0.000,0.006] |
| retrieval_failure (abstain expected) | 0.681 [0.643,0.719] | 0.673 [0.632,0.713] | 0.658 [0.618,0.698] |
| closed_book (parametric recall — contamination-sensitive) | 0.124 [0.112,0.136] | 0.032 [0.022,0.042] | 0.000 [0.000,0.000] |

- grounding gap (realistic − closed_book F1): +0.003 [-0.011,+0.016] — — how much having the right document helps.
- distractor gap (clean − realistic F1): -0.049 [-0.060,-0.037] ✓
- correct abstention (retrieval_failure): 0.658 [0.618,0.698]
- closed_book F1 0.124 vs clean 0.078 — closed_book is a parametric-recall probe (contamination-sensitive), not grounding. ⚠️ closed_book ≈ clean: score may reflect MEMORISED training text, not QA skill

### RAFT breakdown — Gemma-2B RAFT

| condition | token-F1 | judged correctness | abstain rate |
|---|---|---|---|
| clean | 0.145 [0.139,0.152] | 0.975 [0.963,0.986] | 0.008 [0.002,0.016] |
| realistic (with distractors) | 0.142 [0.136,0.149] | 0.956 [0.940,0.971] | 0.006 [0.000,0.014] |
| retrieval_failure (abstain expected) | 0.087 [0.084,0.089] | 0.787 [0.751,0.823] | 0.836 [0.804,0.868] |
| closed_book (parametric recall — contamination-sensitive) | 0.059 [0.054,0.064] | 0.129 [0.103,0.157] | 0.000 [0.000,0.000] |

- grounding gap (realistic − closed_book F1): +0.083 [+0.077,+0.089] ✓ — how much having the right document helps.
- distractor gap (clean − realistic F1): +0.003 [+0.000,+0.005] ✓
- correct abstention (retrieval_failure): 0.836 [0.804,0.868]
- closed_book F1 0.059 vs clean 0.145 — closed_book is a parametric-recall probe (contamination-sensitive), not grounding.

### Illustrative examples (real generations)

**Correct answer that token-F1 badly under-scores (why the judge is the headline)** — _SLM-125M SFT_
- Q: What primary advice is given regarding the reliability of flood maps?
- reference: You should verify the date of the map because many are outdated and underestimate risk.
- model answer: It is important to note that the information on flood maps is based on outdated data.
- token-F1 **0.13** · judge correct **4/5** · grounded True · 16 words

**Degenerate short output (model collapse)** — _SLM-125M SFT_
- Q: What are the two primary sources from which surfactant used in RDS treatment is derived?
- reference: One type is cow-derived and the other is synthetic.
- model answer: Wider lungs.
- token-F1 **0.00** · judge correct **1/5** · grounded False · 2 words

**Over-abstention: refuses a question whose answer is present** — _SLM-125M RAFT_
- Q: What are the two primary sources from which surfactant used in RDS treatment is derived?
- reference: One type is cow-derived and the other is synthetic.
- model answer: not stated in the context
- token-F1 **0.13** · judge correct **1/5** · grounded False · 5 words

**Ungrounded / fabricated content (judge flags it)** — _SLM-125M SFT_
- Q: What specific limitation did the judge place on Dr. Jefferson regarding the Cleveland Clinic's surgical methods for Eggemeyer?
- reference: Dr. Jefferson was precluded from presenting evidence that the use of two plates by physicians at the Cleveland Clinic was improper.
- model answer: The judge precluded Dr. Jefferson from presenting any evidence that: (1) Dr. Jefferson was an expert witness; (2) Dr. Pugh was an expert witness; (3) Dr. Jefferson was an expert witness; and (4) Dr. Pugh was an expert witness.
- token-F1 **0.33** · judge correct **1/5** · grounded False · 39 words


==============================================================================
## Reproducibility

All versions scored on the same eval set (sha256 16e12b23bab8…). ✓
