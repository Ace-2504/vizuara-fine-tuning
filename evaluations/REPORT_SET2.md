# SET2 — alignment: DPO vs RLAIF (125M, 500M, Gemma-2B)

_Detailed evaluation report — this experiment stands alone; no numbers are compared against the other set._

_Headline metric = the correctness rating (1–5, rescaled to 0–1) of an independent LLM judge — Google's **gemini-3.1-flash-lite**, blind to which model wrote each answer — which is fair across the 125M / 500M / Gemma tokenizers and — unlike the reward model — is not the RLAIF training objective. token-F1 is reported alongside as a lexical proxy. Every model-vs-model claim uses a **paired bootstrap** on the same eval items (a difference counts only when the delta's 95% CI excludes 0); overlapping per-model CIs are never used as a test. All models scored on one decontaminated, held-out eval set (500 pair_ids × 4 RAFT conditions)._

## Key findings

- **Best model:** Gemma-2B SFT+DPO — 0.924 [0.906,0.942] judged correctness (clean).
- **Largest gap:** Gemma-2B SFT+DPO beats SLM-125M SFT+DPO by +0.871 [+0.845,+0.894].
- **500M lift:** base 0.000 → best (SLM-500M SFT+RLAIF) 0.443.
- ⚠️ **SLM-125M SFT+DPO collapsed** to ~2-word answers.
- **Word-overlap** turned out to be an unfair judging metric for the Gemma models, so I switched to AI judging as my main metric.

==============================================================================
## Experiment SET2 — 7/7 versions present

### Per-version (clean condition)

| model | judged correctness | groundedness | matches-ref | token-F1 | fabrication↓ | false-abstain↓ | median len | reward (within-family) |
|---|---|---|---|---|---|---|---|---|
| SLM-500M base _(base·few-shot)_ | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.025 [0.022,0.027] | 0.152 [0.120,0.182] | 0.000 [0.000,0.000] | 128w | 0.222 [0.163,0.283] |
| SLM-125M SFT+DPO | 0.054 [0.038,0.071] | 0.112 [0.086,0.140] | 0.042 [0.026,0.060] | 0.061 [0.048,0.075] | 0.306 [0.266,0.346] | 0.002 [0.000,0.006] | 2w | -0.463 [-0.516,-0.410] |
| SLM-125M SFT+RLAIF | 0.271 [0.238,0.304] | 0.446 [0.402,0.490] | 0.178 [0.144,0.212] | 0.245 [0.222,0.268] | 0.064 [0.042,0.086] | 0.000 [0.000,0.000] | 9w | circular — omitted |
| SLM-500M SFT+DPO | 0.279 [0.245,0.313] | 0.462 [0.418,0.504] | 0.198 [0.162,0.234] | 0.224 [0.204,0.246] | 0.066 [0.044,0.088] | 0.004 [0.000,0.010] | 15w | -0.162 [-0.211,-0.118] |
| SLM-500M SFT+RLAIF | 0.443 [0.406,0.480] | 0.648 [0.606,0.690] | 0.332 [0.290,0.374] | 0.367 [0.340,0.394] | 0.014 [0.004,0.026] | 0.000 [0.000,0.000] | 8w | circular — omitted |
| Gemma-2B SFT+DPO | 0.924 [0.906,0.942] | 0.906 [0.880,0.930] | 0.934 [0.910,0.956] | 0.149 [0.140,0.158] | 0.042 [0.026,0.060] | 0.162 [0.130,0.194] | 101w | 0.100 [0.042,0.158] |
| Gemma-2B SFT+RLAIF | 0.899 [0.878,0.918] | 0.856 [0.824,0.886] | 0.916 [0.892,0.940] | 0.115 [0.109,0.121] | 0.294 [0.256,0.334] | 0.006 [0.000,0.014] | 113w | circular — omitted |

_Reward is a SECONDARY signal from a 500M-backbone reward model — meaningful only WITHIN a family, never across families, and shown next to median length because reward models favour longer answers. The judge is the headline (caveat 6)._

### Lexical-F1 vs judge disagreement (caveat 4)

_Fraction of clean items where token-F1 (>0.5) and the judge (correct>0.5) disagree — higher = token-F1 is a worse proxy for that model, usually because it punishes correct paraphrases. This is why the judge, not F1, is the headline._

| model | F1↔judge disagreement |
|---|---|
| SLM-500M base | 0.000 |
| SLM-125M SFT+DPO | 0.036 |
| SLM-125M SFT+RLAIF | 0.136 |
| SLM-500M SFT+DPO | 0.150 |
| SLM-500M SFT+RLAIF | 0.192 |
| Gemma-2B SFT+DPO | 0.908 |
| Gemma-2B SFT+RLAIF | 0.878 |

### Ranking by judged correctness (clean) + paired significance

_Tuned models only; base models are the floor (table above), excluded here._

1. **Gemma-2B SFT+DPO** — 0.924 [0.906,0.942]
2. **Gemma-2B SFT+RLAIF** — 0.899 [0.878,0.918]
3. **SLM-500M SFT+RLAIF** — 0.443 [0.406,0.480]
4. **SLM-500M SFT+DPO** — 0.279 [0.245,0.313]
5. **SLM-125M SFT+RLAIF** — 0.271 [0.238,0.304]
6. **SLM-125M SFT+DPO** — 0.054 [0.038,0.071]

**Pairwise deltas** (A − B on the same items; ✓ = 95% CI excludes 0):

| A | B | Δ (A−B) | 95% CI | significant |
|---|---|---|---|---|
| Gemma-2B SFT+DPO | Gemma-2B SFT+RLAIF | +0.025 | [+0.001,+0.050] | ✓ |
| Gemma-2B SFT+DPO | SLM-500M SFT+RLAIF | +0.481 | [+0.440,+0.521] | ✓ |
| Gemma-2B SFT+DPO | SLM-500M SFT+DPO | +0.645 | [+0.609,+0.682] | ✓ |
| Gemma-2B SFT+DPO | SLM-125M SFT+RLAIF | +0.653 | [+0.619,+0.690] | ✓ |
| Gemma-2B SFT+DPO | SLM-125M SFT+DPO | +0.871 | [+0.845,+0.894] | ✓ |
| Gemma-2B SFT+RLAIF | SLM-500M SFT+RLAIF | +0.456 | [+0.412,+0.498] | ✓ |
| Gemma-2B SFT+RLAIF | SLM-500M SFT+DPO | +0.620 | [+0.580,+0.659] | ✓ |
| Gemma-2B SFT+RLAIF | SLM-125M SFT+RLAIF | +0.628 | [+0.591,+0.665] | ✓ |
| Gemma-2B SFT+RLAIF | SLM-125M SFT+DPO | +0.845 | [+0.818,+0.870] | ✓ |
| SLM-500M SFT+RLAIF | SLM-500M SFT+DPO | +0.164 | [+0.122,+0.207] | ✓ |
| SLM-500M SFT+RLAIF | SLM-125M SFT+RLAIF | +0.172 | [+0.129,+0.217] | ✓ |
| SLM-500M SFT+RLAIF | SLM-125M SFT+DPO | +0.389 | [+0.348,+0.429] | ✓ |
| SLM-500M SFT+DPO | SLM-125M SFT+RLAIF | +0.008 | [-0.035,+0.051] | — |
| SLM-500M SFT+DPO | SLM-125M SFT+DPO | +0.225 | [+0.187,+0.261] | ✓ |
| SLM-125M SFT+RLAIF | SLM-125M SFT+DPO | +0.217 | [+0.181,+0.252] | ✓ |

_Floor (few-shot base models): SLM-500M base 0.000 [0.000,0.000]._

### Illustrative examples (real generations)

**Correct answer that token-F1 badly under-scores (why the judge is the headline)** — _SLM-125M SFT+DPO_
- Q: How did the 1909 Westminster Gazette define the gyratory principle?
- reference: A method of directing vehicles into circular paths designed to prevent intersection.
- model answer: It's the name for a roundabout or through a system of one-way streets to avoid the need for one line of traffic to intersect another.
- token-F1 **0.15** · judge correct **4/5** · grounded True · 25 words

**Degenerate short output (model collapse)** — _SLM-125M SFT+DPO_
- Q: What are the two primary sources from which surfactant used in RDS treatment is derived?
- reference: One type is cow-derived and the other is synthetic.
- model answer: 240824665
- token-F1 **0.00** · judge correct **1/5** · grounded False · 1 words

**Over-abstention: refuses a question whose answer is present** — _SLM-125M SFT+DPO_
- Q: What is the Latin translation for common sense?
- reference: sênsus commúnis, which means common feelings of humanity.
- model answer: {Document 2] Unlike common sense, which is a word that is not stated in the context, common sense is a word that is not given to every person.
- token-F1 **0.11** · judge correct **1/5** · grounded False · 28 words

**Ungrounded / fabricated content (judge flags it)** — _SLM-125M SFT+DPO_
- Q: What are the two primary sources from which surfactant used in RDS treatment is derived?
- reference: One type is cow-derived and the other is synthetic.
- model answer: 240824665
- token-F1 **0.00** · judge correct **1/5** · grounded False · 1 words


==============================================================================
## Reproducibility

All versions scored on the same eval set (sha256 16e12b23bab8…). ✓
