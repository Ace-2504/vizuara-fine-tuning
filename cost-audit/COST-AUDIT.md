# Modal spend audit — invoiced vs. what the websites claim

**Scope.** All four Modal accounts, reconciled against every cost figure displayed on the model
sites, the arena and the e4 pretraining site. **No frontend was modified.**

**Sources.** Invoiced figures are transcribed from the four Modal dashboard *Ephemeral App
Breakdown* panels. Frontend figures are read out of `slm-frontends/lib/models.ts` and
`web-v1-extended/lib/model.ts`.

Companion workbook: `SLM-cost-audit.xlsx` — one sheet per account, plus Summary, Spend by phase
and Reconciliation.

---

## 1. The headline

**Total invoiced across four accounts: $116.39.** Against a stated $120, **$3.61 is unaccounted
for** — see §5.

**The suspicion was correct: the websites understate what this project cost, in every category.**

| | |
|---|---|
| Total invoiced | **$116.39** |
| Total attributable to work the sites describe | **$95.31** |
| What the sites actually claim for that work | **$65.75** |
| **Understated by** | **$29.56 (45%)** |
| Invoiced work never shown on any site | **$21.08** |

Those two rows reconcile exactly: $95.31 attributable + $21.08 never-shown = **$116.39**, the
grand total. Nothing is double-counted or dropped.

Two distinct problems, and they are worth separating:

1. **Under-counting** — every figure shown is lower than the invoice, because they were derived
   from wall-clock × an assumed GPU rate rather than read off the bill.
2. **Omission** — $19.19 of evaluation, the second-largest line in the whole project, appears
   nowhere.

---

## 2. Where the money actually went

| Phase | Invoiced | Share | On the sites? |
|---|---|---|---|
| 125M pretraining (4 legs) | **$70.14** | 60.3% | yes, but understated by $12.36 |
| Fine-tuning (SFT / RAFT) | $19.94 | 17.1% | yes, understated by $13.85 |
| **Evaluation** | **$19.19** | **16.5%** | **no — never shown** |
| Alignment (DPO / RLAIF) | $5.23 | 4.5% | yes, understated by $3.35 |
| Earlier 125M SFT study | $1.66 | 1.4% | no — separate project |
| Probes & smoke tests | $0.18 | 0.2% | no |
| Image builds / exports | $0.05 | 0.0% | no |
| **Total** | **$116.39** | 100% | |

**Pretraining the 125M from scratch is 60% of everything you spent.** Fine-tuning three model
families to five stages each cost $25.17 combined — barely a third of the pretraining bill. That
is the single most interesting economic fact in this project, and the sites currently obscure it
by understating the base.

**Evaluation cost more than all alignment work combined, by nearly 4×.** Judging 13 models on
500 held-out questions is not a rounding error, and it is invisible on every site.

---

## 3. Per-account

| Account | Invoiced | What it was used for |
|---|---|---|
| `ace-2504` | $30.36 | Original v1 pretraining, the extension leg, plus first-pass Gemma and 500M fine-tunes |
| `ace-compoz` | $30.29 | Final fine-tuning + all alignment + **the entire evaluation** |
| `singh1621` | $28.26 | The e4 continued-pretraining leg, essentially one job |
| `aceaynon2504` | $27.48 | The e2 continued-pretraining leg, essentially one job |

The two continued-pretraining legs each consumed a whole account's spend on a single run —
$55.74 between them, 48% of the project.

---

## 4. Line-by-line reconciliation

### 125M pretraining — the largest error

| Leg | Invoiced | Site shows | Diff |
|---|---|---|---|
| v1 | $11.54 | $10.71 | +$0.83 |
| Extension | $2.88 | $2.70 | +$0.18 |
| e2 | $27.47 | $26.31 | +$1.16 |
| **e4** | **$28.25** | **$18.06** | **+$10.19** |
| **Total** | **$70.14** | **$57.78** | **+$12.36** |

The first three legs are close because their costs were taken from run reports. **e4 is off by
56%** because it was derived as `8.6 h × $2.10/h` — the wall-clock figure understates what Modal
actually billed for that job.

Consequence: all four 125M model sites show a lineage total of ~$58.5 when the true figure is
~$70.9.

### Fine-tuning and alignment

| Work | Invoiced | Site shows | Ratio |
|---|---|---|---|
| Gemma SFT + RAFT | $16.66 | $4.56 | **3.7×** |
| Gemma RLAIF (PPO) | $4.23 | $1.26 | **3.4×** |
| Gemma DPO | $0.43 | $0.09 | **4.8×** |
| 500M SFT + RAFT | $2.67 | $1.23 | 2.2× |
| 500M RLAIF (PPO) | $0.33 | $0.28 | 1.2× |
| 500M DPO | $0.08 | $0.05 | 1.6× |
| 125M SFT + RAFT | $0.61 | $0.39 | 1.6× |
| 125M RLAIF (PPO) | $0.08 | $0.07 | 1.1× |
| 125M DPO | $0.02 | $0.01 | 2.0× |
| Reward model | $0.06 | $0.03 | 2.0× |

**The Gemma line is the worst.** `train_gemma.train` appears on *two* accounts — $11.41 on
`ace-2504` and $5.25 on `ace-compoz` — totalling $16.66 against $4.56 shown. The most likely
reading is that the `ace-2504` spend covers earlier or abandoned Gemma runs (this matches the
known history: the first Gemma PPO run diverged and had to be re-run with a firmer KL anchor,
and the earlier eval attempts hit A100 capacity and timeouts). **Retries are real money and the
sites only ever counted the successful run.**

### Never shown on any site — the "N/A" column

| Item | Invoiced |
|---|---|
| **Evaluation (set1 + set2, 13 models × 500 questions)** | **$19.19** |
| Earlier 125M SFT data-scaling study | $1.66 |
| Phase-0 capability probe | $0.18 |
| Image builds, exports, smoke tests | $0.05 |
| **Total** | **$21.08** |

---

## 5. The $3.61 gap

$120 − $116.39 = **$3.61 unaccounted**. The dashboard panels captured are titled *Ephemeral App
Breakdown*, which covers `modal run` invocations only. The gap is most likely one or more of:

- **Volume / storage** — the `ft-data` volume held ~5 GB of checkpoints for weeks
- **Deployed (non-ephemeral) apps**, if any were ever left up
- **Credit-versus-charge rounding**, or spend outside the panel's date window

This is a hypothesis, not a finding. To settle it, capture the **workspace-level total** or the
**storage line** from each usage page; the ephemeral breakdown alone cannot show it.

---

## 6. What this means

**The honest total for the 125M line is ~$70 of pretraining, not $57.79.** Every 125M model site
should read ~$70.9 rather than ~$58.5.

**The honest cost of the Gemma line is ~$21.3**, not the $3.29–$4.73 currently displayed — once
retries and its share of alignment are counted.

**The project's real cost structure is:** 60% pretraining, 17% fine-tuning, 16% evaluation, 5%
alignment. The sites currently present a picture in which pretraining is expensive and everything
else is nearly free; the truth is that evaluation rivals fine-tuning, and retries cost more than
the successful runs they replaced.

### If you decide to correct the sites

Three separate decisions, in descending order of importance:

1. **Fix the e4 leg** ($18.06 → $28.25) and therefore the 125M pretraining total
   ($57.79 → $70.14). Largest single error.
2. **Decide whether retries count.** If yes, Gemma's fine-tune line becomes $16.66. If you'd
   rather show only the run that produced the published weights, keep the current figure but say
   so explicitly — "cost of the final run, excluding earlier attempts".
3. **Decide whether to publish evaluation cost.** $19.19 is a real project cost and arguably the
   most interesting one, since almost nobody reports it.

### One caveat on attribution

Mapping Modal apps to site line-items is inference, not invoice metadata. App names are
unambiguous for pretraining (`modal_train_e4.train` → e4) and for alignment
(`train_dpo.train_125m` → 125M DPO). The one genuinely uncertain case is `train_gemma.train`
appearing on two accounts — I cannot tell from the dashboard alone which runs those were, only
that both were Gemma training. Everything above treats them as the same work item; if the
`ace-2504` spend was in fact a different experiment, Gemma's true fine-tune cost is $5.25, not
$16.66.
