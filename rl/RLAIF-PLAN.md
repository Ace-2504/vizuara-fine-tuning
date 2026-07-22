# RLAIF Plan — reward model + PPO on the QA-SFT models

**Goal.** Classic RLAIF: train a **reward model** on the AI-labeled preferences, then **PPO**
the QA-SFT models against it, producing **new** versions (originals untouched):

```
preferences → reward model (BT) → PPO(slm-500m-sft)   → slm-500m-sft-rlaif
                                 → PPO(gemma-2-2b-sft) → gemma-2-2b-sft-rlaif
```

Reuses `preferences.jsonl`, `ft_data.py`, and the checkpoints.

---

## 1. Reward model — trained Bradley-Terry head (chosen over Gemini-as-reward)

**Why not Gemini-as-reward:** PPO needs thousands of fast reward evaluations; a Gemini call
per rollout adds ~$2–5 **and** 15 min–2 h of pure API latency that bottlenecks the loop. A
local reward model is instant, free during training, and standard.

**The reward model:** add a scalar value head to the **`slm-500m-sft`** backbone; train with the
**Bradley-Terry pairwise loss** on the 500 preferences:

```
loss = -log σ( r(prompt, chosen) − r(prompt, rejected) )
```

One shared RM scores response quality (model-agnostic), used by both PPO runs. Cheap (~10 min).
Report its **held-out pairwise accuracy** (should beat ~0.5; 500 pairs is small, so expect
modest) — a weak RM caps how much PPO can help, so this number gates the whole RLAIF path.

## 2. PPO

| | 500M | Gemma-2B |
| --- | --- | --- |
| Policy init | `slm-500m-sft` (full) | base + `gemma-2-2b-sft` (QLoRA) |
| Reward | shared BT reward model (local) | same |
| Reference (KL) | frozen SFT copy | frozen SFT |
| Library | TRL `PPOTrainer` | TRL `PPOTrainer` + peft |

**PPO loop:** sample response → reward from RM → PPO update with a **KL penalty** to the
reference (stops reward-hacking / drift). Prompts = the 500 from the preference set (or a fresh
prompt sample). A few PPO epochs.

**Hyperparameters (conservative — small data):** LR 1e-5 (full) / 1e-4 (LoRA) · KL coef 0.1–0.2
· batch/mini-batch small · 2–4 PPO epochs · bf16 · seed fixed.

## 3. Honest risk notes (read before running)
- **500 examples is thin** for a stable RM+PPO loop — treat this as a "does RLAIF move the
  needle" run, not a big alignment gain.
- **Gemma-2B QLoRA + PPO is the hard part** — TRL PPO + 4-bit + Gemma-2 soft-capping is finicky;
  this run carries the implementation risk and most of the cost/time.
- **Reward hacking:** watch for the policy gaming the RM (length, keywords). The KL penalty and
  a few held-out generations are the guardrails.
- **Reward model quality caps everything** (see §1).

## 4. Steps
1. Train the BT reward model on `preferences.jsonl`; report held-out pairwise accuracy. **Gate:**
   if accuracy ≈ 0.5, stop and reconsider (data too weak) before spending on PPO.
2. PPO `slm-500m-sft` against the RM (with KL to reference) → `slm-500m-sft-rlaif`.
3. PPO `gemma-2-2b-sft` (QLoRA) likewise → `gemma-2-2b-sft-rlaif`.
4. Compare vs SFT + DPO on sample prompts.

## 5. Cost & time (Modal)

| Run | GPU | Time | Cost |
| --- | --- | --- | --- |
| Reward model (BT, 500M) | L4 | ~10 min | ~$0.15 |
| 500M PPO | L4 | ~45 min | ~$0.60 |
| Gemma PPO (QLoRA) | A100 | ~2–3 h | ~$4–6 |
| **RLAIF total** | | **~2–3 h parallel** | **~$5–7** |

No Gemini cost at train time (local reward model). Runs on `ace-compoz`.

## 6. Definition of done
- [ ] Reward model trained; held-out pairwise accuracy reported (gate passed)
- [ ] `slm-500m-sft-rlaif` and `gemma-2-2b-sft-rlaif` saved (new versions; SFT untouched)
- [ ] KL-to-reference stayed bounded (no collapse); no obvious reward-hacking in samples
- [ ] Comparison vs SFT and DPO versions recorded
