# DPO Plan — align the QA-SFT models on AI preferences

**Goal.** Direct Preference Optimization of the **QA-SFT** models on the 500-triplet preference
set, producing **new** versions (originals untouched):

```
slm-500m-sft   + preferences → slm-500m-sft-dpo
gemma-2-2b-sft + preferences → gemma-2-2b-sft-dpo
```

**No reward model** — DPO trains directly on `(prompt, chosen, rejected)`. Reuses `ft_data.py`
rendering/masking and the checkpoints already trained.

---

## 1. How DPO works here
DPO increases the model's log-prob margin between `chosen` and `rejected`, regularized toward a
frozen **reference** model (a copy of the SFT model) via a `beta` term — so it improves
preference alignment without drifting far from the SFT behavior. One forward pass each over
policy(chosen), policy(rejected), ref(chosen), ref(rejected).

## 2. Per-model setup

| | 500M | Gemma-2B |
| --- | --- | --- |
| Policy init | `slm-500m-sft` (full) | base + `gemma-2-2b-sft` adapter, merged, then a **new** LoRA |
| Reference | frozen copy of `slm-500m-sft` | frozen merged SFT model |
| Mode | full FT | QLoRA |
| Render | `render_custom` (no BOS, eos-fix) | `render_gemma` (system→user, BOS) |
| Max seq | 1024 | 2048 |

**Gemma note:** merge the SFT adapter into the base first (`merge_and_unload`), then DPO a
fresh LoRA on top — keeps the SFT behavior as the reference and adds a clean preference delta.

## 3. Hyperparameters (small data → gentle)

| | Value |
| --- | --- |
| beta | 0.1 |
| LR | 5e-6 (full) / 1e-4 (LoRA) — well below SFT |
| Epochs | 2–3 over 500 pairs |
| Effective batch | 8–16 |
| Precision | bf16 · grad clip 1.0 · seed fixed |

Implementation: TRL `DPOTrainer` (cleanest), or a ~40-line manual DPO loss reusing our
collator. Mask exactly as SFT (loss on the completion only).

## 4. Steps
1. Load `preferences.jsonl`; render `chosen`/`rejected` per model.
2. Load policy + frozen reference (same weights).
3. DPO train (§3); save new version.
4. **Quick check:** on a held-out slice, confirm `chosen` log-prob margin grew and outputs
   didn't degrade vs the SFT model (a few sample generations).

## 5. Cost & time (Modal)

| Run | GPU | Time | Cost |
| --- | --- | --- | --- |
| 500M DPO (full) | L4 | ~10 min | ~$0.15 |
| Gemma DPO (QLoRA) | L4 | ~20 min | ~$0.30 |
| **DPO total** | | **~20 min parallel** | **~$0.45** |

No Gemini cost (dataset already built). Runs on `ace-compoz` (or local 3060).

## 6. Definition of done
- [ ] `slm-500m-sft-dpo` and `gemma-2-2b-sft-dpo` saved (new versions; SFT untouched)
- [ ] Preference margin increased on held-out pairs
- [ ] Sample generations show no obvious regression vs SFT
- [ ] Reuses `preferences.jsonl` — no separate data
