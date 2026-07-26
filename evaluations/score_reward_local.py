"""Post-hoc reward scoring on the local GPU — fills the Gemma reward gap fairly.

The two Gemma set2 versions were evaluated locally (eval_local.py) after the Modal
spend limit, and that runner never loaded the reward model, so their per-item
`scores.reward` is None and the report prints "n/a". Reward is a PURE POST-HOC
scorer (eval.py:230): it reads `user + "\n" + resp` — both stored verbatim in every
per-item record — so it can be computed now without regenerating anything.

Fairness is proven, not assumed: before touching Gemma, --validate rescores items
that Modal DID score (base-500m, slm-500m-sft-dpo, slm-125m-sft-dpo) with this local
scorer and reports the difference vs the stored logits. Only if that agrees do we
score Gemma. Same checkpoint (pulled from the ft-data volume), same libs
(torch 2.4.1+cu121 / transformers 4.46.3 — the versions in every manifest), same
bf16 dtype, same 1024-token truncation, batch size 1 — an exact transcription of
eval.py's reward().

    python evaluations/score_reward_local.py --validate          # prove local == Modal
    python evaluations/score_reward_local.py --score             # score gemma (prints, no write)
    python evaluations/score_reward_local.py --score --write     # patch JSONs (backs up first)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from stats import mean_ci  # noqa: E402  (same n=5000/seed=0 as eval.py's bootstrap_ci)

RM_PATH = os.path.join(ROOT, "migrate", "checkpoints", "reward-500m")
OUT = os.path.join(ROOT, "eval_results")
GEMMA_VERSIONS = ["gemma-2-2b-sft-dpo", "gemma-2-2b-sft-rlaif"]
VALIDATE_VERSIONS = ["base-500m", "slm-500m-sft-dpo", "slm-125m-sft-dpo"]
DEV = "cuda"


def load_rm():
    tok = AutoTokenizer.from_pretrained(RM_PATH)
    rm = AutoModelForSequenceClassification.from_pretrained(
        RM_PATH, num_labels=1, torch_dtype=torch.bfloat16).to(DEV).eval()
    rm.config.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")  # as eval.py:167
    return tok, rm


def reward_fn(tok, rm, user: str, resp: str) -> float:
    # exact transcription of eval.py:230-236
    t = tok(user + "\n" + resp, return_tensors="pt", truncation=True, max_length=1024)
    t = {k: v.to(DEV) for k, v in t.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        return rm(**t).logits.squeeze().item()


def load(version: str, judged: bool):
    p = os.path.join(OUT, f"{version}.judged.json" if judged else f"{version}.json")
    return p, json.load(open(p, encoding="utf-8"))


def validate(tok, rm, per_version: int = 50):
    """Rescore Modal-scored items locally; report the drift.

    bf16 on a different GPU (L4 -> RTX 3060) picks different SDPA/GEMM kernels, so
    individual logits wobble. What decides fairness is not the per-item wobble but
    whether it is BIASED: the table shows a 3-decimal mean over 500 items, so a
    random-sign wobble of |e| <= 0.08 shifts that mean by <= 0.08/sqrt(500) ~ 0.004,
    invisible next to CI half-widths of ~0.05. A systematic offset would not cancel,
    so the tight criterion is the SIGNED mean drift (< 0.01), with a loose sanity cap
    on the worst single item (< 0.08)."""
    print(f"validation: {per_version} items x {len(VALIDATE_VERSIONS)} Modal-scored versions")
    worst, worst_bias = 0.0, 0.0
    for v in VALIDATE_VERSIONS:
        _, blob = load(v, judged=True)
        items = [it for it in blob["per_item"]
                 if it.get("cond") == "clean" and it["scores"].get("reward") is not None]
        step = max(1, len(items) // per_version)
        diffs = []
        for it in items[::step][:per_version]:
            local = reward_fn(tok, rm, it["user"], it["resp"])
            diffs.append(local - it["scores"]["reward"])
        mx = max(abs(d) for d in diffs)
        bias = sum(diffs) / len(diffs)
        worst, worst_bias = max(worst, mx), max(worst_bias, abs(bias))
        print(f"  {v:22s} n={len(diffs):3d}  max|local-modal|={mx:.5f}  "
              f"signed mean={bias:+.5f}  mean|.|={sum(abs(d) for d in diffs)/len(diffs):.5f}")
    ok = worst < 0.08 and worst_bias < 0.01
    print(f"validation {'PASSED' if ok else 'FAILED'} "
          f"(worst item {worst:.5f} < 0.08; worst signed mean {worst_bias:.5f} < 0.01)")
    return ok


def score_gemma(tok, rm, write: bool):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for v in GEMMA_VERSIONS:
        rewards = {}
        t0 = time.time()
        for judged in (True, False):
            path, blob = load(v, judged)
            for i, it in enumerate(blob["per_item"]):
                if it.get("cond") != "clean":
                    continue
                key = it["pair_id"]
                if key not in rewards:                      # score once, reuse for both files
                    rewards[key] = reward_fn(tok, rm, it["user"], it["resp"])
                it["scores"]["reward"] = rewards[key]
            vals = [rewards[it["pair_id"]] for it in blob["per_item"] if it.get("cond") == "clean"]
            blob["result"]["conditions"]["clean"]["reward"] = list(mean_ci(vals))
            # provenance: the eval ran locally without the RM; reward added post hoc.
            blob["result"]["manifest"]["reward_backfill"] = {
                "reward_model": "ft-data:checkpoints/reward-500m",
                "how": "post-hoc rescoring of stored responses (eval.py reward(), bf16, batch 1)",
                "device": torch.cuda.get_device_name(0),
                "validated_against_modal_scored_versions": VALIDATE_VERSIONS,
                "timestamp": stamp,
            }
            if write:
                bak = path + ".pre-reward.bak"
                if not os.path.exists(bak):
                    shutil.copy2(path, bak)
                tmp = path + ".tmp"
                json.dump(blob, open(tmp, "w", encoding="utf-8"), indent=2)
                os.replace(tmp, path)
        m = mean_ci(list(rewards.values()))
        print(f"{v}: n={len(rewards)}  reward = {m[0]:+.3f} [{m[1]:+.3f},{m[2]:+.3f}]"
              f"  ({time.time()-t0:.0f}s){'  [WRITTEN]' if write else '  [dry run]'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if not (a.validate or a.score):
        ap.error("pass --validate and/or --score")
    tok, rm = load_rm()
    print(f"reward model loaded from {RM_PATH} on {torch.cuda.get_device_name(0)}")
    if a.validate:
        if not validate(tok, rm) and a.score:
            print("refusing to score gemma: validation failed"); return
    if a.score:
        score_gemma(tok, rm, write=a.write)


if __name__ == "__main__":
    main()
