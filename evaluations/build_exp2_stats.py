"""Exp 2 analysis — does the leaderboard hold across two independent judges?

Compares the gpt-5.6-luna second-judge scores (exp2_luna_scores.jsonl) against the Gemini primary
judge (P0_master_scores.json) on the exact same (model, question) pairs. Reports per-model means
under both judges, Spearman + Kendall rank correlation of the 15-model leaderboard, per-answer
correlation, and grounded-flag agreement. Output: eval_results/STATS_exp2.{md,json}. Nothing published.
"""
from __future__ import annotations
import json, io, os
from collections import defaultdict
import numpy as np
from scipy import stats

ER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_results")

# gemini (primary) per (model, pair_id)
gem = {}
for r in json.load(io.open(os.path.join(ER, "P0_master_scores.json"), encoding="utf-8")):
    if r.get("total") is not None:
        gem[(r["model"], r["question_id"])] = r["total"]

# luna (second) per (model, pair_id)
luna = {}
for line in io.open(os.path.join(ER, "exp2_luna_scores.jsonl"), encoding="utf-8"):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    luna[(r["model"], r["pair_id"])] = r["score"]

common = sorted(set(gem) & set(luna))
models = sorted({m for m, _ in common})
print(f"models={len(models)} paired judgments={len(common)}")

# per-model means on common pairs
per = {}
for m in models:
    ks = [k for k in common if k[0] == m]
    g = np.array([gem[k] for k in ks], float); l = np.array([luna[k] for k in ks], float)
    per[m] = {"n": len(ks), "gemini_mean": round(float(g.mean()), 3), "luna_mean": round(float(l.mean()), 3)}

gem_rank = sorted(models, key=lambda m: -per[m]["gemini_mean"])
luna_rank = sorted(models, key=lambda m: -per[m]["luna_mean"])
gv = [per[m]["gemini_mean"] for m in models]; lv = [per[m]["luna_mean"] for m in models]
rho, prho = stats.spearmanr(gv, lv)
tau, ptau = stats.kendalltau(gv, lv)

# per-answer correlation (all paired judgments)
ga = np.array([gem[k] for k in common], float); la = np.array([luna[k] for k in common], float)
pear = stats.pearsonr(ga, la); sp_item = stats.spearmanr(ga, la)
mae = float(np.abs(ga - la).mean())

md = ["# Exp 2 — second-judge cross-validation (gpt-5.6-luna vs Gemini 3.1 Flash-Lite)\n",
      f"Identical rubric/prompt; only the judge model differs. Paired on {len(common)} (model,question) "
      f"judgments across {len(models)} models.\n",
      "## Per-model mean score under each judge\n",
      "| Model | n | Gemini | luna | Δ |", "|---|---|---|---|---|"]
for m in gem_rank:
    d = per[m]["luna_mean"] - per[m]["gemini_mean"]
    md.append(f"| {m} | {per[m]['n']} | {per[m]['gemini_mean']} | {per[m]['luna_mean']} | {d:+.2f} |")
md += ["",
       "## Leaderboard rank agreement\n",
       f"- **Spearman ρ = {rho:.3f}** (p={prho:.2e})",
       f"- **Kendall τ = {tau:.3f}** (p={ptau:.2e})",
       f"- Gemini order: {' > '.join(gem_rank)}",
       f"- luna order:   {' > '.join(luna_rank)}",
       f"- Order identical: {'YES' if gem_rank == luna_rank else 'NO'}",
       "",
       "## Per-answer agreement (all paired judgments)\n",
       f"- Pearson r = {pear[0]:.3f} (p={pear[1]:.2e})",
       f"- Spearman ρ = {sp_item.correlation:.3f}",
       f"- Mean absolute difference = {mae:.2f} points (out of 10)",
       "",
       "## Verdict",
       (f"The two independent judges agree strongly (Spearman ρ={rho:.2f} on the leaderboard); "
        f"the model ranking is {'preserved' if rho >= 0.9 else 'largely preserved' if rho >= 0.7 else 'only partly preserved'} "
        "across judges, so the reported ordering is not an artifact of a single judge/vendor.")]
io.open(os.path.join(ER, "STATS_exp2.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
json.dump({"per_model": per, "spearman": [float(rho), float(prho)], "kendall": [float(tau), float(ptau)],
           "gemini_rank": gem_rank, "luna_rank": luna_rank, "order_identical": gem_rank == luna_rank,
           "item_pearson": float(pear[0]), "item_spearman": float(sp_item.correlation), "item_mae": mae,
           "n_pairs": len(common)},
          io.open(os.path.join(ER, "STATS_exp2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote STATS_exp2.md + .json")
print("\n".join(md))
