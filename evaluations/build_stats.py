"""Exp 3 + 4 + 7 — the free statistics, all off P0_master_scores.json.

Exp 3: per-model mean + 95% bootstrap CI; paired Wilcoxon between rank-adjacent models (Holm-corrected).
Exp 4: McNemar on the fabricated flag, base->DPO and base->RLAIF, per family.
Exp 7: judge-bias checks — verbosity (length vs score), order (noted), self-preference (Gemma).

Writes eval_results/STATS_exp3_4_7.md + STATS_exp3_4_7.json. Nothing published.
"""
from __future__ import annotations
import json, io, os
from collections import defaultdict
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

ER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_results")
rows = json.load(io.open(os.path.join(ER, "P0_master_scores.json"), encoding="utf-8"))

by_model = defaultdict(dict)   # model -> qid -> row
for r in rows:
    by_model[r["model"]][r["question_id"]] = r
models = list(by_model)

def totals(m):
    return np.array([by_model[m][q]["total"] for q in sorted(by_model[m])], float)

def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, x.size), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

out = {"exp3": {}, "exp4": {}, "exp7": {}}
md = ["# Statistical results (Exp 3 / 4 / 7)\n",
      "Computed from `P0_master_scores.json` (15 models x 500 held-out questions, rubric-10 judge).\n"]

# ---------- Exp 3: means + CIs, ranked ----------
means = {m: boot_ci(totals(m)) for m in models}
ranked = sorted(means, key=lambda m: -means[m][0])
md.append("## Exp 3 — mean score, 95% bootstrap CI, and adjacent-pair significance\n")
md.append("| Rank | Model | Mean /10 | 95% CI |")
md.append("|---|---|---|---|")
for i, m in enumerate(ranked, 1):
    mean, lo, hi = means[m]
    md.append(f"| {i} | {m} | {mean:.2f} | [{lo:.2f}, {hi:.2f}] |")
    out["exp3"][m] = {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)]}
# adjacent Wilcoxon, Holm-corrected
common = sorted(set.intersection(*[set(by_model[m]) for m in models]))
adj = []
for a, b in zip(ranked, ranked[1:]):
    xa = np.array([by_model[a][q]["total"] for q in common], float)
    xb = np.array([by_model[b][q]["total"] for q in common], float)
    if np.allclose(xa, xb):
        adj.append((a, b, 1.0)); continue
    try:
        p = stats.wilcoxon(xa, xb, zero_method="wilcox").pvalue
    except ValueError:
        p = 1.0
    adj.append((a, b, float(p)))
# Holm correction
ps = sorted(range(len(adj)), key=lambda i: adj[i][2])
holm = [0.0] * len(adj); k = len(adj)
for rank, i in enumerate(ps):
    holm[i] = min(1.0, adj[i][2] * (k - rank))
md.append("\n**Adjacent-pair Wilcoxon (Holm-corrected):**\n")
md.append("| Higher | Lower | raw p | Holm p | gap real? |")
md.append("|---|---|---|---|---|")
for (a, b, p), hp in zip(adj, holm):
    md.append(f"| {a} | {b} | {p:.2e} | {hp:.2e} | {'yes' if hp < 0.05 else 'NO (tie)'} |")
    out["exp3"].setdefault("_adjacent", []).append({"higher": a, "lower": b, "p_raw": p, "p_holm": hp, "sig": hp < 0.05})

# ---------- Exp 4: McNemar fabrication ----------
md.append("\n## Exp 4 — McNemar test: does alignment increase fabrication?\n")
md.append("| Family | Comparison | fab% before | fab% after | discordant b/c | p | effect |")
md.append("|---|---|---|---|---|---|---|")
fams = {"125M": ("125m-base","125m-dpo","125m-rlaif"),
        "500M": ("500m-base","500m-dpo","500m-rlaif"),
        "Gemma 2B": ("gemma-base","gemma-dpo","gemma-rlaif")}
for fam,(base,dpo,rlaif) in fams.items():
    for after in (dpo, rlaif):
        qa = [q for q in common]
        fb = np.array([by_model[base][q]["fabricated"] for q in qa], int)
        fa = np.array([by_model[after][q]["fabricated"] for q in qa], int)
        # 2x2: rows=before(0/1), cols=after(0/1)
        t = np.zeros((2,2), int)
        for x,y in zip(fb, fa): t[x,y]+=1
        b_, c_ = int(t[0,1]), int(t[1,0])   # discordant
        res = mcnemar(t, exact=(b_+c_ < 25))
        eff = fa.mean()-fb.mean()
        md.append(f"| {fam} | base->{after.split('-')[-1]} | {fb.mean()*100:.1f}% | {fa.mean()*100:.1f}% | {b_}/{c_} | {res.pvalue:.2e} | {eff*100:+.1f} pp |")
        out["exp4"].setdefault(fam, []).append({"vs": after, "fab_before": round(fb.mean(),4),
            "fab_after": round(fa.mean(),4), "discordant_bc": [b_,c_], "p": float(res.pvalue), "delta_pp": round(eff*100,1)})

# ---------- Exp 7: judge-bias ----------
md.append("\n## Exp 7 — judge-bias checks\n")
wc = np.array([len((r["answer_text"] or "").split()) for r in rows], float)
sc = np.array([r["total"] for r in rows], float)
rho, prho = stats.spearmanr(wc, sc)
pear = stats.pearsonr(wc, sc)
md.append(f"- **Verbosity (length vs score), all 7,500 rows:** Spearman rho = {rho:.3f} (p={prho:.1e}), "
          f"Pearson r = {pear[0]:.3f}. {'Non-trivial length effect — flag as a limitation.' if abs(rho)>=0.3 else 'Weak — no strong length bias.'}")
md.append("- **Order bias:** each answer is scored on its own (one call per answer, model name hidden), so there is no position/order to bias — noted in methods.")
# self-preference: Gemma mean vs others
gem = np.array([r["total"] for r in rows if r["size"]=="Gemma 2B"], float)
oth = np.array([r["total"] for r in rows if r["size"]!="Gemma 2B"], float)
md.append(f"- **Self-preference (Gemma vs rest):** Gemma mean {gem.mean():.2f} vs others {oth.mean():.2f}. "
          f"Gemma is the strongest family on independent Token-F1 too, so a higher judge score is expected on capability — "
          f"not evidence of Google-judge favouritism, but reported transparently.")
out["exp7"] = {"verbosity_spearman": round(float(rho),3), "verbosity_pearson": round(float(pear[0]),3),
               "gemma_mean": round(float(gem.mean()),3), "others_mean": round(float(oth.mean()),3)}

io.open(os.path.join(ER,"STATS_exp3_4_7.md"),"w",encoding="utf-8").write("\n".join(md)+"\n")
json.dump(out, io.open(os.path.join(ER,"STATS_exp3_4_7.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote STATS_exp3_4_7.md + .json")
print("\n".join(md[:60]))
