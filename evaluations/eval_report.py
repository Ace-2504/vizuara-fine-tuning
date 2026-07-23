"""Aggregate per-version eval results into TWO independent experiment reports (set1, set2).

This replaces the old "compare 95% CIs by eye" report with proper PAIRED significance:
for every pair of models in a set, the delta is bootstrapped on the SAME eval items with
shared indices, so a difference is called only when the delta's own 95% CI excludes zero.

Headline metric = the LLM judge's correctness (independent, cross-family fair). Groundedness,
token-F1, fabrication, and false-abstention are reported alongside. The reward model is shown
only as a SECONDARY signal and is suppressed for RLAIF versions (they optimised it -> circular).

    # 1) pull results + judge them:
    modal run evaluations/eval.py --set set1        # (and --set set2)
    modal volume get ft-data /eval ./eval_results
    python evaluations/judge_eval.py ./eval_results --set all
    # 2) build both reports:
    python evaluations/eval_report.py ./eval_results

Writes REPORT.md and comparisons.json into the results dir, and prints the same to stdout.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import EXPERIMENTS, META  # noqa: E402
from stats import mean_ci, paired_delta_ci, paired_gap_ci  # noqa: E402

HEADLINE = "judge_correct"      # primary metric; falls back to token_f1 if no judge present
LOWER_BETTER = {"fabrication", "false_abstain"}
RAFT_CONDS = ["clean", "realistic", "retrieval_failure", "closed_book"]


# ---------- load ----------
def load_items(results_dir: str) -> dict:
    """version -> list[per_item] (prefers *.judged.json, falls back to *.json)."""
    out = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        base = os.path.basename(path)
        if base.startswith("_") or base == "comparisons.json":
            continue
        version = base[:-len(".judged.json")] if base.endswith(".judged.json") else base[:-len(".json")]
        # a plain .json is skipped if its judged sibling exists
        if not base.endswith(".judged.json") and os.path.exists(
                os.path.join(results_dir, f"{version}.judged.json")):
            continue
        blob = json.load(open(path, encoding="utf-8"))
        out[version] = {"per_item": blob.get("per_item", []),
                        "result": blob.get("result", {}),
                        "judged": base.endswith(".judged.json")}
    return out


def item_metric(it: dict, metric: str):
    """Scalar value of a metric for one per-item record, or None if unavailable."""
    if metric == "judge_correct":
        j = it.get("judge")
        return None if not j else (float(j.get("correct", 0)) - 1.0) / 4.0
    if metric == "grounded":
        j = it.get("judge")
        return None if not j else float(bool(j.get("grounded")))
    if metric == "matches_ref":
        j = it.get("judge")
        return None if not j else float(bool(j.get("matches_ref")))
    return it.get("scores", {}).get(metric)


def by_key(items, metric, cond="clean"):
    """(pair_id) -> value, over one condition. Keyed by pair_id so two models align."""
    d = {}
    for it in items:
        if it.get("cond") != cond:
            continue
        v = item_metric(it, metric)
        if v is not None:
            d[it["pair_id"]] = v
    return d


# condition labels — closed_book is a parametric-recall probe, NOT grounding (caveat 5):
# the eval questions come from the same corpus these models were pretrained on, so a high
# closed_book score can reflect MEMORISATION of training text rather than QA ability.
COND_LABEL = {
    "clean": "clean",
    "realistic": "realistic (with distractors)",
    "retrieval_failure": "retrieval_failure (abstain expected)",
    "closed_book": "closed_book (parametric recall — contamination-sensitive)",
}


# ---------- formatting ----------
def fmt_ci(triple):
    return f"{triple[0]:.3f} [{triple[1]:.3f},{triple[2]:.3f}]"


def median(vals):
    import statistics
    vals = list(vals)
    return statistics.median(vals) if vals else 0.0


def rank_and_matrix(data, versions, metric, cond="clean"):
    """Ranking by mean + full pairwise paired-delta table for one metric/condition."""
    keyed = {v: by_key(data[v]["per_item"], metric, cond) for v in versions if v in data}
    keyed = {v: k for v, k in keyed.items() if k}          # drop versions with no data
    if len(keyed) < 1:
        return None
    means = {v: mean_ci(k.values()) for v, k in keyed.items()}
    lower = metric in LOWER_BETTER
    order = sorted(keyed, key=lambda v: means[v][0], reverse=not lower)

    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            d = paired_delta_ci(keyed[a], keyed[b])
            pairs.append((a, b, d))
    return {"metric": metric, "cond": cond, "order": order, "means": means, "pairs": pairs,
            "lower_better": lower}


def render_experiment(name, versions, data, L):
    present = [v for v in versions if v in data]
    L.append(f"\n{'='*78}\n## Experiment {name.upper()} — {len(present)}/{len(versions)} versions present\n")
    missing = [v for v in versions if v not in data]
    if missing:
        L.append(f"_missing (not yet evaluated): {', '.join(missing)}_\n")
    if not present:
        return
    any_judged = any(data[v]["judged"] and any(it.get("judge") for it in data[v]["per_item"])
                     for v in present)
    headline = HEADLINE if any_judged else "token_f1"
    if not any_judged:
        L.append("_no judge scores found — falling back to token-F1 as headline. Run "
                 "judge_eval.py for the fair cross-family metric._\n")

    # base models are a FLOOR, not peers (caveat 8): they're few-shot-prompted in eval.py so
    # they show real capability, but a few-shot base vs a zero-shot tuned model still isn't
    # apples-to-apples, so they're kept as a reference line and excluded from the matrix.
    bases = [v for v in present if META.get(v, {}).get("is_base")]
    models = [v for v in present if not META.get(v, {}).get("is_base")]

    # per-version table (clean condition)
    L.append("### Per-version (clean condition)\n")
    L.append(f"| model | {label(headline)} | groundedness | matches-ref | token-F1 | "
             f"fabrication↓ | false-abstain↓ | median len | reward (within-family) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for v in present:
        it = data[v]["per_item"]
        def m(metric, _it=it):
            k = by_key(_it, metric)
            return fmt_ci(mean_ci(list(k.values()))) if k else "n/a"
        if META.get(v, {}).get("reward_circular"):
            rw = "circular — omitted"                       # caveat 6: RLAIF trained on this RM
        else:
            rk = by_key(it, "reward")
            rw = fmt_ci(mean_ci(list(rk.values()))) if rk else "n/a"
        lk = by_key(it, "resp_len_words")
        med = f"{median(lk.values()):.0f}w" if lk else "n/a"
        tag = " _(base·few-shot)_" if v in bases else ""
        L.append(f"| {META.get(v, {}).get('label', v)}{tag} | {m(headline)} | {m('grounded')} | "
                 f"{m('matches_ref')} | {m('token_f1')} | {m('fabrication')} | {m('false_abstain')} | "
                 f"{med} | {rw} |")
    L.append("\n_Reward is a SECONDARY signal from a 500M-backbone reward model — meaningful only "
             "WITHIN a family, never across families, and shown next to median length because "
             "reward models favour longer answers. The judge is the headline (caveat 6)._")

    # caveat 4: where lexical token-F1 disagrees with the (meaning-based) judge
    if headline == HEADLINE:
        L.append("\n### Lexical-F1 vs judge disagreement (caveat 4)\n")
        L.append("_Fraction of clean items where token-F1 (>0.5) and the judge (correct>0.5) "
                 "disagree — higher = token-F1 is a worse proxy for that model, usually because "
                 "it punishes correct paraphrases. This is why the judge, not F1, is the headline._\n")
        L.append("| model | F1↔judge disagreement |")
        L.append("|---|---|")
        for v in present:
            it = data[v]["per_item"]
            f1k, jck = by_key(it, "token_f1"), by_key(it, headline)
            common = set(f1k) & set(jck)
            dis = (f"{sum((f1k[k] > 0.5) != (jck[k] > 0.5) for k in common) / len(common):.3f}"
                   if common else "n/a")
            L.append(f"| {META.get(v, {}).get('label', v)} | {dis} |")

    # pairwise significance on the headline metric — TUNED models only (bases are a floor)
    rm = rank_and_matrix(data, models, headline)
    if rm:
        L.append(f"\n### Ranking by {label(headline)} (clean) + paired significance\n")
        L.append("_Tuned models only; base models are the floor (table above), excluded here._\n")
        for i, v in enumerate(rm["order"], 1):
            L.append(f"{i}. **{META.get(v, {}).get('label', v)}** — {fmt_ci(rm['means'][v])}")
        L.append("\n**Pairwise deltas** (A − B on the same items; ✓ = 95% CI excludes 0):\n")
        L.append("| A | B | Δ (A−B) | 95% CI | significant |")
        L.append("|---|---|---|---|---|")
        for a, b, d in rm["pairs"]:
            sig = "✓" if d["significant"] else "—"
            L.append(f"| {META.get(a,{}).get('label',a)} | {META.get(b,{}).get('label',b)} | "
                     f"{d['delta']:+.3f} | [{d['lo']:+.3f},{d['hi']:+.3f}] | {sig} |")
    if bases:
        floor = ", ".join(
            f"{META.get(v,{}).get('label',v)} {fmt_ci(mean_ci(list(by_key(data[v]['per_item'], headline).values())))}"
            for v in bases if by_key(data[v]["per_item"], headline))
        if floor:
            L.append(f"\n_Floor (few-shot base models): {floor}._")

    # RAFT four-condition breakdown (paired gap CIs)
    rafts = [v for v in present if META.get(v, {}).get("is_raft")]
    for v in rafts:
        it = data[v]["per_item"]
        L.append(f"\n### RAFT breakdown — {META.get(v,{}).get('label', v)}\n")
        L.append("| condition | token-F1 | judged correctness | abstain rate |")
        L.append("|---|---|---|---|")
        for c in RAFT_CONDS:
            f1 = by_key(it, "token_f1", c)
            jc = by_key(it, headline, c) if headline == HEADLINE else {}
            ab = by_key(it, "abstain", c)
            L.append(f"| {COND_LABEL.get(c, c)} | {fmt_ci(mean_ci(list(f1.values()))) if f1 else 'n/a'} | "
                     f"{fmt_ci(mean_ci(list(jc.values()))) if jc else 'n/a'} | "
                     f"{fmt_ci(mean_ci(list(ab.values()))) if ab else 'n/a'} |")
        # paired gaps with CIs (was a bare point estimate before)
        gg = paired_gap_ci(by_key(it, "token_f1", "realistic"), by_key(it, "token_f1", "closed_book"))
        dg = paired_gap_ci(by_key(it, "token_f1", "clean"), by_key(it, "token_f1", "realistic"))
        ca = mean_ci(list(by_key(it, "abstain", "retrieval_failure").values()))
        L.append(f"\n- grounding gap (realistic − closed_book F1): {gg['delta']:+.3f} "
                 f"[{gg['lo']:+.3f},{gg['hi']:+.3f}] {'✓' if gg['significant'] else '—'} "
                 f"— how much having the right document helps.")
        L.append(f"- distractor gap (clean − realistic F1): {dg['delta']:+.3f} "
                 f"[{dg['lo']:+.3f},{dg['hi']:+.3f}] {'✓' if dg['significant'] else '—'}")
        L.append(f"- correct abstention (retrieval_failure): {fmt_ci(ca)}")
        # caveat 5: flag when closed_book rivals clean -> likely memorisation, not grounding
        cb = mean_ci(list(by_key(it, "token_f1", "closed_book").values())) if by_key(it, "token_f1", "closed_book") else None
        cl_ = mean_ci(list(by_key(it, "token_f1", "clean").values())) if by_key(it, "token_f1", "clean") else None
        if cb and cl_:
            flag = " ⚠️ closed_book ≈ clean: score may reflect MEMORISED training text, not QA skill" if cb[0] >= cl_[0] - 0.05 else ""
            L.append(f"- closed_book F1 {cb[0]:.3f} vs clean {cl_[0]:.3f} — closed_book is a "
                     f"parametric-recall probe (contamination-sensitive), not grounding.{flag}")


def label(metric):
    return {"judge_correct": "judged correctness", "token_f1": "token-F1",
            "grounded": "groundedness"}.get(metric, metric)


def manifest_check(data, L):
    hashes = {}
    for v, d in data.items():
        h = d["result"].get("manifest", {}).get("eval_sha256")
        if h:
            hashes.setdefault(h, []).append(v)
    L.append(f"\n{'='*78}\n## Reproducibility\n")
    if not hashes:
        L.append("_no manifest found — re-run the updated eval.py to record eval-set hash, "
                 "decoding params, and library versions._")
    elif len(hashes) == 1:
        L.append(f"All versions scored on the same eval set (sha256 {list(hashes)[0][:12]}…). ✓")
    else:
        L.append("⚠️ Versions were scored on DIFFERENT eval sets — comparisons may be invalid:")
        for h, vs in hashes.items():
            L.append(f"  - {h[:12]}…: {', '.join(vs)}")


def main():
    # the report uses ✓/—/↓; Windows consoles default to cp1252 and would crash on print.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./eval_results"
    data = load_items(results_dir)
    if not data:
        print(f"no eval json found in {results_dir}"); return

    L = ["# SLM fine-tuning evaluation — set1 & set2\n",
         "_Two independent experiments. Headline = LLM-judge correctness (cross-family fair); "
         "reward model is secondary and suppressed for RLAIF (circular). All model-vs-model "
         "claims use paired bootstrap on shared items — overlapping per-model CIs are NOT used "
         "as a test._\n"]
    for name in ("set1", "set2"):
        render_experiment(name, EXPERIMENTS[name], data, L)
    manifest_check(data, L)

    report = "\n".join(L) + "\n"
    print(report)
    with open(os.path.join(results_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)

    # machine-readable comparisons for both sets
    comparisons = {}
    for name in ("set1", "set2"):
        present = [v for v in EXPERIMENTS[name] if v in data]
        any_judged = any(data[v]["judged"] and any(it.get("judge") for it in data[v]["per_item"])
                         for v in present)
        headline = HEADLINE if any_judged else "token_f1"
        rm = rank_and_matrix(data, present, headline)
        if rm:
            comparisons[name] = {
                "headline": headline, "order": rm["order"],
                "means": {v: rm["means"][v] for v in rm["order"]},
                "pairs": [{"a": a, "b": b, **d} for a, b, d in rm["pairs"]]}
    json.dump(comparisons, open(os.path.join(results_dir, "comparisons.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(results_dir, 'REPORT.md')} and comparisons.json")


if __name__ == "__main__":
    main()
