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


METHOD_BLURB = (
    "_Headline metric = the correctness rating (1–5, rescaled to 0–1) of an independent LLM judge "
    "— Google's **gemini-3.1-flash-lite**, blind to which model wrote each answer — which is "
    "fair across the 125M / 500M / Gemma tokenizers and — unlike the reward model — is not the "
    "RLAIF training objective. token-F1 is reported alongside as a lexical proxy. Every "
    "model-vs-model claim uses a **paired bootstrap** on the same eval items (a difference counts "
    "only when the delta's 95% CI excludes 0); overlapping per-model CIs are never used as a test. "
    "All models scored on one decontaminated, held-out eval set (500 pair_ids × 4 RAFT conditions)._\n"
)


def key_findings(name, versions, data, headline):
    """Data-driven bullets — computed from the numbers, not hand-written."""
    present = [v for v in versions if v in data]
    models = [v for v in present if not META.get(v, {}).get("is_base")]
    F = []
    rm = rank_and_matrix(data, models, headline)
    if rm:
        top = rm["order"][0]
        F.append(f"**Best model:** {META[top]['label']} — {fmt_ci(rm['means'][top])} judged correctness (clean).")
        sig = [p for p in rm["pairs"] if p[2]["significant"]]
        F.append(f"**{len(sig)} of {len(rm['pairs'])}** pairwise comparisons are statistically resolved "
                 f"(paired bootstrap, 95% CI excludes 0).")
        # biggest significant gap
        if sig:
            a, b, d = max(sig, key=lambda p: abs(p[2]["delta"]))
            F.append(f"**Largest gap:** {META[a]['label']} beats {META[b]['label']} by "
                     f"{d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}].")
    # base -> best lift, per family
    for fam in ("125m", "500m", "gemma"):
        fbase = [v for v in present if META.get(v, {}).get("family") == fam and META.get(v, {}).get("is_base")]
        fmods = [v for v in models if META.get(v, {}).get("family") == fam]
        if fbase and fmods:
            b0 = mean_ci(list(by_key(data[fbase[0]]["per_item"], headline).values()))[0]
            best = max(fmods, key=lambda v: mean_ci(list(by_key(data[v]["per_item"], headline).values()))[0])
            bb = mean_ci(list(by_key(data[best]["per_item"], headline).values()))[0]
            F.append(f"**{fam.upper()} lift:** base {b0:.3f} → best ({META[best]['label']}) {bb:.3f}.")
    # anomaly flags
    for v in models:
        ab = by_key(data[v]["per_item"], "abstain", "clean")
        if ab and mean_ci(list(ab.values()))[0] > 0.3:
            F.append(f"⚠️ **{META[v]['label']} over-abstains** — says 'not stated' on "
                     f"{mean_ci(list(ab.values()))[0]*100:.0f}% of *answerable* clean questions.")
        ln = by_key(data[v]["per_item"], "resp_len_words", "clean")
        if ln and median(list(ln.values())) <= 3:
            F.append(f"⚠️ **{META[v]['label']} collapsed** to ~{median(list(ln.values())):.0f}-word answers.")
        f1k, jck = by_key(data[v]["per_item"], "token_f1"), by_key(data[v]["per_item"], headline)
        common = set(f1k) & set(jck)
        if common:
            dis = sum((f1k[k] > 0.5) != (jck[k] > 0.5) for k in common) / len(common)
            if dis > 0.4:
                F.append(f"**token-F1 misleads for {META[v]['label']}** (F1↔judge disagreement "
                         f"{dis:.2f}) — trust the judge, not word-overlap.")
    return F


def _q_of(user):
    return user.split("Question:")[-1].strip()[:130] if "Question:" in user else user[-130:]


def examples_section(name, versions, data, L):
    """Pull real generations that illustrate the headline phenomena."""
    present = [v for v in versions if v in data]
    tuned = [(v, it) for v in present if not META.get(v, {}).get("is_base")
             for it in data[v]["per_item"] if it.get("cond") == "clean"]
    L.append("\n### Illustrative examples (real generations)\n")

    def show(title, v, it):
        s, j = it["scores"], it.get("judge", {})
        L.append(f"**{title}** — _{META.get(v, {}).get('label', v)}_")
        L.append(f"- Q: {esc_inline(_q_of(it['user']))}")
        L.append(f"- reference: {esc_inline(it['ref'][:150])}")
        L.append(f"- model answer: {esc_inline(it['resp'][:240])}")
        L.append(f"- token-F1 **{s['token_f1']:.2f}** · judge correct **{j.get('correct','?')}/5** · "
                 f"grounded {j.get('grounded','?')} · {int(s['resp_len_words'])} words\n")

    def first(pred):
        for v, it in tuned:
            if pred(v, it):
                return v, it
        return None

    shown = 0
    ex = first(lambda v, it: it.get("judge", {}).get("correct", 0) >= 4
               and it["scores"]["token_f1"] < 0.2 and it["scores"]["resp_len_words"] >= 8)
    if ex:
        show("Correct answer that token-F1 badly under-scores (why the judge is the headline)", *ex); shown += 1
    ex = first(lambda v, it: it["scores"]["resp_len_words"] <= 3 and len(it["resp"].strip()) > 0)
    if ex:
        show("Degenerate short output (model collapse)", *ex); shown += 1
    ex = first(lambda v, it: it.get("answerable") and "not stated in the context" in it["resp"].lower())
    if ex:
        show("Over-abstention: refuses a question whose answer is present", *ex); shown += 1
    ex = first(lambda v, it: it["scores"].get("fabrication", 0) > 0 and it.get("judge", {}).get("grounded") is False)
    if ex:
        show("Ungrounded / fabricated content (judge flags it)", *ex); shown += 1
    if not shown:
        L.append("_(no notable-pattern examples surfaced in this set)_")


def esc_inline(s):
    return s.replace("\n", " ").replace("|", "\\|").strip()


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

    TITLES = {"set1": "SET1 — base vs SFT vs RAFT (125M & Gemma-2B)",
              "set2": "SET2 — alignment: DPO vs RLAIF (125M, 500M, Gemma-2B)"}
    comparisons = {}
    for name in ("set1", "set2"):
        present = [v for v in EXPERIMENTS[name] if v in data]
        any_judged = any(data[v]["judged"] and any(it.get("judge") for it in data[v]["per_item"])
                         for v in present)
        headline = HEADLINE if any_judged else "token_f1"

        # ---- detailed per-experiment report ----
        L = [f"# {TITLES[name]}\n", "_Detailed evaluation report — this experiment stands alone; "
             "no numbers are compared against the other set._\n", METHOD_BLURB]
        L.append("## Key findings\n")
        for b in key_findings(name, EXPERIMENTS[name], data, headline):
            L.append(f"- {b}")
        render_experiment(name, EXPERIMENTS[name], data, L)      # tables + rankings + RAFT
        examples_section(name, EXPERIMENTS[name], data, L)       # real generations
        manifest_check({v: data[v] for v in present}, L)         # per-set reproducibility
        report = "\n".join(L) + "\n"
        out = os.path.join(results_dir, f"REPORT_{name.upper()}.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"wrote {out}")

        rm = rank_and_matrix(data, present, headline)
        if rm:
            comparisons[name] = {
                "headline": headline, "order": rm["order"],
                "means": {v: rm["means"][v] for v in rm["order"]},
                "pairs": [{"a": a, "b": b, **d} for a, b, d in rm["pairs"]]}

    json.dump(comparisons, open(os.path.join(results_dir, "comparisons.json"), "w"), indent=2)
    print("wrote comparisons.json")


if __name__ == "__main__":
    main()
