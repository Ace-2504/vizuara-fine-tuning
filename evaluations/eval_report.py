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


# ---------- formatting ----------
def fmt_ci(triple):
    return f"{triple[0]:.3f} [{triple[1]:.3f},{triple[2]:.3f}]"


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

    # per-version headline table (clean condition)
    L.append("### Per-version (clean condition)\n")
    L.append(f"| model | {label(headline)} | groundedness | token-F1 | fabrication↓ | "
             f"false-abstain↓ | reward (secondary) |")
    L.append("|---|---|---|---|---|---|---|")
    for v in present:
        it = data[v]["per_item"]
        hv = mean_ci(list(by_key(it, headline).values())) if by_key(it, headline) else None
        gr = mean_ci(list(by_key(it, "grounded").values())) if by_key(it, "grounded") else None
        f1 = mean_ci(list(by_key(it, "token_f1").values()))
        fb = mean_ci(list(by_key(it, "fabrication").values()))
        fa = mean_ci(list(by_key(it, "false_abstain").values()))
        rw_key = by_key(it, "reward")
        if META.get(v, {}).get("reward_circular"):
            rw = "circular — omitted"
        elif rw_key:
            rw = fmt_ci(mean_ci(list(rw_key.values())))
        else:
            rw = "n/a"
        L.append(f"| {META.get(v, {}).get('label', v)} | {fmt_ci(hv) if hv else 'n/a'} | "
                 f"{fmt_ci(gr) if gr else 'n/a'} | {fmt_ci(f1)} | {fmt_ci(fb)} | {fmt_ci(fa)} | {rw} |")

    # pairwise significance on the headline metric
    rm = rank_and_matrix(data, present, headline)
    if rm:
        L.append(f"\n### Ranking by {label(headline)} (clean) + paired significance\n")
        for i, v in enumerate(rm["order"], 1):
            L.append(f"{i}. **{META.get(v, {}).get('label', v)}** — {fmt_ci(rm['means'][v])}")
        L.append("\n**Pairwise deltas** (A − B on the same items; ✓ = 95% CI excludes 0):\n")
        L.append("| A | B | Δ (A−B) | 95% CI | significant |")
        L.append("|---|---|---|---|---|")
        for a, b, d in rm["pairs"]:
            sig = "✓" if d["significant"] else "—"
            L.append(f"| {META.get(a,{}).get('label',a)} | {META.get(b,{}).get('label',b)} | "
                     f"{d['delta']:+.3f} | [{d['lo']:+.3f},{d['hi']:+.3f}] | {sig} |")

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
            L.append(f"| {c} | {fmt_ci(mean_ci(list(f1.values()))) if f1 else 'n/a'} | "
                     f"{fmt_ci(mean_ci(list(jc.values()))) if jc else 'n/a'} | "
                     f"{fmt_ci(mean_ci(list(ab.values()))) if ab else 'n/a'} |")
        # paired gaps with CIs (was a bare point estimate before)
        gg = paired_gap_ci(by_key(it, "token_f1", "realistic"), by_key(it, "token_f1", "closed_book"))
        dg = paired_gap_ci(by_key(it, "token_f1", "clean"), by_key(it, "token_f1", "realistic"))
        ca = mean_ci(list(by_key(it, "abstain", "retrieval_failure").values()))
        L.append(f"\n- grounding gap (realistic − closed_book F1): {gg['delta']:+.3f} "
                 f"[{gg['lo']:+.3f},{gg['hi']:+.3f}] {'✓' if gg['significant'] else '—'}")
        L.append(f"- distractor gap (clean − realistic F1): {dg['delta']:+.3f} "
                 f"[{dg['lo']:+.3f},{dg['hi']:+.3f}] {'✓' if dg['significant'] else '—'}")
        L.append(f"- correct abstention (retrieval_failure): {fmt_ci(ca)}")


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
