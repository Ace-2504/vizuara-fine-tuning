"""Position-swapped pairwise judging for one head-to-head (caveat 2).

The main judge (`judge_eval.py`) is POINTWISE: it scores each answer alone. That's cheap,
has no position bias, and feeds the paired bootstrap directly — but for the single most
important comparison in a set (e.g. DPO vs RLAIF within a family), a direct A-vs-B judgement
is sharper. LLM pairwise judges have a known POSITION BIAS (they favour whichever answer is
shown first), so this asks the judge BOTH orderings and only counts a win when the verdict
is consistent across the swap; inconsistent verdicts are ties (that's the bias, surfaced).

    python evaluations/judge_pairwise.py ./eval_results slm-500m-sft-dpo slm-500m-sft-rlaif
    python evaluations/judge_pairwise.py ./eval_results slm-125m-sft-dpo slm-125m-sft-rlaif --sample 150

Runs locally over the saved generations, resumable and budget-aware like judge_eval.py.
Writes pairwise_<A>_vs_<B>.json and prints win-rates + a bootstrap CI on the net preference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root -> teacher.py
sys.path.insert(0, _HERE)                    # evaluations/ -> stats.py, experiments.py
from teacher import TeacherClient, BudgetExhausted  # noqa: E402
from stats import mean_ci  # noqa: E402
from experiments import META  # noqa: E402

WINNER_SCHEMA = {"type": "object", "properties": {
    "winner": {"type": "string"}, "reason": {"type": "string"}}, "required": ["winner"]}


def pair_prompt(context_and_question: str, reference: str, ans1: str, ans2: str) -> str:
    return (
        "You are a strict evaluator of a grounded question-answering system. Two candidate "
        "answers are shown for the same question. Decide which is better at correctly and "
        "faithfully answering using ONLY the context. Judge meaning, not length or style; a "
        "correct paraphrase is fine; an answer that invents facts is worse.\n\n"
        f"CONTEXT + QUESTION:\n{context_and_question}\n\n"
        f"REFERENCE (a correct answer):\n{reference}\n\n"
        f"ANSWER 1:\n{ans1}\n\nANSWER 2:\n{ans2}\n\n"
        'Return JSON {"winner": "1" | "2" | "tie", "reason": "one short sentence"}.')


def load_version(results_dir, version):
    for name in (f"{version}.judged.json", f"{version}.json"):
        p = os.path.join(results_dir, name)
        if os.path.exists(p):
            blob = json.load(open(p, encoding="utf-8"))
            return {it["pair_id"]: it for it in blob.get("per_item", []) if it.get("cond") == "clean"}
    return {}


def cache_key(a, b, pid):
    return hashlib.sha1(f"{a}|{b}|{pid}".encode()).hexdigest()


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line); cache[r["key"]] = r["verdict"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def decide(teacher, ctx, ref, ans_a, ans_b):
    """Two calls with positions swapped. Returns 'A', 'B', or 'tie' (incl. inconsistent)."""
    o1 = teacher.generate_json(pair_prompt(ctx, ref, ans_a, ans_b), WINNER_SCHEMA, temperature=0.1)
    o2 = teacher.generate_json(pair_prompt(ctx, ref, ans_b, ans_a), WINNER_SCHEMA, temperature=0.1)
    if o1 is None or o2 is None:
        return None
    w1, w2 = str(o1.get("winner", "")).lower(), str(o2.get("winner", "")).lower()
    v1 = {"1": "A", "2": "B"}.get(w1, "tie")   # ordering 1: answer1=A
    v2 = {"1": "B", "2": "A"}.get(w2, "tie")   # ordering 2: answer1=B
    return v1 if v1 == v2 else "tie"           # consistent across swap, else tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="./eval_results")
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--sample", type=int, default=0, help="max items (0 = all shared clean items)")
    args = ap.parse_args()

    A = load_version(args.results_dir, args.a)
    B = load_version(args.results_dir, args.b)
    shared = sorted(set(A) & set(B))
    if not shared:
        sys.exit(f"no shared clean items between {args.a} and {args.b} — run eval.py for both.")
    if args.sample > 0:
        shared = shared[: args.sample]

    cache_path = os.path.join(args.results_dir, "_pairwise_cache.jsonl")
    cache = load_cache(cache_path)
    fh = open(cache_path, "a", encoding="utf-8")
    teacher = TeacherClient()
    la = META.get(args.a, {}).get("label", args.a)
    lb = META.get(args.b, {}).get("label", args.b)
    print(f"pairwise {la} vs {lb} on {len(shared)} items (position-swapped)", flush=True)

    signed, verdicts = [], {"A": 0, "B": 0, "tie": 0}
    try:
        for pid in shared:
            k = cache_key(args.a, args.b, pid)
            if k in cache:
                v = cache[k]
            else:
                v = decide(teacher, A[pid]["user"], A[pid]["ref"], A[pid]["resp"], B[pid]["resp"])
                if v is None:
                    continue
                cache[k] = v
                fh.write(json.dumps({"key": k, "verdict": v}) + "\n"); fh.flush()
            verdicts[v] += 1
            signed.append(1.0 if v == "A" else (-1.0 if v == "B" else 0.0))
    except BudgetExhausted as e:
        print(f"\nBUDGET EXHAUSTED: {str(e)[:160]}\nprogress cached — re-run to resume.", flush=True)
    finally:
        fh.close()

    n = len(signed)
    if n:
        net, lo, hi = mean_ci(signed)      # >0 favours A; CI excludes 0 => significant
        out = {"a": args.a, "b": args.b, "n": n, "wins_a": verdicts["A"], "wins_b": verdicts["B"],
               "ties": verdicts["tie"], "net_pref_a_minus_b": net, "ci": [lo, hi],
               "significant": bool(lo > 0 or hi < 0)}
        json.dump(out, open(os.path.join(args.results_dir,
                  f"pairwise_{args.a}_vs_{args.b}.json"), "w"), indent=2)
        print(f"\n{la}: {verdicts['A']} wins | {lb}: {verdicts['B']} wins | ties/incons.: "
              f"{verdicts['tie']}  (n={n})")
        print(f"net preference ({la} − {lb}): {net:+.3f} [{lo:+.3f},{hi:+.3f}] "
              f"-> {'significant' if out['significant'] else 'not resolved'}", flush=True)


if __name__ == "__main__":
    main()
