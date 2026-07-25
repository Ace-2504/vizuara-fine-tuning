"""Re-score the frozen evaluation with the arena's 4-dimension 0-10 rubric.

    python evaluations/rejudge_rubric10.py            # resumable; safe to re-run

Why: the offline harness stores a single integer 1-5, which maps to only five possible scores
(0/2.5/5/7.5/10). The live arena grades Correctness 0-5 + Completeness 0-2 + Groundedness 0-2 +
Clarity 0-1. Until the frozen set is scored the same way, a leaderboard mean and an arena score
are different measurements.

This calls the RUNNING serve_api `/judge` endpoint rather than re-implementing the prompt, so the
rubric, judge model and parsing are identical to the arena by construction. One HTTP call per
question carries all 13 models' answers and fans out internally.

Originals are never touched: results go to eval_results/rubric10.jsonl (resumable) and a summary
at eval_results/rubric10_summary.json.
"""
from __future__ import annotations
import json, os, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.join(os.path.dirname(HERE), "eval_results")
BASE = os.environ.get("SLM_API", "http://127.0.0.1:8000")
OUT = os.path.join(ER, "rubric10.jsonl")
SUMMARY = os.path.join(ER, "rubric10_summary.json")

MODELS = [
    ("125m-base",  "SLM-125M",         "base-125m.judged.json"),
    ("125m-qa",    "SLM-125M · QA",    "slm-125m-sft.judged.json"),
    ("125m-raft",  "SLM-125M · RAFT",  "slm-125m-raft.judged.json"),
    ("125m-dpo",   "SLM-125M · DPO",   "slm-125m-sft-dpo.judged.json"),
    ("125m-rlaif", "SLM-125M · RLAIF", "slm-125m-sft-rlaif.judged.json"),
    ("500m-base",  "SLM-500M",         "base-500m.judged.json"),
    ("500m-dpo",   "SLM-500M · DPO",   "slm-500m-sft-dpo.judged.json"),
    ("500m-rlaif", "SLM-500M · RLAIF", "slm-500m-sft-rlaif.judged.json"),
    ("gemma-base", "Gemma 2B",         "base-gemma.judged.json"),
    ("gemma-qa",   "Gemma 2B · QA",    "gemma-2-2b-sft.judged.json"),
    ("gemma-raft", "Gemma 2B · RAFT",  "gemma-2-2b-raft.judged.json"),
    ("gemma-dpo",  "Gemma 2B · DPO",   "gemma-2-2b-sft-dpo.judged.json"),
    ("gemma-rlaif","Gemma 2B · RLAIF", "gemma-2-2b-sft-rlaif.judged.json"),
]


def post(path, body, timeout=900):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.load(f)


def main():
    data = {}
    for mid, _name, fn in MODELS:
        d = json.load(open(os.path.join(ER, fn), encoding="utf-8"))
        data[mid] = {i["pair_id"]: i for i in d["per_item"] if i["cond"] == "clean"}
    common = sorted(set.intersection(*[set(v) for v in data.values()]))

    # A question counts as done only when EVERY model got a score. A partially-graded question
    # (e.g. the API ran out of credits mid-batch) is retried — grading it again is cheap next to
    # silently shipping a hole in the table.
    prior: dict = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    prior[r["pair_id"]] = r          # later lines win
                except Exception:
                    pass
    complete = {p for p, r in prior.items()
                if all((r["graded"].get(m[0]) or {}).get("score") is not None for m in MODELS)}
    todo = [p for p in common if p not in complete]
    partial = len(prior) - len(complete)
    if partial:
        print(f"retrying {partial} partially-graded question(s)", flush=True)
    print(f"questions: {len(common)}  complete: {len(complete)}  to do: {len(todo)}", flush=True)
    print(f"judge calls remaining: {len(todo) * len(MODELS)}", flush=True)

    ref_model = "gemma-qa"
    t0 = time.time()
    fh = open(OUT, "a", encoding="utf-8")
    for n, pid in enumerate(todo, 1):
        base = data[ref_model][pid]
        u = base["user"]
        i = u.rfind("Question:")
        question = u[i + len("Question:"):].strip() if i > 0 else u.strip()
        context = u[:i].strip() if i > 0 else ""
        answers = {mid: (data[mid][pid]["resp"] or "").strip() for mid, _, _ in MODELS}
        try:
            g = post("/judge", {"question": question, "context": context,
                                "reference": base["ref"], "answers": answers})
        except Exception as e:
            print(f"  [{n}] {pid} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        rec = {"pair_id": pid, "source": base["source"],
               "graded": {k: {kk: v.get(kk) for kk in ("score", "parts", "grounded")}
                          for k, v in g["graded"].items()}}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            rate = n / el
            eta = (len(todo) - n) / rate / 60 if rate else 0
            print(f"  {n}/{len(todo)} questions  {n*len(MODELS)/el:.2f} judge-calls/s  ETA {eta:.0f} min",
                  flush=True)
    fh.close()

    # ---- summarise ----
    latest = {}
    for l in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(l); latest[r["pair_id"]] = r    # a retry supersedes its earlier attempt
        except Exception:
            pass
    recs = list(latest.values())
    summary = {}
    for mid, name, _ in MODELS:
        vals, by_src, gr = [], {}, []
        for r in recs:
            g = r["graded"].get(mid) or {}
            if g.get("score") is None:
                continue
            vals.append(g["score"])
            by_src.setdefault(r["source"], []).append(g["score"])
            gr.append(1 if g.get("grounded") else 0)
        if not vals:
            continue
        summary[mid] = {
            "name": name,
            "score10": round(sum(vals) / len(vals), 2),
            "grounded": round(sum(gr) / len(gr) * 100, 1),
            "by_source": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_src.items())},
            "n": len(vals),
        }
    json.dump(summary, open(SUMMARY, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {SUMMARY}")
    for mid, s in sorted(summary.items(), key=lambda x: -x[1]["score10"]):
        print(f"  {s['name']:20s} {s['score10']:5.2f}/10  grounded {s['grounded']:5.1f}%  n={s['n']}")


if __name__ == "__main__":
    main()
