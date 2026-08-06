"""Rubric-10 judge the two new 500M models (500m-qa, 500m-raft) for real leaderboard rows.

Identical method to rejudge_rubric10.py — same running /judge endpoint (gemini-3.1-flash-lite,
gold-referenced), same 4-dimension 0-10 rubric — but only for the 2 new models, so it's cheap.
Combines the judge output (score10, grounded, by_source) with token_f1 + fabrication already
computed at generation time, producing arena-ready leaderboard rows.

Resumable: eval_results/rubric10_500m_new.jsonl
Output:    eval_results/leaderboard_500m_new.json

    python evaluations/judge_500m_new.py
"""
from __future__ import annotations
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.join(os.path.dirname(HERE), "eval_results")
BASE = os.environ.get("SLM_API", "http://127.0.0.1:8000")
OUT = os.path.join(ER, "rubric10_500m_new.jsonl")
SUMMARY = os.path.join(ER, "leaderboard_500m_new.json")

MODELS = [
    ("500m-qa",   "SLM-500M · QA",   "slm-500m-sft.judged.json"),
    ("500m-raft", "SLM-500M · RAFT", "slm-500m-raft.judged.json"),
]


def post(path, body, timeout=900):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.load(f)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    data, gen_scores = {}, {}
    for mid, _n, fn in MODELS:
        d = json.load(open(os.path.join(ER, fn), encoding="utf-8"))
        data[mid] = {i["pair_id"]: i for i in d["per_item"] if i["cond"] == "clean"}
        gen_scores[mid] = {i["pair_id"]: i["scores"] for i in d["per_item"] if i["cond"] == "clean"}
    common = sorted(set.intersection(*[set(v) for v in data.values()]))

    prior = {}
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line); prior[r["pair_id"]] = r
            except Exception:
                pass
    complete = {p for p, r in prior.items()
                if all((r["graded"].get(m[0]) or {}).get("score") is not None for m in MODELS)}
    todo = [p for p in common if p not in complete]
    print(f"questions: {len(common)}  done: {len(complete)}  to do: {len(todo)}", flush=True)

    ref_mid = MODELS[0][0]
    t0 = time.time()
    fh = open(OUT, "a", encoding="utf-8")
    for n, pid in enumerate(todo, 1):
        base = data[ref_mid][pid]
        u = base["user"]; i = u.rfind("Question:")
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
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        if n % 20 == 0 or n == len(todo):
            el = time.time() - t0; rate = n / el
            print(f"  {n}/{len(todo)}  {rate:.2f} q/s  ETA {(len(todo)-n)/rate/60:.0f} min", flush=True)
    fh.close()

    # ---- summarise into leaderboard rows ----
    latest = {}
    for l in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(l); latest[r["pair_id"]] = r
        except Exception:
            pass
    recs = list(latest.values())
    lb = {}
    for mid, name, _ in MODELS:
        vals, by_src, gr, f1s, fabs = [], {}, [], [], []
        for r in recs:
            g = r["graded"].get(mid) or {}
            if g.get("score") is None:
                continue
            vals.append(g["score"]); gr.append(1 if g.get("grounded") else 0)
            by_src.setdefault(r["source"], []).append(g["score"])
            gs = gen_scores[mid].get(r["pair_id"], {})
            f1s.append(gs.get("token_f1", 0.0)); fabs.append(gs.get("fabrication", 0.0))
        if not vals:
            continue
        lb[mid] = {
            "name": name,
            "score": round(sum(vals) / len(vals), 2),
            "grounded": round(sum(gr) / len(gr) * 100, 1),
            "by_source": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_src.items())},
            "fabrication": round(sum(fabs) / len(fabs) * 100, 1),
            "token_f1": round(sum(f1s) / len(f1s), 3),
            "n": len(vals),
        }
    json.dump(lb, open(SUMMARY, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {SUMMARY}")
    for mid, s in lb.items():
        print(f"  {s['name']:18s} score {s['score']:.2f}/10  grounded {s['grounded']:.1f}%  "
              f"F1 {s['token_f1']:.3f}  fab {s['fabrication']:.1f}%  by_source {s['by_source']}", flush=True)


if __name__ == "__main__":
    main()
