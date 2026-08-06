"""Judge a Modal-generated seeded eval (rubric-10 via the local server) and save the result.

    python evaluations/judge_modal_eval.py --seeded gemma-2-2b-sft-seed1

Reads eval_results/seed-evals/_modal_raw/<seeded>.json (produced by finetune/eval_modal.py:
per_item already has system/user/ref/resp/scores{token_f1,fabrication}), judges each answer with
the running serve_api /judge (serial warm-up then ThreadPoolExecutor), and writes
eval_results/seed-evals/<seeded>.judged.json + appends a summary row. Nothing is published.
"""
from __future__ import annotations
import argparse, json, io, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ER = os.path.join(ROOT, "eval_results")
SEED_DIR = os.path.join(ER, "seed-evals")
RAW_DIR = os.path.join(SEED_DIR, "_modal_raw")
BASE = "http://127.0.0.1:8000"


def post(path, body, timeout=120):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeded", required=True)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.makedirs(SEED_DIR, exist_ok=True)

    raw = json.load(io.open(os.path.join(RAW_DIR, f"{a.seeded}.json"), encoding="utf-8"))
    per_item = raw["per_item"]
    print(f"[{a.seeded}] judging {len(per_item)} answers (family={raw.get('family')})...", flush=True)

    def judge_one(it):
        u = it["user"]; i = u.rfind("Question:")
        q = u[i + len("Question:"):].strip() if i >= 0 else u
        ctx = u[:i].strip() if i >= 0 else ""
        try:
            g = post("/judge", {"question": q, "context": ctx, "reference": it["ref"],
                                "answers": {a.seeded: it["resp"]}})["graded"][a.seeded]
            return {"score": g.get("score"), "parts": g.get("parts"), "grounded": g.get("grounded")}
        except Exception as e:
            return {"error": str(e)[:160]}

    if per_item:
        per_item[0]["judge"] = judge_one(per_item[0])           # warm up serially
        with ThreadPoolExecutor(max_workers=6) as ex:
            for it, res in zip(per_item[1:], ex.map(judge_one, per_item[1:])):
                it["judge"] = res

    vals, gr, bysrc, f1s, fabs, errs = [], [], {}, [], [], 0
    for it in per_item:
        g = it.get("judge") or {}
        if g.get("error"):
            errs += 1
        if g.get("score") is not None:
            vals.append(g["score"]); gr.append(1 if g.get("grounded") else 0)
            bysrc.setdefault(it.get("source"), []).append(g["score"])
            f1s.append(it["scores"]["token_f1"]); fabs.append(it["scores"]["fabrication"])

    summary = {"seeded": a.seeded, "family": raw.get("family"), "source": "modal-eval", "n": len(vals),
               "judge_errors": errs,
               "score": round(sum(vals) / len(vals), 2) if vals else None,
               "grounded": round(sum(gr) / len(gr) * 100, 1) if gr else None,
               "by_source": {k: round(sum(v) / len(v), 2) for k, v in sorted(bysrc.items(), key=lambda kv: str(kv[0]))},
               "token_f1": round(sum(f1s) / len(f1s), 3) if f1s else None,
               "fabrication": round(sum(fabs) / len(fabs) * 100, 1) if fabs else None}
    json.dump({"summary": summary, "per_item": per_item},
              io.open(os.path.join(SEED_DIR, f"{a.seeded}.judged.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    sp = os.path.join(SEED_DIR, "seed_eval_summary.jsonl")
    io.open(sp, "a", encoding="utf-8").write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[{a.seeded}] DONE  score {summary['score']}/10  grounded {summary['grounded']}%  "
          f"F1 {summary['token_f1']}  fab {summary['fabrication']}%  errors {errs}  "
          f"-> seed-evals/{a.seeded}.judged.json", flush=True)


if __name__ == "__main__":
    main()
