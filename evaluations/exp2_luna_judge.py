"""Exp 2 — second-judge cross-validation with gpt-5.6-luna (OpenAI).

Re-scores the SAME per-model answers the Gemini primary judge scored, with the IDENTICAL rubric
prompt (only the judge model changes), then lets build_exp2_stats.py compute rank correlation vs
Gemini. Reads the 15 published models' <file>.judged.json (clean items), writes per-(model,pair_id)
luna grades to eval_results/exp2_luna_scores.jsonl (resumable — skips rows already present).

    python evaluations/exp2_luna_judge.py --limit 5            # smoke (5 q/model, measures cost)
    python evaluations/exp2_luna_judge.py --limit 150          # 150-question subset
    python evaluations/exp2_luna_judge.py                      # full 15x500

Nothing is published.
"""
from __future__ import annotations
import argparse, json, io, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ER = os.path.join(ROOT, "eval_results")
OUT = os.path.join(ER, "exp2_luna_scores.jsonl")
MODEL = "gpt-5.6-luna"

# same 15 models + per-item judged files as build_p0.py
MODELS = [
    ("125m-base","base-125m.judged.json"),("125m-qa","slm-125m-sft.judged.json"),
    ("125m-raft","slm-125m-raft.judged.json"),("125m-dpo","slm-125m-sft-dpo.judged.json"),
    ("125m-rlaif","slm-125m-sft-rlaif.judged.json"),("500m-base","base-500m.judged.json"),
    ("500m-qa","slm-500m-sft.judged.json"),("500m-raft","slm-500m-raft.judged.json"),
    ("500m-dpo","slm-500m-sft-dpo.judged.json"),("500m-rlaif","slm-500m-sft-rlaif.judged.json"),
    ("gemma-base","base-gemma.judged.json"),("gemma-qa","gemma-2-2b-sft.judged.json"),
    ("gemma-raft","gemma-2-2b-raft.judged.json"),("gemma-dpo","gemma-2-2b-sft-dpo.judged.json"),
    ("gemma-rlaif","gemma-2-2b-sft-rlaif.judged.json"),
]

# ---- verbatim copy of the server's rubric + prompt builder (so luna judges identically) ----
RUBRIC = (
    'Score four dimensions, then nothing else:\n'
    '- "correctness" (0-5): factual agreement with the answer. 5 = fully right, 0 = wrong.\n'
    '- "completeness" (0-2): covers the key points, not just one of them.\n'
    '- "groundedness" (0-2): 2 = invents nothing; 0 = fabricated figures, cases or citations.\n'
    '- "clarity" (0-1): answers what was actually asked, without padding or contradiction.\n'
    '- "reason": one short sentence.\n'
    'The four add up to a score out of 10. Judge meaning, not wording.\n')


def judge_prompt(question, context, reference, candidate):
    head = ("You are a strict evaluator of a question-answering system. Judge ONLY the "
            "CANDIDATE answer; you do not know which system produced it.\n")
    if reference:
        rule = ("The REFERENCE is a correct short answer. Grade the CANDIDATE against it. "
                "Judge meaning, not wording — a correct paraphrase scores high.\n")
        ref_block = f"\nREFERENCE:\n{reference}\n"
    else:
        rule = ("No reference answer is available. Grade the CANDIDATE on factual accuracy and "
                "whether it actually answers the question, using your own knowledge. Be strict: "
                "vague, evasive, repetitive or fabricated answers score low.\n")
        ref_block = ""
    ctx_block = f"\nCONTEXT:\n{context}\n" if context else ""
    ground = ("Groundedness is judged against the CONTEXT.\n" if context else "")
    return (head + rule + "\n" + RUBRIC + ground + ctx_block +
            f"\nQUESTION:\n{question}\n" + ref_block + f"\nCANDIDATE:\n{candidate}")


def load_key():
    for line in io.open(os.path.join("/d/slm-arena-15/.env.local".replace("/d/", "D:/")), encoding="utf-8"):
        line = line.strip()
        if line.startswith("OPENAI_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no OPENAI_API_KEY in slm-arena-15/.env.local")


def split_qc(user):
    i = user.rfind("Question:")
    q = user[i + len("Question:"):].strip() if i >= 0 else user
    ctx = user[:i].strip() if i >= 0 else ""
    return q, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="questions/model (0=all 500)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=3000)
    a = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    # resume: load already-judged (model,pair_id)
    done = set()
    if os.path.exists(OUT):
        for line in io.open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line); done.add((r["model"], r["pair_id"]))
            except Exception: pass
    print(f"[exp2] resume: {len(done)} rows already judged", flush=True)

    # build work list
    work = []
    for mid, fn in MODELS:
        p = os.path.join(ER, fn)
        if not os.path.exists(p):
            print(f"  WARN missing {fn} for {mid}"); continue
        items = [it for it in json.load(io.open(p, encoding="utf-8"))["per_item"] if it["cond"] == "clean"]
        items.sort(key=lambda it: it["pair_id"])            # fixed order -> same subset across models
        if a.limit: items = items[:a.limit]
        for it in items:
            if (mid, it["pair_id"]) in done: continue
            work.append((mid, it))
    print(f"[exp2] {len(work)} judgments to run (model={MODEL}, workers={a.workers}, limit={a.limit or 'all'})", flush=True)
    if not work: print("[exp2] nothing to do"); return

    lock = threading.Lock(); t0 = time.time(); n = [0]; toks = [0, 0]; errs = [0]
    fout = io.open(OUT, "a", encoding="utf-8")

    def one(job):
        mid, it = job
        q, ctx = split_qc(it["user"])
        prompt = judge_prompt(q, ctx, it["ref"], it["resp"])
        msg = (prompt + '\n\nRespond with ONLY a JSON object: '
               '{"correctness":<0-5>,"completeness":<0-2>,"groundedness":<0-2>,"clarity":<0-1>,"reason":"..."}')
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": msg}],
                    response_format={"type": "json_object"}, max_completion_tokens=a.max_tokens)
                txt = r.choices[0].message.content or ""
                g = json.loads(txt)
                cc = int(max(0, min(5, g.get("correctness", 0))))
                cp = int(max(0, min(2, g.get("completeness", 0))))
                gr = int(max(0, min(2, g.get("groundedness", 0))))
                cl = int(max(0, min(1, g.get("clarity", 0))))
                row = {"model": mid, "pair_id": it["pair_id"], "source": it.get("source"),
                       "parts": {"correctness": cc, "completeness": cp, "groundedness": gr, "clarity": cl},
                       "score": float(cc + cp + gr + cl), "grounded": gr == 2}
                u = getattr(r, "usage", None)
                with lock:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
                    n[0] += 1
                    if u: toks[0] += u.prompt_tokens; toks[1] += u.completion_tokens
                    if n[0] % 50 == 0:
                        print(f"  {n[0]} done ({(time.time()-t0)/n[0]:.2f}s/call, "
                              f"in={toks[0]} out={toks[1]} tok)", flush=True)
                return
            except Exception as e:
                if attempt == 3:
                    with lock: errs[0] += 1
                    print(f"  ERR {mid}/{it['pair_id']}: {str(e)[:120]}", flush=True)
                else:
                    time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, work))
    fout.close()
    dt = time.time() - t0
    # rough cost: $0.20/1M in, $1.20/1M out
    cost = toks[0] / 1e6 * 0.20 + toks[1] / 1e6 * 1.20
    print(f"[exp2] DONE {n[0]} judged, {errs[0]} errors, {dt/60:.1f} min | "
          f"tokens in={toks[0]} out={toks[1]} | est ${cost:.2f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
