"""Build the RLAIF/DPO preference dataset: 500 (prompt, chosen, rejected) triplets.

Off-policy: Gemini writes a strong and a weaker grounded answer per prompt; a blind LLM-judge
verifies the intended `chosen` actually wins; deduped. Reuses grounded QA prompts (passage +
question) from the SFT build's raw.jsonl so the format matches how the QA-SFT models were trained.

Run from repo root:  python rl/build_prefs.py
Resumable: generation and judging both cache to disk; a credit-out pauses, re-run resumes.
"""
from __future__ import annotations
import hashlib, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as _C  # noqa: sets HF token env
from teacher import TeacherClient, BudgetExhausted

RAW_QA = "data/sft/raw.jsonl"
OUT = "rl/data"; os.makedirs(OUT, exist_ok=True)
RAW_PREFS = f"{OUT}/raw_prefs.jsonl"
JUDGE_CACHE = f"{OUT}/pref_judge.jsonl"
PREFS = f"{OUT}/preferences.jsonl"

N_TARGET, N_RAW, WORKERS = 500, 720, 8
MODELS = ("gemini-3.1-flash-lite",)          # full 3.1-flash 404s on this key
SYS_GROUNDED = ("You are a precise legal and financial assistant. Answer clearly using the "
                "provided context; do not invent facts.")
JUDGE_MARGIN_MIN = 3

PAIR_SCHEMA = {"type": "object", "properties": {
    "chosen": {"type": "string"}, "rejected": {"type": "string"}},
    "required": ["chosen", "rejected"]}
JUDGE_SCHEMA = {"type": "object", "properties": {
    "better": {"type": "string"}, "margin": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["better", "margin"]}


def pair_prompt(passage, q):
    return (f"CONTEXT:\n{passage}\n\nQUESTION: {q}\n\n"
            "Return JSON with two answers to the QUESTION using ONLY the CONTEXT:\n"
            '  "chosen": a correct, complete, well-structured answer grounded in the context.\n'
            '  "rejected": a PLAUSIBLE but clearly worse answer — less accurate, missing a key '
            "point, vague, or over-hedged. Still a real attempt, not gibberish.\n"
            "Both answer the same question; only quality differs. Keep each under 90 words.")


def judge_prompt(passage, q, a, b):
    return (f"CONTEXT:\n{passage}\n\nQUESTION: {q}\n\nAnswer A: {a}\n\nAnswer B: {b}\n\n"
            "Which answer is better — more accurate, complete, and grounded in the CONTEXT?\n"
            'Return {"better":"A" or "B","margin":1-5,"reason":...}. margin = how much better.')


def norm(s): return " ".join(s.split()).lower()
def pid(q): return hashlib.sha256(norm(q).encode()).hexdigest()[:16]


def load_prompts():
    """Distinct grounded QA prompts (passage + question) from raw.jsonl, embedding-deduped."""
    seen, rows = set(), []
    for l in open(RAW_QA, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("task") != "qa" or not r.get("question") or not r.get("passage"):
            continue
        h = pid(r["question"])
        if h in seen:
            continue
        seen.add(h); rows.append(r)
    random.Random(1337).shuffle(rows)
    # semantic dedup on a generous slice, then take N_RAW
    from sentence_transformers import SentenceTransformer
    import numpy as np, torch
    cand = rows[:N_RAW * 3]
    emb = SentenceTransformer(_C.EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu"
                              ).encode([r["question"] for r in cand], normalize_embeddings=True,
                                       batch_size=256, show_progress_bar=False).astype(np.float32)
    keep, idx = [], []
    for i in range(len(cand)):
        if idx and float((emb[idx] @ emb[i]).max()) >= 0.90:
            continue
        keep.append(cand[i]); idx.append(i)
        if len(keep) >= N_RAW:
            break
    print(f"prompts: {len(rows)} distinct -> {len(keep)} after semantic dedup", flush=True)
    return keep


def _run_stage(items, fn, label):
    """Concurrent map with budget-halt + append-as-you-go handled inside fn."""
    stop = threading.Event()
    teachers = [TeacherClient(models=MODELS, min_interval_s=0.0) for _ in range(WORKERS)]
    hit = [False]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fn, teachers[i % WORKERS], it) for i, it in enumerate(items) if not stop.is_set()]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except BudgetExhausted:
                hit[0] = True; stop.set(); break
            if n % 100 == 0:
                print(f"  {label}: {n}/{len(items)}", flush=True)
    if hit[0]:
        print(f"[BUDGET] {label} paused — cached. Top up and re-run to resume.", flush=True)
    return not hit[0]


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    prompts = load_prompts()
    if args.limit:
        prompts = prompts[:args.limit]

    # ---- stage 1: generate (chosen, rejected) ----
    done = set()
    if os.path.exists(RAW_PREFS):
        for l in open(RAW_PREFS, encoding="utf-8"):
            try: done.add(json.loads(l)["prompt_id"])
            except Exception: pass
    todo = [r for r in prompts if pid(r["question"]) not in done]
    print(f"generate: {len(done)} cached, {len(todo)} to make", flush=True)
    lock = threading.Lock(); fh = open(RAW_PREFS, "a", encoding="utf-8")

    def gen(teacher, r):
        out = teacher.generate_json(pair_prompt(r["passage"], r["question"]), PAIR_SCHEMA, temperature=0.9)
        if not out:
            return
        ch, rj = (out.get("chosen") or "").strip(), (out.get("rejected") or "").strip()
        if len(ch) < 8 or len(rj) < 8 or norm(ch) == norm(rj):
            return
        rec = {"prompt_id": pid(r["question"]), "question": r["question"], "passage": r["passage"],
               "source": r.get("source", ""), "chosen": ch, "rejected": rj}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
    if todo:
        _run_stage(todo, gen, "generate")
    fh.close()

    raw = [json.loads(l) for l in open(RAW_PREFS, encoding="utf-8") if l.strip()]
    print(f"raw pairs after clean: {len(raw)}", flush=True)

    # ---- stage 2: blind judge verification ----
    jc = {}
    if os.path.exists(JUDGE_CACHE):
        for l in open(JUDGE_CACHE, encoding="utf-8"):
            v = json.loads(l); jc[v["prompt_id"]] = v
    jtodo = [r for r in raw if r["prompt_id"] not in jc]
    print(f"judge: {len(jc)} cached, {len(jtodo)} to verify", flush=True)
    jlock = threading.Lock(); jfh = open(JUDGE_CACHE, "a", encoding="utf-8")
    rng = random.Random(1337)

    def judge(teacher, r):
        swap = rng.random() < 0.5                       # blind: randomize A/B
        a, b = (r["rejected"], r["chosen"]) if swap else (r["chosen"], r["rejected"])
        v = teacher.generate_json(judge_prompt(r["passage"], r["question"], a, b), JUDGE_SCHEMA,
                                  temperature=0.0)
        if not v:
            return
        better = str(v.get("better", "")).strip().upper()
        chosen_is = "B" if swap else "A"               # where chosen sits
        rec = {"prompt_id": r["prompt_id"],
               "chosen_wins": better == chosen_is, "margin": int(v.get("margin", 0))}
        with jlock:
            jc[r["prompt_id"]] = rec
            jfh.write(json.dumps(rec) + "\n"); jfh.flush()
    if jtodo:
        _run_stage(jtodo, judge, "judge")
    jfh.close()

    # ---- stage 3: keep judge-confirmed, format ----
    kept = []
    for r in raw:
        v = jc.get(r["prompt_id"])
        if v and v["chosen_wins"] and v["margin"] >= JUDGE_MARGIN_MIN:
            kept.append(r)
    rng.shuffle(kept); kept = kept[:N_TARGET]
    with open(PREFS, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({
                "prompt": [{"role": "system", "content": SYS_GROUNDED},
                           {"role": "user", "content": f"Context:\n{r['passage']}\n\nQuestion: {r['question']}"}],
                "chosen": r["chosen"], "rejected": r["rejected"],
                "meta": {"source": r["source"], "judge_margin": jc[r["prompt_id"]]["margin"],
                         "prompt_id": r["prompt_id"]}}, ensure_ascii=False) + "\n")
    print(f"\nwrote {PREFS}: {len(kept)} preference triplets", flush=True)
    print(f"judge keep rate: {sum(1 for v in jc.values() if v['chosen_wins'] and v['margin']>=JUDGE_MARGIN_MIN)}/{len(jc)}")


if __name__ == "__main__":
    main()
