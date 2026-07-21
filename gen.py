"""Generation driver — concurrent, resumable, budget-aware.

Walks the corpus, assigns each consumed chunk one recipe (weighted to hit the task mix),
calls the teacher, and appends raw pairs to data/sft/raw.jsonl as they arrive. Resumable by
(chunk_id, task) via gen_state.json; halts cleanly and saves state when the balance is spent.

    python gen.py                 # full run (targets in config)
    python gen.py --limit 120     # small validation run (~a few hundred pairs)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as C
import prompts as P
from chunker import chunks_of_shard, shards, token_len
from teacher import TeacherClient, BudgetExhausted

WORKERS = 8
TEACHER_MODELS = ("gemini-3.1-flash-lite",)   # smoke test: full flash 404s on this key

_lock = threading.Lock()
_stop = threading.Event()


def raw_target(task: str) -> int:
    """How many raw pairs of each task to aim for (pre-gauntlet)."""
    if task == "qa":
        # qa pool feeds RAFT golden + QA-SFT qa slice
        net = C.RAFT_SIZE * C.RAFT_P_GOLDEN + C.QA_SFT_SIZE * C.TASK_MIX["qa"]
    elif task == "unanswerable":
        net = C.RAFT_SIZE * C.RAFT_ABSTAIN_FRAC + \
              C.RAFT_SIZE * C.RAFT_P_GOLDEN * C.ANSWERABLE_ABSENT_FRAC
    else:
        net = C.QA_SFT_SIZE * C.TASK_MIX[task]
    return math.ceil(net / C.KEEP_ASSUMED)


def plan_tasks() -> list[str]:
    """A shuffled schedule of per-chunk recipes sized to the raw targets."""
    per_chunk = {"qa": C.QA_PER_CHUNK, "summarize": 1, "extract": 1, "rewrite": 1,
                 "unanswerable": 3}
    sched = []
    for task, per in per_chunk.items():
        sched += [task] * math.ceil(raw_target(task) / per)
    random.Random(C.SEED).shuffle(sched)
    return sched


def _source_chunks(source):
    """All chunks of one source, shard by shard. `source` is a bound arg -> no closure bug."""
    for sh in shards(source):
        yield from chunks_of_shard(source, sh)


def iter_chunks():
    """Round-robin chunks across sources weighted by the domain mix."""
    active = {s: _source_chunks(s) for s in C.SOURCES}
    acc = {s: 0.0 for s in C.SOURCES}
    exhausted = set()
    while len(exhausted) < len(C.SOURCES):
        s = max((k for k in C.SOURCES if k not in exhausted),
                key=lambda k: acc[k] + C.DOMAIN_MIX[k])
        acc[s] -= 1.0
        for k in C.SOURCES:
            acc[k] += C.DOMAIN_MIX[k]
        try:
            yield next(active[s])
        except StopIteration:
            exhausted.add(s)


def _gen_one(teacher, chunk, task):
    """One API call -> list of raw records (or [])."""
    recs = []
    try:
        if task == "qa":
            out = teacher.generate_json(P.qa(chunk.text), P.QA_SCHEMA, temperature=C.GEN_TEMPERATURE)
            for p in out or []:
                recs.append(_rec(chunk, "qa", p.get("question", ""), p.get("answer", ""),
                                 quote=p.get("quote", ""), difficulty=p.get("difficulty", "medium")))
        elif task == "unanswerable":
            out = teacher.generate_json(P.unanswerable(chunk.text), P.UNANSWERABLE_SCHEMA,
                                        temperature=C.GEN_TEMPERATURE)
            for p in out or []:
                recs.append(_rec(chunk, "unanswerable", p.get("question", ""),
                                 C.ABSTAIN_STRING, quote="", difficulty="medium"))
        else:  # summarize / extract / rewrite
            out = teacher.generate_json(P.aux(task, chunk.text), P.AUX_SCHEMA,
                                        temperature=C.GEN_TEMPERATURE)
            if out:
                recs.append(_rec(chunk, task, out.get("instruction", ""), out.get("answer", ""),
                                 quote="", difficulty="medium"))
    except BudgetExhausted:
        _stop.set()
        raise
    return recs


def _rec(chunk, task, q, a, *, quote, difficulty):
    return {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "source": chunk.source,
            "task": task, "question": q, "answer": a, "quote": quote,
            "difficulty": difficulty, "passage": chunk.text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after N chunks (validation)")
    args = ap.parse_args()

    C.OUT.mkdir(parents=True, exist_ok=True)
    C.PROV.mkdir(parents=True, exist_ok=True)

    state = json.load(open(C.GEN_STATE)) if C.GEN_STATE.exists() else {"done": []}
    done = set(tuple(x) for x in state["done"])
    raw_fh = open(C.RAW, "a", encoding="utf-8")

    sched = plan_tasks()
    print(f"raw targets: " + ", ".join(f"{t}={raw_target(t)}" for t in
          ("qa", "summarize", "extract", "rewrite", "unanswerable")))
    print(f"scheduled chunks: {len(sched)} | workers: {WORKERS} | already done: {len(done)}")

    # build the work list (chunk, task), skipping done, up to --limit
    work = []
    chunk_iter = iter_chunks()
    for task in sched:
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            break
        if (chunk.chunk_id, task) in done:
            continue
        work.append((chunk, task))
        if args.limit and len(work) >= args.limit:
            break
    print(f"work items this run: {len(work)}")

    teachers = [TeacherClient(models=TEACHER_MODELS, min_interval_s=0.0) for _ in range(WORKERS)]
    kept = [0]
    total_tok = [0]

    def run(i, chunk, task):
        if _stop.is_set():
            return
        recs = _gen_one(teachers[i % WORKERS], chunk, task)
        with _lock:
            for r in recs:
                raw_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            raw_fh.flush()
            done.add((chunk.chunk_id, task))
            kept[0] += len(recs)

    budget_hit = False
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(run, i, c, t): (c, t) for i, (c, t) in enumerate(work)}
            for n, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except BudgetExhausted:
                    budget_hit = True
                    break
                if n % 100 == 0:
                    print(f"  {n}/{len(work)} calls done | raw pairs so far {kept[0]}", flush=True)
    finally:
        state["done"] = [list(x) for x in done]
        json.dump(state, open(C.GEN_STATE, "w"))
        raw_fh.close()
        tot_in = sum(t.usage.in_tokens for t in teachers)
        tot_out = sum(t.usage.out_tokens for t in teachers)
        cost = tot_in/1e6*0.10 + tot_out/1e6*0.40
        print(f"\nraw pairs added this run: {kept[0]}")
        print(f"tokens in={tot_in:,} out={tot_out:,} | est cost this run ~${cost:.3f} (lite rates)")
        if budget_hit or _stop.is_set():
            print("\n[BUDGET] balance spent — state saved. Top up and re-run `python gen.py` to RESUME.")
        else:
            print("\ngeneration pass complete for scheduled work.")


if __name__ == "__main__":
    main()
