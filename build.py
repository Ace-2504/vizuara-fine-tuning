"""Gauntlet + assembly: raw.jsonl -> qa_sft.jsonl + raft.jsonl + eval.jsonl.

Stages: rule filters -> grounding -> dedup (exact + embedding) -> LLM judge (resumable) ->
eval holdout (doc_id quarantine, matched conditions) -> decontaminate -> RAFT assembly
(hard distractors, doc_id exclusion) -> diversity report -> write chat JSONL (text-level).

    python build.py            # run all stages; judge stage is resumable + budget-aware
Judge verdicts cache to provenance/judge.jsonl, so a credit-out during judging RESUMES.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import config as C
import prompts as P
from chunker import token_len
from teacher import TeacherClient, BudgetExhausted

RNG = random.Random(C.SEED)

SYS_GROUNDED = ("You are a precise legal and financial assistant. Answer clearly using the "
                "provided context; do not invent facts.")
SYS_CLOSED = "You are a precise legal and financial assistant. Answer clearly and accurately."
SYS_RAFT = ("You are a grounded assistant. Answer using ONLY the provided documents. Quote the "
            "exact supporting text, then give your answer. If the answer is not in the documents, "
            f"reply exactly: {C.ABSTAIN_STRING}")

STOPWORDS = set("the a an of to in and or for on at by is are was were be been being with as "
                "that this it its from which who whom whose what when where how".split())


def norm(s: str) -> str:
    return " ".join(s.split()).lower()


def content_words(s: str) -> set[str]:
    return {w for w in norm(s).replace(",", " ").replace(".", " ").split()
            if w and w not in STOPWORDS and len(w) > 2}


# ---------------- stage 1: rule filters ----------------
def rule_ok(r: dict) -> bool:
    q, a = r.get("question", ""), r.get("answer", "")
    if not (C.MIN_QUESTION_CHARS <= len(q)):
        return False
    if not (C.MIN_ANSWER_CHARS <= len(a) <= C.MAX_ANSWER_CHARS):
        return False
    low = norm(q) + " " + norm(a)
    if any(p in low for p in ("the passage", "the text", "the document", "as stated above",
                              "here are", "provided text")):
        return False
    if r["task"] == "qa":
        if not r.get("quote") or norm(r["quote"]) not in norm(r["passage"]):
            return False
        if token_len(r["quote"]) > C.MAX_QUOTE_TOKENS:
            return False
    return True


# ---------------- stage 2: grounding ----------------
def grounded_ok(r: dict) -> bool:
    if r["task"] == "unanswerable":
        return True
    if r["answer"].strip() == C.ABSTAIN_STRING:
        return True
    aw = content_words(r["answer"])
    if not aw:
        return False
    pw = content_words(r["passage"])
    return len(aw & pw) / len(aw) >= C.GROUNDING_MIN


# ---------------- stage 3: dedup ----------------
def dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:                              # exact question hash
        h = hashlib.sha256(norm(r["question"]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h); out.append(r)
    print(f"  after exact dedup: {len(out)}")
    # embedding near-dup
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  embedding {len(out)} questions on {dev} ...")
    emb = SentenceTransformer(C.EMBED_MODEL, device=dev).encode(
        [r["question"] for r in out], normalize_embeddings=True, batch_size=256,
        show_progress_bar=False).astype(np.float32)
    keep, kept_idx = [], []
    for i in range(len(out)):
        if kept_idx:
            sims = emb[kept_idx] @ emb[i]
            if float(sims.max()) >= C.DEDUP_COSINE:
                continue
        keep.append(out[i]); kept_idx.append(i)
    print(f"  after embedding dedup ({C.DEDUP_COSINE}): {len(keep)}")
    for r, i in zip(keep, kept_idx):
        r["_emb"] = emb[i]
    return keep


# ---------------- stage 4: judge (resumable, budget-aware) ----------------
def judge_key(r: dict) -> str:
    return hashlib.sha256((r["chunk_id"] + "|" + norm(r["question"])).encode()).hexdigest()[:20]


def judge_all(rows: list[dict]) -> dict:
    cache = {}
    if C.JUDGE_CACHE.exists():
        for l in open(C.JUDGE_CACHE, encoding="utf-8"):
            v = json.loads(l); cache[v["k"]] = v
    todo = [r for r in rows if r["task"] != "unanswerable" and judge_key(r) not in cache
            and r["answer"].strip() != C.ABSTAIN_STRING]
    print(f"  judge: {len(cache)} cached, {len(todo)} to score")
    if not todo:
        return cache
    teachers = [TeacherClient(models=("gemini-3.1-flash-lite",), min_interval_s=0.0)
                for _ in range(8)]
    lock = threading.Lock(); stop = threading.Event()
    fh = open(C.JUDGE_CACHE, "a", encoding="utf-8")

    def work(i, r):
        if stop.is_set():
            return
        try:
            v = teachers[i % 8].generate_json(P.judge(r["passage"], r["question"], r["answer"]),
                                              P.JUDGE_SCHEMA, temperature=C.JUDGE_TEMPERATURE)
        except BudgetExhausted:
            stop.set(); raise
        if v:
            rec = {"k": judge_key(r), "correct": int(v.get("correct", 0)),
                   "grounded": bool(v.get("grounded"))}
            with lock:
                cache[rec["k"]] = rec
                fh.write(json.dumps(rec) + "\n"); fh.flush()

    hit = False
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(work, i, r) for i, r in enumerate(todo)]
            for n, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except BudgetExhausted:
                    hit = True; break
                if n % 200 == 0:
                    print(f"    judged {n}/{len(todo)}", flush=True)
    finally:
        fh.close()
    if hit:
        print("  [BUDGET] judging paused — verdicts cached. Top up and re-run build.py to resume.")
        raise SystemExit(2)
    return cache


def keep_by_judge(r: dict, cache: dict) -> bool:
    if r["task"] == "unanswerable" or r["answer"].strip() == C.ABSTAIN_STRING:
        return True
    v = cache.get(judge_key(r))
    return bool(v) and v["correct"] >= C.JUDGE_KEEP_CORRECT and v["grounded"]


# ---------------- stage 5: eval holdout ----------------
def carve_eval(pool: list[dict]) -> tuple[list[dict], set]:
    qa = [r for r in pool if r["task"] == "qa"]
    RNG.shuffle(qa)
    picked, quar = [], set()
    for r in qa:
        if len(picked) >= C.EVAL_SIZE:
            break
        if r["doc_id"] in quar:
            continue
        picked.append(r); quar.add(r["doc_id"])
    return picked, quar


# ---------------- RAFT distractor selection ----------------
def build_index(pool: list[dict]):
    emb = np.stack([r["_emb"] for r in pool]).astype(np.float32)
    return emb


def pick_distractors(i, emb, pool, k, golden_doc):
    sims = emb @ emb[i]; sims[i] = -1
    out = []
    for j in np.argsort(-sims):
        if pool[j]["doc_id"] == golden_doc:
            continue
        out.append(j)
        if len(out) == k:
            break
    return out


# ---------------- rendering ----------------
def chat(system, user, assistant, meta):
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}], "meta": meta}


def main():
    raw = [json.loads(l) for l in open(C.RAW, encoding="utf-8")]
    print(f"raw pairs: {len(raw)}")
    r1 = [r for r in raw if rule_ok(r)]
    print(f"after rule filters: {len(r1)}")
    r2 = [r for r in r1 if grounded_ok(r)]
    print(f"after grounding: {len(r2)}")
    r3 = dedup(r2)
    cache = judge_all(r3)
    r4 = [r for r in r3 if keep_by_judge(r, cache)]
    print(f"after judge (>= {C.JUDGE_KEEP_CORRECT} & grounded): {len(r4)}")

    eval_pool, quar = carve_eval([r for r in r4 if r["task"] == "qa"])
    train = [r for r in r4 if r["doc_id"] not in quar]
    print(f"eval held out: {len(eval_pool)} ({len(quar)} docs quarantined) | train pool: {len(train)}")

    # ---- QA-SFT set ----
    qa_sft = []
    qa_rows = [r for r in train if r["task"] == "qa"]
    aux_rows = [r for r in train if r["task"] in ("summarize", "extract", "rewrite")]
    RNG.shuffle(qa_rows)
    n_closed = int(len(qa_rows) * C.CLOSED_BOOK_FRAC)
    for n, r in enumerate(qa_rows):
        closed = n < n_closed and r["source"] == "fineweb-edu"
        if closed:
            qa_sft.append(chat(SYS_CLOSED, r["question"], r["answer"],
                               {"task": "qa", "mode": "closed_book", "source": r["source"],
                                "difficulty": r["difficulty"], "chunk_id": r["chunk_id"]}))
        else:
            qa_sft.append(chat(SYS_GROUNDED, f"Context:\n{r['passage']}\n\nQuestion: {r['question']}",
                               r["answer"], {"task": "qa", "mode": "grounded", "source": r["source"],
                                             "difficulty": r["difficulty"], "chunk_id": r["chunk_id"]}))
    for r in aux_rows:
        qa_sft.append(chat(SYS_GROUNDED, r["question"], r["answer"],
                           {"task": r["task"], "mode": "grounded", "source": r["source"],
                            "difficulty": r["difficulty"], "chunk_id": r["chunk_id"]}))
    RNG.shuffle(qa_sft)
    qa_sft = qa_sft[:C.QA_SFT_SIZE]

    # ---- RAFT set ----
    golden = [r for r in train if r["task"] == "qa"]
    unans = [r for r in train if r["task"] == "unanswerable"]
    emb = build_index(golden)
    raft = []
    n_golden = int(C.RAFT_SIZE * C.RAFT_P_GOLDEN)
    n_absent = int(n_golden * C.ANSWERABLE_ABSENT_FRAC)
    for i, r in enumerate(golden[:n_golden]):
        dists = pick_distractors(i, emb, golden, C.RAFT_K, r["doc_id"])
        docs = [golden[j]["passage"] for j in dists]
        answerable_absent = i < n_absent and unans
        if answerable_absent:                                    # golden present, but ask an absent Q
            uq = unans[i % len(unans)]["question"]
            docs.insert(RNG.randrange(len(docs) + 1), r["passage"])
            ctx = "\n\n".join(f"[Document {d+1}]\n{t}" for d, t in enumerate(docs))
            raft.append(chat(SYS_RAFT, f"{ctx}\n\nQuestion: {uq}", C.ABSTAIN_STRING,
                             {"mode": "raft", "golden_present": True, "answerable": False,
                              "source": r["source"], "chunk_id": r["chunk_id"], "doc_id": r["doc_id"]}))
        else:
            docs.insert(RNG.randrange(len(docs) + 1), r["passage"])
            ctx = "\n\n".join(f"[Document {d+1}]\n{t}" for d, t in enumerate(docs))
            ans = f"##begin_quote## {r['quote']} ##end_quote##\n{r['answer']}"
            raft.append(chat(SYS_RAFT, f"{ctx}\n\nQuestion: {r['question']}", ans,
                             {"mode": "raft", "golden_present": True, "answerable": True,
                              "source": r["source"], "chunk_id": r["chunk_id"], "doc_id": r["doc_id"]}))
    # distractors-only abstention
    n_distractoronly = C.RAFT_SIZE - len(raft)
    for i in range(n_distractoronly):
        base = golden[(n_golden + i) % len(golden)]
        dists = pick_distractors((n_golden + i) % len(golden), emb, golden, C.RAFT_K, base["doc_id"])
        docs = [golden[j]["passage"] for j in dists]
        ctx = "\n\n".join(f"[Document {d+1}]\n{t}" for d, t in enumerate(docs))
        raft.append(chat(SYS_RAFT, f"{ctx}\n\nQuestion: {base['question']}", C.ABSTAIN_STRING,
                         {"mode": "raft", "golden_present": False, "answerable": False,
                          "source": base["source"], "chunk_id": base["chunk_id"], "doc_id": base["doc_id"]}))
    RNG.shuffle(raft)
    raft = raft[:C.RAFT_SIZE]

    # ---- eval (matched conditions) ----
    eval_rows = []
    ei = {r["chunk_id"]: i for i, r in enumerate(golden)}
    for r in eval_pool:
        idx = golden.index(r) if r in golden else None
        gp = r["passage"]
        # find distractors from the training index by embedding of this eval question
        q_emb = r.get("_emb")
        if q_emb is not None:
            sims = emb @ q_emb;
            dd = [j for j in np.argsort(-sims) if golden[j]["doc_id"] != r["doc_id"]][:C.RAFT_K]
        else:
            dd = list(range(C.RAFT_K))
        dtexts = [golden[j]["passage"] for j in dd]
        def ctx_of(docs):
            return "\n\n".join(f"[Document {d+1}]\n{t}" for d, t in enumerate(docs))
        base_meta = {"pair_id": r["chunk_id"], "source": r["source"]}
        # 4 matched conditions
        eval_rows.append(chat(SYS_RAFT, f"{ctx_of([gp])}\n\nQuestion: {r['question']}", r["answer"],
                              {**base_meta, "cond": "clean"}))
        docs = dtexts + [gp]; RNG.shuffle(docs)
        eval_rows.append(chat(SYS_RAFT, f"{ctx_of(docs)}\n\nQuestion: {r['question']}", r["answer"],
                              {**base_meta, "cond": "realistic"}))
        eval_rows.append(chat(SYS_RAFT, f"{ctx_of(dtexts)}\n\nQuestion: {r['question']}", C.ABSTAIN_STRING,
                              {**base_meta, "cond": "retrieval_failure"}))
        eval_rows.append(chat(SYS_CLOSED, r["question"], r["answer"], {**base_meta, "cond": "closed_book"}))

    # ---- write ----
    _write(C.OUT / "qa_sft.jsonl", qa_sft)
    _write(C.OUT / "raft.jsonl", raft)
    _write(C.OUT / "eval.jsonl", eval_rows)
    diversity_report(qa_sft, raft)


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            r = {k: v for k, v in r.items()}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path.name}: {len(rows)}")


def diversity_report(qa_sft, raft):
    def distinct_n(texts, n):
        grams, tot = set(), 0
        for t in texts:
            toks = norm(t).split()
            for i in range(len(toks) - n + 1):
                grams.add(tuple(toks[i:i + n])); tot += 1
        return len(grams) / max(tot, 1)
    qs = [m["messages"][1]["content"] for m in qa_sft]
    print("\n=== diversity (QA-SFT questions) ===")
    print(f"  distinct-2: {distinct_n(qs,2):.3f}  distinct-3: {distinct_n(qs,3):.3f}")
    print("  task mix:", dict(collections.Counter(m["meta"]["task"] for m in qa_sft)))
    print("  source:", dict(collections.Counter(m["meta"]["source"] for m in qa_sft)))
    print("  mode:", dict(collections.Counter(m["meta"]["mode"] for m in qa_sft)))
    print("  RAFT modes:", dict(collections.Counter(
        (m["meta"]["golden_present"], m["meta"]["answerable"]) for m in raft)))


if __name__ == "__main__":
    main()
