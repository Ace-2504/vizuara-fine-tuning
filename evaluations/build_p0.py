"""P0 — master score export: one row per (model x question), all 15 models.

Joins:
  * rubric parts + total + grounded  <- rubric10.jsonl (13) + rubric10_500m_new.jsonl (2 new)
  * fabricated flag + answer + gold + question + source  <- eval_results/<model>.judged.json

Output: eval_results/P0_master_scores.{json,csv}  (question_id = pair_id, shared across models)
This is the single artifact Exp 1/2/3/4/7 read. Nothing here is published.
"""
from __future__ import annotations
import csv, json, io, os

ER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_results")

# arena id -> (size, stage, per-item judged file)
MODELS = [
    ("125m-base","125M","Base","base-125m.judged.json"),
    ("125m-qa","125M","QA-SFT","slm-125m-sft.judged.json"),
    ("125m-raft","125M","RAFT","slm-125m-raft.judged.json"),
    ("125m-dpo","125M","DPO","slm-125m-sft-dpo.judged.json"),
    ("125m-rlaif","125M","RLAIF","slm-125m-sft-rlaif.judged.json"),
    ("500m-base","500M","Base","base-500m.judged.json"),
    ("500m-qa","500M","QA-SFT","slm-500m-sft.judged.json"),
    ("500m-raft","500M","RAFT","slm-500m-raft.judged.json"),
    ("500m-dpo","500M","DPO","slm-500m-sft-dpo.judged.json"),
    ("500m-rlaif","500M","RLAIF","slm-500m-sft-rlaif.judged.json"),
    ("gemma-base","Gemma 2B","Base","base-gemma.judged.json"),
    ("gemma-qa","Gemma 2B","QA-SFT","gemma-2-2b-sft.judged.json"),
    ("gemma-raft","Gemma 2B","RAFT","gemma-2-2b-raft.judged.json"),
    ("gemma-dpo","Gemma 2B","DPO","gemma-2-2b-sft-dpo.judged.json"),
    ("gemma-rlaif","Gemma 2B","RLAIF","gemma-2-2b-sft-rlaif.judged.json"),
]


def load_rubric():
    """pair_id -> mid -> {score, parts{...}, grounded}. Later lines win (retries)."""
    r = {}
    for fn in ("rubric10.jsonl", "rubric10_500m_new.jsonl"):
        p = os.path.join(ER, fn)
        if not os.path.exists(p):
            continue
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pid = rec["pair_id"]
            r.setdefault(pid, {}).update(rec.get("graded", {}))
    return r


def qtext(user):
    i = user.rfind("Question:")
    return user[i + len("Question:"):].strip() if i >= 0 else user.strip()


def main():
    rub = load_rubric()
    rows = []
    for mid, size, stage, fn in MODELS:
        d = json.load(io.open(os.path.join(ER, fn), encoding="utf-8"))
        items = {it["pair_id"]: it for it in d["per_item"] if it["cond"] == "clean"}
        for pid, it in items.items():
            g = (rub.get(pid, {}) or {}).get(mid)
            if not g:
                continue                      # no rubric10 score for this model/question
            parts = g.get("parts") or {}
            rows.append({
                "model": mid, "size": size, "stage": stage,
                "question_id": pid, "source": it.get("source"),
                "correctness": parts.get("correctness"),
                "completeness": parts.get("completeness"),
                "groundedness": parts.get("groundedness"),
                "clarity": parts.get("clarity"),
                "total": g.get("score"),
                "grounded": bool(g.get("grounded")),
                "fabricated": bool(it.get("scores", {}).get("fabrication")),
                "question": qtext(it.get("user", "")),
                "answer_text": (it.get("resp") or "").strip(),
                "gold_text": (it.get("ref") or "").strip(),
            })
    # stable order: model, then question
    rows.sort(key=lambda r: (r["model"], r["question_id"]))
    json.dump(rows, io.open(os.path.join(ER, "P0_master_scores.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    cols = ["model","size","stage","question_id","source","correctness","completeness",
            "groundedness","clarity","total","grounded","fabricated","question","answer_text","gold_text"]
    with io.open(os.path.join(ER, "P0_master_scores.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    # report
    from collections import Counter
    per_model = Counter(r["model"] for r in rows)
    print(f"P0 written: {len(rows)} rows across {len(per_model)} models")
    miss = [m[0] for m in MODELS if per_model.get(m[0], 0) == 0]
    print("  rows/model:", dict(per_model))
    print("  models with 0 rows:", miss or "none")
    print("  columns:", ", ".join(cols))


if __name__ == "__main__":
    main()
