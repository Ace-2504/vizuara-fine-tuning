"""Smoke test — validate the teacher + the generate->judge path on a few real chunks,
and measure EXACT token cost so the full-build estimate becomes a real number.

Costs ~$0.05. Does NOT write the dataset. Run:
    python smoke.py
"""
from __future__ import annotations

import json
import math
import os

from transformers import AutoTokenizer

from teacher import TeacherClient, BudgetExhausted

CORPUS = "data/corpus"
TOK = AutoTokenizer.from_pretrained("data/tokenizer")
CHUNK_TOKENS = 256
N_CHUNKS = 3                    # one per source
QA_PER_CHUNK = 5

# Assumed Gemini text rates ($/1M tokens) — VERIFY against current pricing. Cost scales
# linearly from the exact token counts this script measures, so only the rate is uncertain.
RATES = {
    "gemini-3.1-flash":      {"in": 0.30, "out": 1.20},
    "gemini-3.1-flash-lite": {"in": 0.10, "out": 0.40},
}

QA_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "question": {"type": "string"}, "quote": {"type": "string"},
    "answer": {"type": "string"}, "difficulty": {"type": "string"}},
    "required": ["question", "quote", "answer", "difficulty"]}}

JUDGE_SCHEMA = {"type": "object", "properties": {
    "correct": {"type": "integer"}, "grounded": {"type": "boolean"},
    "reason": {"type": "string"}}, "required": ["correct", "grounded", "reason"]}


def first_chunk(source: str) -> str:
    path = os.path.join(CORPUS, source, sorted(os.listdir(os.path.join(CORPUS, source)))[0])
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            t = line.rstrip("\n")
            if t:
                ids = TOK(t, add_special_tokens=False)["input_ids"][:CHUNK_TOKENS]
                return TOK.decode(ids)
    return ""


def qa_prompt(passage: str, n: int) -> str:
    return (f"Read the PASSAGE and write {n} diverse question-answer pairs.\n"
            'Each item: "question" (self-contained, names the entity, no "the passage"), '
            '"quote" (EXACT verbatim span from the passage supporting the answer), '
            '"answer" (short, fully supported by the quote), "difficulty" (easy|medium|hard).\n'
            "Do NOT invent facts. No near-duplicate questions.\n\n"
            f"PASSAGE:\n{passage}")


def judge_prompt(passage: str, q: str, a: str) -> str:
    return ("Grade a synthetic training pair using ONLY the PASSAGE.\n"
            "- correct (1-5): does the ANSWER correctly and completely answer the QUESTION?\n"
            "- grounded (bool): is every claim in the ANSWER supported by the PASSAGE?\n"
            "Be strict; a fluent but unsupported/incomplete answer scores low.\n\n"
            f"PASSAGE:\n{passage}\n\nQUESTION: {q}\nANSWER: {a}")


def norm(s: str) -> str:
    return " ".join(s.split()).lower()


def main() -> None:
    t = TeacherClient()
    print(f"teacher primary: {t.model}\n")
    pairs, kept, quote_ok = [], 0, 0

    for src in ("sec", "case-law", "fineweb-edu"):
        passage = first_chunk(src)
        try:
            out = t.generate_json(qa_prompt(passage, QA_PER_CHUNK), QA_SCHEMA, temperature=0.9)
        except BudgetExhausted as e:
            print(f"\n[BUDGET] {e}\n-> a real run would checkpoint here and resume on top-up.")
            break
        out = out or []
        print(f"[{src}] model={t.model} got {len(out)} pairs")
        for p in out:
            q, quote, a = p.get("question", ""), p.get("quote", ""), p.get("answer", "")
            qok = norm(quote) in norm(passage)
            quote_ok += int(qok)
            pairs.append({"src": src, "passage": passage, "q": q, "quote": quote,
                          "a": a, "quote_ok": qok, "difficulty": p.get("difficulty")})

    gen_in, gen_out = t.usage.in_tokens, t.usage.out_tokens
    gen_req = t.usage.requests

    # Judge every generated pair (grounding + correctness).
    judged = []
    for p in pairs:
        try:
            v = t.generate_json(judge_prompt(p["passage"], p["q"], p["a"]), JUDGE_SCHEMA,
                                temperature=0.0)
        except BudgetExhausted as e:
            print(f"\n[BUDGET during judge] {e}"); break
        if v:
            keep = int(v.get("correct", 0)) >= 4 and bool(v.get("grounded"))
            kept += int(keep)
            judged.append({**p, "correct": v.get("correct"), "grounded": v.get("grounded"),
                           "keep": keep})

    # ---- report -------------------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"SMOKE RESULTS  (model(s) used: {', '.join(t.usage.by_model)})")
    print("=" * 68)
    print(f"pairs generated:     {len(pairs)}")
    print(f"quote verified:      {quote_ok}/{len(pairs)}  "
          f"({100*quote_ok/max(len(pairs),1):.0f}% exact-substring)")
    print(f"judge kept (>=4 & grounded): {kept}/{len(judged)}")
    print(f"\ntotal requests: {t.usage.requests}  retries: {t.usage.retries}")
    print(f"tokens  in: {t.usage.in_tokens:,}   out: {t.usage.out_tokens:,}")
    judge_in = t.usage.in_tokens - gen_in
    judge_out = t.usage.out_tokens - gen_out
    print(f"  generation  in={gen_in:,} out={gen_out:,}  ({gen_req} calls)")
    print(f"  judging     in={judge_in:,} out={judge_out:,}  ({t.usage.requests-gen_req} calls)")

    kept_pairs = max(kept, 1)
    tok_per_kept = t.usage.in_tokens + t.usage.out_tokens
    print(f"\nper KEPT pair: ~{tok_per_kept/kept_pairs:.0f} tokens "
          f"(in+out, gen+judge, at this keep rate)")

    # cost per model actually used
    def cost(model, i, o):
        r = RATES.get(model, RATES["gemini-3.1-flash-lite"])
        return i/1e6*r["in"] + o/1e6*r["out"]
    total_cost = sum(cost(m, d["in"], d["out"]) for m, d in t.usage.by_model.items())
    print(f"\nsmoke cost (assumed rates): ${total_cost:.4f}")
    if kept:
        per_kept_cost = total_cost / kept_pairs
        print(f"per kept pair: ${per_kept_cost:.5f}")
        for target, label in [(25000, "25k kept (15k QA + 10k RAFT)")]:
            print(f"  -> extrapolated full build ({label}): ${per_kept_cost*target:.2f}")
    print("\nNOTE: token counts are EXACT (from usage_metadata); dollar figures use the "
          "assumed RATES above — verify current Gemini pricing and rescale linearly.")

    with open("smoke_result.json", "w", encoding="utf-8") as f:
        json.dump({"pairs": judged, "usage": t.usage.__dict__,
                   "cost_assumed": total_cost}, f, indent=2, default=str)
    print("wrote smoke_result.json")


if __name__ == "__main__":
    main()
