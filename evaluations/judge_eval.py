"""LLM-as-judge over the saved eval generations — the fair, cross-family quality metric.

Why this exists: token-F1 punishes correct paraphrases, and the reward model is both
family-biased (a head on the 500M SFT backbone) and CIRCULAR for the RLAIF models (they
were trained to maximise it). An independent Gemini judge, blind to which model produced
each answer, restores a headline quality signal that is comparable across the 125M / 500M
/ Gemma families and is not the RLAIF training objective.

Architecture: generation + GPU work already happened on Modal (`eval.py`); this runs
LOCALLY over the saved per-item outputs, so no GPU and no API key on Modal. Resumable and
budget-aware exactly like `build_prefs.py` — a credit-out checkpoints and a re-run resumes.

    # after `modal run eval.py --all` and `modal volume get ft-data /eval ./eval_results`
    python evaluations/judge_eval.py ./eval_results --set all          # judge every version
    python evaluations/judge_eval.py ./eval_results --set set1         # one experiment
    python evaluations/judge_eval.py ./eval_results --sample 200       # 200 items/condition (cheaper)

Writes `<version>.judged.json` next to each input (per_item augmented with a `judge`
block) and appends to a shared `_judge_cache.jsonl` so identical responses are never
re-judged. Deterministic given the cache; judge temperature is low for stability.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root -> teacher.py
sys.path.insert(0, _HERE)                    # evaluations/ -> experiments.py
from teacher import TeacherClient, BudgetExhausted  # noqa: E402
from experiments import EXPERIMENTS, META  # noqa: E402

ABSTAIN = "not stated in the context"

JUDGE_SCHEMA = {"type": "object", "properties": {
    "correct": {"type": "integer"},        # 1..5, how well it answers vs the reference
    "grounded": {"type": "boolean"},        # every claim supported by the context (no fabrication)
    "matches_ref": {"type": "boolean"},     # semantically equivalent to the reference answer
    "reason": {"type": "string"}},
    "required": ["correct", "grounded", "matches_ref"]}


def judge_prompt(context_and_question: str, reference: str, candidate: str, answerable: bool) -> str:
    """Model-blind, rubric-based, pointwise. The reference is provided so 'correct' is
    graded against a known-good answer rather than the judge's own guess. Abstention is
    handled explicitly so the four RAFT conditions share one judge."""
    if answerable:
        rule = ("The QUESTION is answerable from the CONTEXT. The REFERENCE is a correct "
                "short answer. Grade the CANDIDATE against it.")
    else:
        rule = (f"The QUESTION is NOT answerable from the CONTEXT. The only correct behaviour "
                f"is to decline, e.g. '{ABSTAIN}'. A CANDIDATE that invents an answer is wrong "
                f"and ungrounded; a CANDIDATE that abstains is correct and grounded.")
    return (
        "You are a strict evaluator of a grounded question-answering system. Judge ONLY the "
        "CANDIDATE answer; you do not know which system produced it.\n"
        f"{rule}\n\n"
        "Return JSON:\n"
        '- "correct" (1-5): 5 = fully correct and complete, 1 = wrong/irrelevant. Judge meaning, '
        "not wording — a correct paraphrase scores high.\n"
        '- "grounded" (bool): true only if every factual claim is supported by the CONTEXT '
        "(for an unanswerable item, an abstention is grounded).\n"
        '- "matches_ref" (bool): is the CANDIDATE semantically equivalent to the REFERENCE?\n'
        '- "reason": one short sentence.\n\n'
        f"CONTEXT + QUESTION:\n{context_and_question}\n\n"
        f"REFERENCE:\n{reference}\n\n"
        f"CANDIDATE:\n{candidate}"
    )


def item_key(version: str, cond: str, pair_id: str, resp: str) -> str:
    h = hashlib.sha1(f"{version}|{cond}|{pair_id}|{resp}".encode("utf-8")).hexdigest()
    return h


def load_cache(path: str) -> dict:
    cache = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                cache[r["key"]] = r["judge"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def target_versions(which: str, results_dir: str) -> list[str]:
    if which == "all":
        found = [os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(results_dir, "*.json"))
                 if not p.endswith(".judged.json") and not os.path.basename(p).startswith("_")]
        return sorted(found)
    if which in EXPERIMENTS:
        return list(EXPERIMENTS[which])
    return [which]  # a single explicit version name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="./eval_results")
    ap.add_argument("--set", dest="which", default="all",
                    help="all | set1 | set2 | <single version name>")
    ap.add_argument("--sample", type=int, default=0,
                    help="judge at most N items per condition (0 = all); items are the first N "
                         "by pair_id order, so the subsample is identical across models -> the "
                         "cross-model comparison stays paired.")
    ap.add_argument("--temperature", type=float, default=0.1)
    a = ap.parse_args()

    cache_path = os.path.join(a.results_dir, "_judge_cache.jsonl")
    cache = load_cache(cache_path)
    teacher = TeacherClient()
    cache_fh = open(cache_path, "a", encoding="utf-8")

    versions = target_versions(a.which, a.results_dir)
    print(f"judging {len(versions)} version(s) from {a.results_dir} "
          f"(cache has {len(cache)} entries)", flush=True)

    n_new = 0
    try:
        for version in versions:
            src = os.path.join(a.results_dir, f"{version}.json")
            if not os.path.exists(src):
                print(f"  [skip] {version}: no {src}", flush=True)
                continue
            blob = json.load(open(src, encoding="utf-8"))
            per_item = blob.get("per_item", [])
            if per_item and "user" not in per_item[0]:
                print(f"  [skip] {version}: per_item lacks 'user'/'answerable' — re-run the "
                      f"updated eval.py so the judge has the context.", flush=True)
                continue

            # optional identical-across-models subsample: first N pair_ids per condition
            if a.sample > 0:
                seen = {}
                kept = []
                for it in per_item:
                    c = it["cond"]
                    seen[c] = seen.get(c, 0)
                    if seen[c] < a.sample:
                        kept.append(it); seen[c] += 1
                per_item = kept

            judged = 0
            for it in per_item:
                key = item_key(version, it["cond"], it["pair_id"], it["resp"])
                if key in cache:
                    it["judge"] = cache[key]
                    continue
                answerable = it.get("answerable", it["cond"] != "retrieval_failure")
                out = teacher.generate_json(
                    judge_prompt(it["user"], it["ref"], it["resp"], answerable),
                    JUDGE_SCHEMA, temperature=a.temperature)
                if out is None:  # skipped (empty/unparseable) — leave unjudged, retry next run
                    continue
                jr = {"correct": int(out.get("correct", 0)),
                      "grounded": bool(out.get("grounded", False)),
                      "matches_ref": bool(out.get("matches_ref", False)),
                      "model": teacher.model}
                it["judge"] = jr
                cache[key] = jr
                cache_fh.write(json.dumps({"key": key, "judge": jr}) + "\n"); cache_fh.flush()
                judged += 1; n_new += 1

            out_path = os.path.join(a.results_dir, f"{version}.judged.json")
            json.dump(blob, open(out_path, "w", encoding="utf-8"), indent=2)
            print(f"  [{version}] judged {judged} new, wrote {out_path}", flush=True)
    except BudgetExhausted as e:
        print(f"\nBUDGET EXHAUSTED: {str(e)[:160]}\nProgress cached to {cache_path}. "
              f"Re-run the same command to resume.", flush=True)
    finally:
        cache_fh.close()
        u = teacher.usage
        print(f"judge usage: {u.requests} requests, {u.in_tokens:,} in / {u.out_tokens:,} out "
              f"tokens, {u.retries} retries, {n_new} newly judged", flush=True)


if __name__ == "__main__":
    main()
