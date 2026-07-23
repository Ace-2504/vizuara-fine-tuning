"""Local, power-cut-resilient eval runner — for the Gemma models on an RTX 3060.

Runs the SAME evaluation as evaluations/eval.py, but on the local GPU in bf16 with no
Modal, for when cloud credits are exhausted. Only the 4 gemma versions need this (the 9
others are already in eval_results/). Output format is IDENTICAL to eval.py, so
judge_eval.py and eval_report.py consume it unchanged.

RESILIENCE (survives power loss / Ctrl-C / crash / OOM restart):
  * per-ITEM checkpoint — every generation is appended and fsync'd to
    eval_results/_partial/<version>.jsonl the instant it is produced, so a crash loses
    at most the single in-flight item.
  * resume — on restart, already-done (cond, pair_id) items are loaded and skipped, so
    the run continues exactly where it stopped; it never repeats finished work.
  * atomic finalize — a completed version is written to eval_results/<version>.json via a
    temp file + os.replace; if that file exists the whole version is skipped.
Re-running the exact same command after ANY interruption resumes. See PRECAUTIONS below.

    python evaluations/eval_local.py                         # all 4 gemma versions
    python evaluations/eval_local.py --versions gemma-2-2b-sft gemma-2-2b-raft
    python evaluations/eval_local.py --no-cache              # if the KV cache errors on gemma-2

Paths are the local migrate/ assets pulled off the (now spend-limited) Modal volume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL_PATH = os.path.join(ROOT, "migrate", "sft", "eval.jsonl")
SFT_PATH = os.path.join(ROOT, "migrate", "sft", "qa_sft.jsonl")
CKPT = os.path.join(ROOT, "migrate", "checkpoints")
OUT = os.path.join(ROOT, "eval_results")
PARTIAL = os.path.join(OUT, "_partial")
ABSTAIN = "not stated in the context"
GEMMA_VERSIONS = ["gemma-2-2b-sft", "gemma-2-2b-raft", "gemma-2-2b-sft-dpo", "gemma-2-2b-sft-rlaif"]


# ---------- metrics (identical to eval.py) ----------
def _words(s):
    return [w for w in "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower()).split()]

def token_f1(pred, ref):
    p, r = _words(pred), _words(ref)
    if not p or not r:
        return 0.0
    common = {}
    for w in p:
        common[w] = common.get(w, 0) + 1
    overlap = sum(min(cnt, r.count(w)) for w, cnt in common.items())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)

def exact_match(pred, ref):
    n = lambda s: " ".join(_words(s))
    return float(n(pred) == n(ref))

def _norm_num(s):
    return s.replace(",", "").rstrip("%").rstrip(".")

def fabrication(pred, ctx):
    ctx_norm = _norm_num(ctx.replace(",", ""))
    nums_p = {_norm_num(x) for x in re.findall(r"\d[\d,\.]*", pred)}
    nums_c = {_norm_num(x) for x in re.findall(r"\d[\d,\.]*", ctx)}
    bad = [x for x in nums_p if x and x not in nums_c and x not in ctx_norm]
    return float(len(bad) > 0)

def is_abstain(pred):
    return float(ABSTAIN in pred.lower())

def false_abstain(pred, answerable):
    return float(answerable and (ABSTAIN in pred.lower()))

def bootstrap_ci(vals, n=5000, seed=0):
    import numpy as np
    a = np.asarray(vals, float)
    if a.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    ms = rng.choice(a, (n, a.size), replace=True).mean(1)
    return (float(a.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)))


# ---------- resumable per-item store ----------
def partial_path(version):
    return os.path.join(PARTIAL, f"{version}.jsonl")

def load_done(version):
    """Return {(cond, pair_id): record} already computed (survives restarts)."""
    done = {}
    p = partial_path(version)
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[(r["cond"], r["pair_id"])] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return done

def append_item(fh, record):
    fh.write(json.dumps(record) + "\n")
    fh.flush()
    os.fsync(fh.fileno())          # force to disk so a power cut can't lose it

def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------- model ----------
def load_gemma(version, hf_token):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    d = os.path.join(CKPT, version)
    base_id = json.load(open(os.path.join(d, "adapter_config.json")))["base_model_name_or_path"]
    base = AutoModelForCausalLM.from_pretrained(
        base_id, attn_implementation="eager", torch_dtype=torch.bfloat16, token=hf_token).to("cuda")
    model = PeftModel.from_pretrained(base, d).eval()
    tok = AutoTokenizer.from_pretrained(d, token=hf_token)
    return model, tok


def run_version(version, rows_by_cond, hf_token, commit, use_cache):
    import torch
    final = os.path.join(OUT, f"{version}.json")
    if os.path.exists(final):
        print(f"[{version}] final result exists -> skip", flush=True)
        return
    is_raft = version.endswith("-raft")
    conds = ["clean", "realistic", "retrieval_failure", "closed_book"] if is_raft else ["clean"]

    done = load_done(version)
    total = sum(len(rows_by_cond.get(c, [])) for c in conds)
    print(f"[{version}] {len(done)}/{total} items already done (resuming)", flush=True)

    model, tok = load_gemma(version, hf_token)
    eos = tok.eos_token_id

    def gen(system, user):
        text = tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tok.pad_token_id or eos, eos_token_id=eos,
                                 use_cache=use_cache)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    os.makedirs(PARTIAL, exist_ok=True)
    t0, n_new = time.time(), 0
    with open(partial_path(version), "a", encoding="utf-8") as fh:
        for cond in conds:
            for r in rows_by_cond.get(cond, []):
                pid = r["meta"]["pair_id"]
                if (cond, pid) in done:
                    continue
                system, user = r["messages"][0]["content"], r["messages"][1]["content"]
                ref = r["messages"][2]["content"]
                answerable = ABSTAIN not in ref.lower()
                resp = gen(system, user)
                s = {"token_f1": token_f1(resp, ref), "exact_match": exact_match(resp, ref),
                     "fabrication": fabrication(resp, user), "abstain": is_abstain(resp),
                     "false_abstain": false_abstain(resp, answerable), "reward": None,
                     "resp_len_words": float(len(resp.split()))}
                rec = {"cond": cond, "pair_id": pid, "source": r["meta"].get("source"),
                       "answerable": answerable, "system": system, "user": user,
                       "ref": ref, "resp": resp, "scores": s}
                append_item(fh, rec)                       # <-- crash-safe checkpoint
                done[(cond, pid)] = rec
                n_new += 1
                if n_new % 25 == 0:
                    rate = (time.time() - t0) / n_new
                    remaining = (total - len(done)) * rate / 60
                    print(f"[{version}] {len(done)}/{total} ({rate:.1f}s/item, ~{remaining:.0f} min left)",
                          flush=True)

    # ---- finalize: aggregate CIs + manifest, write atomically ----
    per_item = [done[(c, r["meta"]["pair_id"])] for c in conds for r in rows_by_cond.get(c, [])
                if (c, r["meta"]["pair_id"]) in done]
    result = {"version": version, "family": "gemma", "is_raft": is_raft, "conditions": {}}
    for cond in conds:
        items = [it for it in per_item if it["cond"] == cond]
        g = lambda k: [it["scores"][k] for it in items]
        agg = {"n": len(items), "token_f1": bootstrap_ci(g("token_f1")),
               "exact_match": bootstrap_ci(g("exact_match")), "fabrication": bootstrap_ci(g("fabrication")),
               "abstain_rate": bootstrap_ci(g("abstain")), "false_abstain_rate": bootstrap_ci(g("false_abstain")),
               "reward": None}
        result["conditions"][cond] = agg
    if is_raft:
        c = result["conditions"]
        result["grounding_gap"] = c["realistic"]["token_f1"][0] - c["closed_book"]["token_f1"][0]
        result["distractor_gap"] = c["clean"]["token_f1"][0] - c["realistic"]["token_f1"][0]
        result["correct_abstention"] = c["retrieval_failure"]["abstain_rate"][0]

    import torch, transformers
    result["manifest"] = {
        "eval_sha256": hashlib.sha256(open(EVAL_PATH, "rb").read()).hexdigest(),
        "n_eval_items": sum(len(v) for v in rows_by_cond.values()),
        "n_per_condition": {c: len(rows_by_cond.get(c, [])) for c in conds},
        "decoding": {"preset": "plain", "max_new_tokens": 160, "greedy": True, "use_cache": use_cache},
        "few_shot_k": 0, "model_source": os.path.join(CKPT, version), "family": "gemma",
        "reward_model": None, "device": torch.cuda.get_device_name(0),
        "precision": "bfloat16", "run_location": "local-rtx3060",
        "libs": {"torch": torch.__version__, "transformers": transformers.__version__},
        "code_commit": commit, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    atomic_write_json(final, {"result": result, "per_item": per_item})
    cl = result["conditions"]["clean"]
    print(f"[{version}] DONE clean F1 {cl['token_f1'][0]:.3f} "
          f"[{cl['token_f1'][1]:.3f},{cl['token_f1'][2]:.3f}] -> {final}", flush=True)
    del model
    torch.cuda.empty_cache()


def git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="*", default=GEMMA_VERSIONS)
    ap.add_argument("--no-cache", action="store_true", help="disable KV cache (if gemma-2 errors)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN not found in .env — needed for the gated gemma-2-2b-it base.")

    rows = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8") if l.strip()]
    rows_by_cond = {}
    for r in rows:
        rows_by_cond.setdefault(r["meta"]["cond"], []).append(r)
    os.makedirs(OUT, exist_ok=True)
    commit = git_commit()
    print(f"local eval: {len(a.versions)} version(s); device check pending; commit={commit or 'n/a'}; "
          f"use_cache={not a.no_cache}", flush=True)
    for v in a.versions:
        try:
            run_version(v, rows_by_cond, hf_token, commit, use_cache=not a.no_cache)
        except Exception as e:
            import traceback
            print(f"[{v}] ERROR: {type(e).__name__}: {str(e)[:200]}", flush=True)
            traceback.print_exc()
            print(f"[{v}] partial progress is saved; re-run to resume.", flush=True)


if __name__ == "__main__":
    main()
