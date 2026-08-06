"""Generate eval answers for the two 500M fine-tunes that were never evaluated:
slm-500m-sft (arena id 500m-qa) and slm-500m-raft (arena id 500m-raft).

Faithful to evaluations/eval.py's CUSTOM-model path so the scores are directly comparable
to the other 11 non-gemma results:
  * same 500 CLEAN items (taken from an existing 500M judged file -> identical pair_ids,
    system, user, ref, source)
  * same render:  <|system|>\n{system}<|eos|>\n<|user|>\n{user}<|eos|>\n<|assistant|>\n
  * PURE GREEDY decoding, max_new_tokens=160, add_special_tokens=False, eos=<|eos|>
  * same token_f1 / fabrication functions

Output matches eval.py's schema (per_item with cond/pair_id/source/answerable/system/user/
ref/resp/scores) so rejudge_rubric10.py consumes it unchanged.

Resumable: each item is appended+fsync'd to eval_results/_partial/<model>.jsonl.

    python evaluations/gen_500m_new.py
"""
from __future__ import annotations
import json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "finetune"))
from serve_local import load  # the proven custom/gemma loader (handles <|eos|> fix)

ER = os.path.join(ROOT, "eval_results")
PARTIAL = os.path.join(ER, "_partial")
CKPT = os.path.join(ROOT, "checkpoints")
TEMPLATE = os.path.join(ER, "slm-500m-sft-dpo.judged.json")  # source of the 500 clean items
SYS_FALLBACK = "You are a precise legal and financial assistant. Answer clearly using the provided context; do not invent facts."
ABSTAIN = "not stated in the context"

# arena id -> checkpoint dir
JOBS = [
    ("slm-500m-sft",  os.path.join(CKPT, "slm-500m-sft")),
    ("slm-500m-raft", os.path.join(CKPT, "slm-500m-raft")),
]


# ---- metrics: identical to eval.py ----
def _words(s):
    return "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower()).split()

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


def render(system, user):
    # custom scheme, NO bos — identical to eval.py render() for fam=custom, k_shot=0
    return f"<|system|>\n{system}<|eos|>\n<|user|>\n{user}<|eos|>\n<|assistant|>\n"


def load_done(model_name):
    p = os.path.join(PARTIAL, f"{model_name}.jsonl")
    done = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line); done[r["pair_id"]] = r
            except Exception:
                continue
    return done


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import torch
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    os.makedirs(PARTIAL, exist_ok=True)

    tmpl = json.load(open(TEMPLATE, encoding="utf-8"))
    items = [it for it in tmpl["per_item"] if it["cond"] == "clean"]
    print(f"template: {len(items)} clean items from {os.path.basename(TEMPLATE)}", flush=True)

    for model_name, ckpt in JOBS:
        final = os.path.join(ER, f"{model_name}.judged.json")
        if os.path.exists(final):
            print(f"[{model_name}] final exists -> skip", flush=True)
            continue
        done = load_done(model_name)
        print(f"[{model_name}] loading {ckpt}  ({len(done)}/{len(items)} already done)", flush=True)
        model, tok, family, dev = load(ckpt)
        eos = tok.convert_tokens_to_ids("<|eos|>") if family == "custom" else tok.eos_token_id

        def gen(system, user):
            text = render(system, user)
            ids = tok(text, return_tensors="pt", add_special_tokens=(family == "gemma")).input_ids.to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(ids, max_new_tokens=160, do_sample=False,
                                     pad_token_id=tok.pad_token_id or eos, eos_token_id=eos)
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

        pth = os.path.join(PARTIAL, f"{model_name}.jsonl")
        t0, n_new = time.time(), 0
        with open(pth, "a", encoding="utf-8") as fh:
            for it in items:
                pid = it["pair_id"]
                if pid in done:
                    continue
                system = it.get("system") or SYS_FALLBACK
                user, ref = it["user"], it["ref"]
                answerable = ABSTAIN not in ref.lower()
                resp = gen(system, user)
                s = {"token_f1": token_f1(resp, ref), "exact_match": exact_match(resp, ref),
                     "fabrication": fabrication(resp, user), "abstain": is_abstain(resp),
                     "false_abstain": false_abstain(resp, answerable), "reward": None,
                     "resp_len_words": float(len(resp.split()))}
                rec = {"cond": "clean", "pair_id": pid, "source": it.get("source"),
                       "answerable": answerable, "system": system, "user": user,
                       "ref": ref, "resp": resp, "scores": s}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush(); os.fsync(fh.fileno())
                done[pid] = rec
                n_new += 1
                if n_new % 25 == 0:
                    rate = (time.time() - t0) / n_new
                    left = (len(items) - len(done)) * rate / 60
                    print(f"[{model_name}] {len(done)}/{len(items)}  {rate:.2f}s/item  ~{left:.0f} min left", flush=True)

        per_item = [done[it["pair_id"]] for it in items if it["pair_id"] in done]
        f1 = sum(x["scores"]["token_f1"] for x in per_item) / len(per_item)
        fab = sum(x["scores"]["fabrication"] for x in per_item) / len(per_item)
        result = {"version": model_name, "family": "custom", "is_raft": model_name.endswith("-raft"),
                  "conditions": {"clean": {"n": len(per_item),
                                           "token_f1_mean": round(f1, 4),
                                           "fabrication_mean": round(fab, 4)}}}
        json.dump({"result": result, "per_item": per_item},
                  open(final, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{model_name}] DONE n={len(per_item)} tokenF1={f1:.3f} fabrication={fab*100:.1f}% -> {final}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("ALL GENERATION DONE", flush=True)


if __name__ == "__main__":
    main()
