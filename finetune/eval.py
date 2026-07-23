"""Evaluate any fine-tuned/aligned version on the frozen eval set, on Modal (ace-compoz).

    modal run eval.py --version slm-500m-sft         # one version
    modal run eval.py --version base-500m             # base zero point
    modal run eval.py --all                           # every version + bases (sequential)

Scores the 'clean' grounded-QA condition (comparable across ALL versions) with token-F1,
exact-match, and fabrication rate; adds the RAFT four-condition (grounding gap, abstention)
for *-raft versions; and scores every response with the reward model for a quality/win-rate
signal. Every metric gets a bootstrap 95% CI. Saves /eval/<version>.json.
Aggregate into a comparison table locally with eval_report.py.
"""
from __future__ import annotations
import modal

app = modal.App("rl-eval")
vol = modal.Volume.from_name("ft-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate>=0.34",
                 "peft>=0.13", "bitsandbytes>=0.44", "numpy>=1.26,<2.0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("ft_config")
)

EVAL = "/data/rl/../sft/eval.jsonl"          # /data/sft/eval.jsonl
EVAL_PATH = "/data/sft/eval.jsonl"
CKPT = "/data/checkpoints"
RM = "/data/checkpoints/reward-500m"
ABSTAIN = "not stated in the context"

# base zero points
BASES = {"base-125m": "Ace-2504/slm-125m-base", "base-500m": "thesreedath/slm-500m-base",
         "base-gemma": "google/gemma-2-2b-it"}
ALL_VERSIONS = [
    "base-125m", "slm-125m-sft", "slm-125m-raft", "slm-125m-sft-dpo", "slm-125m-sft-rlaif",
    "base-500m", "slm-500m-sft", "slm-500m-raft", "slm-500m-sft-dpo", "slm-500m-sft-rlaif",
    "base-gemma", "gemma-2-2b-sft", "gemma-2-2b-raft", "gemma-2-2b-sft-dpo", "gemma-2-2b-sft-rlaif",
]

STOP = set("the a an of to in and or for on at by is are was were be with as that this it".split())


# ---------- metrics ----------
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

def fabrication(pred, ctx):
    import re
    nums_p = set(re.findall(r"\d[\d,\.]*", pred))
    nums_c = set(re.findall(r"\d[\d,\.]*", ctx))
    bad = [x for x in nums_p if x not in nums_c and x not in ctx]
    return float(len(bad) > 0)

def is_abstain(pred):
    return float(ABSTAIN in pred.lower())

def bootstrap_ci(vals, n=5000, seed=0):
    import numpy as np
    a = np.asarray(vals, float)
    if a.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    ms = rng.choice(a, (n, a.size), replace=True).mean(1)
    return (float(a.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)))


@app.function(image=image, gpu="L4", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60 * 2)
def evaluate(version: str):
    import json, os
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              AutoModelForSequenceClassification)
    dev = "cuda"; token = os.environ.get("HF_TOKEN")

    # ---- load the model (base / full / adapter) ----
    fam = "gemma" if "gemma" in version else "custom"
    if version in BASES:
        src = BASES[version]
        tok = AutoTokenizer.from_pretrained(src, token=token)
        if fam == "gemma":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(src, quantization_config=bnb,
                    attn_implementation="eager", torch_dtype=torch.bfloat16, token=token).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16,
                    token=token).to(dev).eval()
    else:
        d = f"{CKPT}/{version}"
        is_adapter = os.path.exists(f"{d}/adapter_config.json")
        if is_adapter:
            from transformers import BitsAndBytesConfig
            from peft import PeftModel
            base_id = json.load(open(f"{d}/adapter_config.json"))["base_model_name_or_path"]
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb,
                   attn_implementation="eager", torch_dtype=torch.bfloat16, token=token)
            model = PeftModel.from_pretrained(base, d).eval()
            tok = AutoTokenizer.from_pretrained(d, token=token)
        else:
            tok = AutoTokenizer.from_pretrained(d, token=token)
            model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to(dev).eval()
    if fam == "custom":
        eid = tok.convert_tokens_to_ids("<|eos|>")
        model.config.eos_token_id = eid
    eos = tok.eos_token_id if fam == "gemma" else tok.convert_tokens_to_ids("<|eos|>")
    gk = {} if fam == "custom" else {"use_cache": False}

    # ---- reward model (quality signal) ----
    rm_tok = AutoTokenizer.from_pretrained(RM)
    rm = AutoModelForSequenceClassification.from_pretrained(RM, num_labels=1,
         torch_dtype=torch.bfloat16).to(dev).eval()
    rm.config.pad_token_id = rm_tok.convert_tokens_to_ids("<|pad|>")

    def render(system, user):
        if fam == "gemma":
            return tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}],
                                           tokenize=False, add_generation_prompt=True)
        return f"<|system|>\n{system}<|eos|>\n<|user|>\n{user}<|eos|>\n<|assistant|>\n"

    def gen(system, user):
        text = render(system, user)
        ids = tok(text, return_tensors="pt", add_special_tokens=(fam == "gemma")).input_ids.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new_tokens=160, do_sample=False, no_repeat_ngram_size=3,
                                 repetition_penalty=1.2, pad_token_id=tok.pad_token_id or eos,
                                 eos_token_id=eos, **gk)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def reward(user, resp):
        t = rm_tok(user + "\n" + resp, return_tensors="pt", truncation=True, max_length=1024)
        t = {k: v.to(dev) for k, v in t.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            return rm(**t).logits.squeeze().item()

    rows = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8") if l.strip()]
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["meta"]["cond"], []).append(r)

    is_raft = version.endswith("-raft")
    conds = ["clean", "realistic", "retrieval_failure", "closed_book"] if is_raft else ["clean"]
    per_item = []
    result = {"version": version, "family": fam, "is_raft": is_raft, "conditions": {}}

    for cond in conds:
        f1s, ems, fabs, rews, abst = [], [], [], [], []
        for r in by_cond.get(cond, []):
            system, user = r["messages"][0]["content"], r["messages"][1]["content"]
            ref = r["messages"][2]["content"]
            resp = gen(system, user)
            f1s.append(token_f1(resp, ref)); ems.append(exact_match(resp, ref))
            fabs.append(fabrication(resp, user)); rews.append(reward(user, resp))
            abst.append(is_abstain(resp))
            per_item.append({"cond": cond, "pair_id": r["meta"]["pair_id"],
                             "resp": resp, "ref": ref, "reward": rews[-1]})
        result["conditions"][cond] = {
            "n": len(f1s),
            "token_f1": bootstrap_ci(f1s), "exact_match": bootstrap_ci(ems),
            "fabrication": bootstrap_ci(fabs), "reward": bootstrap_ci(rews),
            "abstain_rate": bootstrap_ci(abst)}

    if is_raft:
        c = result["conditions"]
        result["grounding_gap"] = c["realistic"]["token_f1"][0] - c["closed_book"]["token_f1"][0]
        result["distractor_gap"] = c["clean"]["token_f1"][0] - c["realistic"]["token_f1"][0]
        result["correct_abstention"] = c["retrieval_failure"]["abstain_rate"][0]

    os.makedirs("/data/eval", exist_ok=True)
    json.dump({"result": result, "per_item": per_item},
              open(f"/data/eval/{version}.json", "w"), indent=2)
    vol.commit()
    cl = result["conditions"]["clean"]
    print(f"[eval/{version}] clean F1 {cl['token_f1'][0]:.3f} "
          f"[{cl['token_f1'][1]:.3f},{cl['token_f1'][2]:.3f}] | reward {cl['reward'][0]:.3f} "
          f"| fabrication {cl['fabrication'][0]:.3f}", flush=True)
    return result


@app.local_entrypoint()
def main(version: str = "", all: bool = False):
    targets = ALL_VERSIONS if all else [version]
    for v in targets:
        try:
            evaluate.remote(v)
        except Exception as e:
            print(f"[eval/{v}] FAILED: {str(e)[:120]}")
