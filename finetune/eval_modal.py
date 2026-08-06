"""Generate a seeded checkpoint's eval answers ON MODAL (A100) — no local GPU.

    modal run eval_modal.py --seeded gemma-2-2b-sft-seed1 --family gemma
    modal run eval_modal.py --seeded slm-500m-sft-seed2 --family custom

Faithful to eval.py's clean-condition generation (greedy, family render, max_new 160). Reads the
500 clean items from /data/sft/eval.jsonl, loads the checkpoint from /data/checkpoints/<seeded>,
writes per-item generations + token_f1/fabrication to /data/eval_out/<seeded>.json on the volume.
Judging (Gemini) is done locally afterwards — this step is GPU-only and power-cut-proof in the cloud.
"""
from __future__ import annotations
import modal
import ft_config as C

app = modal.App("eval-seed")
vol = modal.Volume.from_name("ft-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate>=0.34",
                 "peft>=0.13", "bitsandbytes>=0.44", "numpy>=1.26,<2.0")
    .env({"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("ft_config")
)
ABSTAIN = "not stated in the context"
GEMMA_BASE = "google/gemma-2-2b-it"


@app.function(image=image, gpu="A100-40GB", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60 * 3)
def evaluate(seeded: str, family: str, limit: int = 0, batch: int = 0):
    import json, os, re, time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def words(s): return "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower()).split()
    def token_f1(pred, ref):
        p, r = words(pred), words(ref)
        if not p or not r: return 0.0
        cc = {}
        for w in p: cc[w] = cc.get(w, 0) + 1
        ov = sum(min(c, r.count(w)) for w, c in cc.items())
        if not ov: return 0.0
        pr, rc = ov/len(p), ov/len(r); return 2*pr*rc/(pr+rc)
    def nn(s): return s.replace(",", "").rstrip("%").rstrip(".")
    def fabrication(pred, ctx):
        cnorm = nn(ctx.replace(",", ""))
        npd = {nn(x) for x in re.findall(r"\d[\d,\.]*", pred)}
        ncx = {nn(x) for x in re.findall(r"\d[\d,\.]*", ctx)}
        return float(len([x for x in npd if x and x not in ncx and x not in cnorm]) > 0)

    token = os.environ.get("HF_TOKEN")
    ckpt = f"/data/checkpoints/{seeded}"
    # find the actual dir (download/nesting safety)
    import glob
    if not (os.path.exists(f"{ckpt}/config.json") or os.path.exists(f"{ckpt}/adapter_config.json")):
        for d in glob.glob(f"{ckpt}/*"):
            if os.path.exists(f"{d}/config.json") or os.path.exists(f"{d}/adapter_config.json"):
                ckpt = d; break

    if family == "gemma":                       # adapter on the 4-bit base
        from transformers import BitsAndBytesConfig
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(ckpt, token=token)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(GEMMA_BASE, quantization_config=bnb,
               attn_implementation="eager", torch_dtype=torch.bfloat16, token=token)
        model = PeftModel.from_pretrained(base, ckpt).eval()
        eos = tok.eos_token_id
    else:                                        # custom full model (125m/500m)
        tok = AutoTokenizer.from_pretrained(ckpt, token=token)
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, token=token).to("cuda").eval()
        eos = tok.convert_tokens_to_ids("<|eos|>")

    rows = [json.loads(l) for l in open("/data/sft/eval.jsonl", encoding="utf-8") if l.strip()]
    items = [r for r in rows if r["meta"]["cond"] == "clean"]
    if limit:
        items = items[:limit]
    BATCH = batch or (8 if family == "gemma" else 32)
    print(f"[{seeded}] {len(items)} clean items | family={family} | batch={BATCH}", flush=True)

    # left-pad for decoder-only batched greedy generation (batch-invariant given attention_mask)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if family == "gemma" else "<|pad|>"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos
    add_special = (family == "gemma")
    gk = {"use_cache": False} if family == "gemma" else {}

    def render_one(system, user):
        if family == "gemma":
            return tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}],
                                           tokenize=False, add_generation_prompt=True)
        return f"<|system|>\n{system}<|eos|>\n<|user|>\n{user}<|eos|>\n<|assistant|>\n"

    def gen_batch(texts):
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=add_special)
        # some custom tokenizers emit token_type_ids, which generate() rejects — keep only these two
        enc = {k: v.to("cuda") for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(**enc, max_new_tokens=160, do_sample=False,
                                 pad_token_id=pad_id, eos_token_id=eos, **gk)
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip() for g in gen_ids]

    per_item, t0 = [], time.time()
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        metas = []
        for r in chunk:
            m = r["messages"]
            metas.append((r, m[0]["content"], m[1]["content"], m[2]["content"]))
        resps = gen_batch([render_one(s, u) for _, s, u, _ in metas])
        for (r, system, user, ref), resp in zip(metas, resps):
            per_item.append({"cond": "clean", "pair_id": r["meta"]["pair_id"], "source": r["meta"].get("source"),
                             "system": system, "user": user, "ref": ref, "resp": resp,
                             "scores": {"token_f1": token_f1(resp, ref), "fabrication": fabrication(resp, user)}})
        done = len(per_item)
        print(f"  {done}/{len(items)} ({(time.time()-t0)/done:.2f}s/item)", flush=True)

    os.makedirs("/data/eval_out", exist_ok=True)
    out = f"/data/eval_out/{seeded}{'-limit'+str(limit) if limit else ''}.json"
    json.dump({"seeded": seeded, "family": family, "per_item": per_item},
              open(out, "w", encoding="utf-8"), ensure_ascii=False)
    vol.commit()
    f1 = sum(x["scores"]["token_f1"] for x in per_item)/len(per_item)
    fab = sum(x["scores"]["fabrication"] for x in per_item)/len(per_item)
    print(f"[{seeded}] DONE n={len(per_item)} tokenF1={f1:.3f} fab={fab*100:.1f}% wall={(time.time()-t0)/60:.1f}min -> {out}", flush=True)
    return {"seeded": seeded, "n": len(per_item), "token_f1": round(f1, 3), "fabrication": round(fab*100, 1), "out": out}


@app.local_entrypoint()
def main(seeded: str, family: str = "custom", limit: int = 0, batch: int = 0):
    print(evaluate.remote(seeded, family, limit, batch))
