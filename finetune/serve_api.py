"""Local HTTP inference server for every SLM version — one endpoint, load-on-demand.

    ../.venv-cuda/Scripts/python.exe finetune/serve_api.py            # :8000

Holds at most MAX_RESIDENT models on the GPU (LRU-evicted), so a 12GB card can serve all 13
versions without ever loading them all at once. Model loading/generation logic mirrors
serve_local.py (adapter vs full model, custom <|role|> vs Gemma template, eos fix,
Gemma cache workaround).

    GET  /health          -> {ok, device, resident, catalog}
    GET  /models          -> catalog with `available` per model
    POST /generate        -> {model_id, prompt|question, context?, max_new_tokens?, temperature?...}
"""
from __future__ import annotations
import gc, os, threading, time
from collections import OrderedDict
from typing import Optional

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from serve_local import model_dir, build_prompt  # reuse the proven logic

CKPT = os.path.join(ROOT, "checkpoints")
MAX_RESIDENT = int(os.environ.get("MAX_RESIDENT", "2"))

# frontend model id -> (checkpoint dir or HF repo, demo kind)
#   kind: "completion" = base completer, "qa" = question answering, "grounded" = context+question
CATALOG = {
    "125m-base":   {"src": "Ace-2504/slm-125m-e4",                  "kind": "completion", "label": "SLM-125M (e4 base)"},
    "125m-qa":     {"src": os.path.join(CKPT, "slm-125m-sft"),      "kind": "qa",         "label": "SLM-125M · QA"},
    "125m-raft":   {"src": os.path.join(CKPT, "slm-125m-raft"),     "kind": "grounded",   "label": "SLM-125M · RAFT"},
    "125m-dpo":    {"src": os.path.join(CKPT, "slm-125m-sft-dpo"),  "kind": "qa",         "label": "SLM-125M · DPO"},
    "125m-rlaif":  {"src": os.path.join(CKPT, "slm-125m-sft-rlaif"),"kind": "qa",         "label": "SLM-125M · RLAIF"},
    "500m-base":   {"src": "thesreedath/slm-500m-base",             "kind": "completion", "label": "SLM-500M (base)"},
    "500m-dpo":    {"src": os.path.join(CKPT, "slm-500m-sft-dpo"),  "kind": "qa",         "label": "SLM-500M · DPO"},
    "500m-rlaif":  {"src": os.path.join(CKPT, "slm-500m-sft-rlaif"),"kind": "qa",         "label": "SLM-500M · RLAIF"},
    "gemma-base":  {"src": "google/gemma-2-2b-it",                  "kind": "qa",         "label": "Gemma 2 2B (base)"},
    "gemma-qa":    {"src": os.path.join(CKPT, "gemma-2-2b-sft"),    "kind": "qa",         "label": "Gemma 2 2B · QA"},
    "gemma-raft":  {"src": os.path.join(CKPT, "gemma-2-2b-raft"),   "kind": "grounded",   "label": "Gemma 2 2B · RAFT"},
    "gemma-dpo":   {"src": os.path.join(CKPT, "gemma-2-2b-sft-dpo"),"kind": "qa",         "label": "Gemma 2 2B · DPO"},
    "gemma-rlaif": {"src": os.path.join(CKPT, "gemma-2-2b-sft-rlaif"),"kind": "qa",       "label": "Gemma 2 2B · RLAIF"},
}

_lock = threading.Lock()
_resident: "OrderedDict[str, tuple]" = OrderedDict()   # id -> (model, tok, family)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _is_local(src: str) -> bool:
    return os.path.isabs(src) or os.path.exists(src)


def available(mid: str) -> bool:
    src = CATALOG[mid]["src"]
    if not _is_local(src):
        return True                      # HF repo — downloaded on first use
    d = model_dir(src)
    return os.path.exists(os.path.join(d, "config.json")) or \
           os.path.exists(os.path.join(d, "adapter_config.json"))


def _evict_one():
    mid, (model, tok, fam) = _resident.popitem(last=False)
    print(f"[evict] {mid}", flush=True)
    del model, tok
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def get_model(mid: str):
    """Load (or reuse) a model, evicting the least-recently-used when at capacity."""
    with _lock:
        if mid in _resident:
            _resident.move_to_end(mid)
            return _resident[mid]

        src = CATALOG[mid]["src"]
        if _is_local(src) and not available(mid):
            raise HTTPException(503, f"weights for '{mid}' are not downloaded yet ({src})")

        while len(_resident) >= MAX_RESIDENT:
            _evict_one()

        t0 = time.time()
        print(f"[load] {mid} <- {src}", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok_hf = os.environ.get("HF_TOKEN")
        d = model_dir(src) if _is_local(src) else src
        is_adapter = _is_local(src) and os.path.exists(os.path.join(d, "adapter_config.json"))

        if is_adapter:
            import json
            from peft import PeftModel
            base_id = json.load(open(os.path.join(d, "adapter_config.json")))["base_model_name_or_path"]
            tok = AutoTokenizer.from_pretrained(d, token=tok_hf)
            try:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16,
                                         bnb_4bit_use_double_quant=True)
                base = AutoModelForCausalLM.from_pretrained(
                    base_id, quantization_config=bnb, attn_implementation="eager",
                    torch_dtype=torch.bfloat16, token=tok_hf)
            except Exception as e:
                print(f"[warn] 4-bit unavailable ({type(e).__name__}); using fp16", flush=True)
                base = AutoModelForCausalLM.from_pretrained(
                    base_id, torch_dtype=torch.float16, attn_implementation="eager",
                    token=tok_hf).to(DEV)
            model = PeftModel.from_pretrained(base, d).eval()
            family = "gemma"
        else:
            tok = AutoTokenizer.from_pretrained(d, token=tok_hf)
            kw = {"attn_implementation": "eager"} if "gemma" in str(d).lower() else {}
            model = AutoModelForCausalLM.from_pretrained(
                d, torch_dtype=torch.bfloat16, token=tok_hf, **kw).to(DEV).eval()
            family = "gemma" if "gemma" in (getattr(model.config, "model_type", "") or "") else "custom"

        if family == "custom":
            eid = tok.convert_tokens_to_ids("<|eos|>")
            if eid is not None and eid >= 0:
                model.config.eos_token_id = eid
                model.generation_config.eos_token_id = eid

        print(f"[ready] {mid} ({family}) in {time.time()-t0:.1f}s", flush=True)
        _resident[mid] = (model, tok, family)
        return _resident[mid]


class GenReq(BaseModel):
    model_id: str
    prompt: Optional[str] = None       # completion-style (base models)
    question: Optional[str] = None     # qa / grounded
    context: Optional[str] = None
    max_new_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


app = FastAPI(title="SLM local inference")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": True, "device": DEV,
        "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None,
        "max_resident": MAX_RESIDENT,
        "resident": list(_resident.keys()),
        "models": len(CATALOG),
    }


@app.get("/models")
def models():
    return {mid: {"label": c["label"], "kind": c["kind"], "available": available(mid)}
            for mid, c in CATALOG.items()}


@app.post("/generate")
def generate(r: GenReq):
    if r.model_id not in CATALOG:
        raise HTTPException(404, f"unknown model '{r.model_id}'")
    kind = CATALOG[r.model_id]["kind"]
    model, tok, family = get_model(r.model_id)

    if kind == "completion":
        text = (r.prompt or r.question or "").strip()
        if not text:
            raise HTTPException(400, "prompt required")
        full = text
    else:
        q = (r.question or r.prompt or "").strip()
        if not q:
            raise HTTPException(400, "question required")
        full = build_prompt(tok, family, q, r.context)

    t0 = time.time()
    enc = tok(full, return_tensors="pt",
              add_special_tokens=(family == "gemma" and kind != "completion"))
    ids = enc.input_ids.to(DEV)
    attn = enc.attention_mask.to(DEV)      # explicit: pad and eos share an id on the custom models
    eos = tok.eos_token_id if family == "gemma" else tok.convert_tokens_to_ids("<|eos|>")
    gk = {"use_cache": False} if family == "gemma" else {}
    with torch.no_grad():
        out = model.generate(
            ids, attention_mask=attn,
            max_new_tokens=min(r.max_new_tokens, 512), do_sample=r.temperature > 0,
            temperature=max(r.temperature, 1e-5), top_p=r.top_p, top_k=r.top_k,
            no_repeat_ngram_size=3, repetition_penalty=1.2,
            pad_token_id=tok.pad_token_id or eos, eos_token_id=eos, **gk)
    completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return {
        "model_id": r.model_id, "kind": kind, "completion": completion,
        "tokens": int(out.shape[1] - ids.shape[1]),
        "seconds": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    import uvicorn
    print(f"device={DEV} | max_resident={MAX_RESIDENT} | models={len(CATALOG)}")
    for m in CATALOG:
        print(f"   {'OK ' if available(m) else 'MISSING'} {m}")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
