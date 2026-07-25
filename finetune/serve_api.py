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
import contextlib, gc, os, threading, time
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
MAX_RESIDENT = int(os.environ.get("MAX_RESIDENT", "13"))   # 13 = pin every model, no eviction

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

# --- concurrency model -------------------------------------------------------------
# _cv guards _resident and _inuse. A model is only evictable while _inuse[mid] == 0, so a
# request that is mid-generation can never have its weights pulled out from under it.
# Generation itself is serialised by _gpu: there is one GPU, so overlapping generations
# would only interleave and double peak VRAM.
_cv = threading.Condition()
_resident: "OrderedDict[str, tuple]" = OrderedDict()   # id -> (model, tok, family), LRU order
_inuse: dict = {}                                      # id -> in-flight request count
_gpu = threading.Semaphore(1)
BUSY_WAIT_S = float(os.environ.get("BUSY_WAIT_S", "180"))   # wait for a free slot before 503
DEV = "cuda" if torch.cuda.is_available() else "cpu"


HEADROOM_GB = float(os.environ.get("HEADROOM_GB", "1.5"))   # left free for activations / KV cache


def _vram() -> float:
    return torch.cuda.memory_allocated() / 2**30 if DEV == "cuda" else 0.0


def _free_vram() -> float:
    """Free VRAM as the driver sees it — includes memory held by other processes."""
    if DEV != "cuda":
        return 1e9
    free, _total = torch.cuda.mem_get_info()
    return free / 2**30


def _estimate_gb(mid: str) -> float:
    """Rough VRAM a model will need. Loading 4-bit weights can abort the PROCESS rather than
    raise, so admission is decided BEFORE loading — a reactive except-OOM cannot save us."""
    src = CATALOG[mid]["src"]
    if not _is_local(src):                       # HF repo, not on disk yet
        return 5.6 if "gemma" in mid else 1.2 if "500m" in mid else 0.4
    d = model_dir(src)
    if os.path.exists(os.path.join(d, "adapter_config.json")):
        return 2.4                               # 4-bit Gemma base + LoRA adapter
    try:
        b = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                if f.endswith((".safetensors", ".bin")))
        return max(b / 2**30 * 1.05, 0.2)
    except OSError:
        return 1.0


def _is_local(src: str) -> bool:
    return os.path.isabs(src) or os.path.exists(src)


def available(mid: str) -> bool:
    src = CATALOG[mid]["src"]
    if not _is_local(src):
        return True                      # HF repo — downloaded on first use
    d = model_dir(src)
    return os.path.exists(os.path.join(d, "config.json")) or \
           os.path.exists(os.path.join(d, "adapter_config.json"))


def _evict_idle_one() -> bool:
    """Evict the least-recently-used model that has NO in-flight request. Call holding _cv.

    Returns False when every resident model is busy — the caller then waits rather than
    loading anyway, because an 'eviction' of an in-use model frees no VRAM (the serving
    thread still holds a reference) and would silently over-commit the GPU.
    """
    for mid in list(_resident.keys()):                 # LRU order
        if _inuse.get(mid, 0) == 0:
            before = _vram()
            entry = _resident.pop(mid)
            _inuse.pop(mid, None)
            del entry
            gc.collect()
            if DEV == "cuda":
                torch.cuda.empty_cache()
            print(f"[evict] {mid}  vram {before:.2f} -> {_vram():.2f} GB", flush=True)
            return True
    return False


@contextlib.contextmanager
def acquire(mid: str):
    """Check a model out for the duration of one request; it cannot be evicted meanwhile."""
    entry = _checkout(mid)
    try:
        yield entry
    finally:
        with _cv:
            _inuse[mid] = max(0, _inuse.get(mid, 0) - 1)
            _cv.notify_all()


def _checkout(mid: str):
    deadline = time.time() + BUSY_WAIT_S
    with _cv:
        while True:
            if mid in _resident:
                _resident.move_to_end(mid)
                _inuse[mid] = _inuse.get(mid, 0) + 1
                return _resident[mid]

            src = CATALOG[mid]["src"]
            if _is_local(src) and not available(mid):
                raise HTTPException(503, f"weights for '{mid}' are not downloaded yet ({src})")

            # Admission control: make room by COUNT and by actual free VRAM before loading.
            need = _estimate_gb(mid) + HEADROOM_GB
            if len(_resident) >= MAX_RESIDENT or _free_vram() < need:
                if _evict_idle_one():
                    continue                            # freed something; re-evaluate
                left = deadline - time.time()
                if left <= 0:
                    if _free_vram() < need and _resident:
                        raise HTTPException(
                            503, f"not enough GPU memory for '{mid}': needs ~{need:.1f} GB, "
                                 f"{_free_vram():.1f} GB free, and all resident models are busy")
                    raise HTTPException(503, f"all {len(_resident)} resident models are busy")
                _cv.wait(timeout=min(left, 5.0))
                continue                                # re-check: it may now be loaded/free

            entry = _load(mid)
            _resident[mid] = entry
            _inuse[mid] = 1
            return entry


def _load(mid: str):
    """Load with an out-of-memory safety net. Call holding _cv."""
    try:
        return _load_once(mid)
    except torch.cuda.OutOfMemoryError:
        print(f"[oom] loading {mid} — dropping idle models and retrying", flush=True)
        while _evict_idle_one():
            pass
        try:
            return _load_once(mid)
        except torch.cuda.OutOfMemoryError as e:
            raise HTTPException(503, f"out of GPU memory loading '{mid}'") from e


def _load_once(mid: str):
        src = CATALOG[mid]["src"]
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

        print(f"[ready] {mid} ({family}) in {time.time()-t0:.1f}s  vram {_vram():.2f} GB", flush=True)
        return (model, tok, family)          # _checkout owns registration into _resident


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
        "in_use": {k: v for k, v in _inuse.items() if v},
        "vram_allocated_gb": round(_vram(), 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2) if DEV == "cuda" else 0,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if DEV == "cuda" else 0,
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

    # The model stays checked out (un-evictable) for this whole block.
    with acquire(r.model_id) as (model, tok, family):
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
        attn = enc.attention_mask.to(DEV)  # explicit: pad and eos share an id on the custom models
        eos = tok.eos_token_id if family == "gemma" else tok.convert_tokens_to_ids("<|eos|>")
        gk = {"use_cache": False} if family == "gemma" else {}
        with _gpu:                          # one generation at a time on the single GPU
            try:
                with torch.no_grad():
                    out = model.generate(
                        ids, attention_mask=attn,
                        max_new_tokens=min(r.max_new_tokens, 512), do_sample=r.temperature > 0,
                        temperature=max(r.temperature, 1e-5), top_p=r.top_p, top_k=r.top_k,
                        no_repeat_ngram_size=3, repetition_penalty=1.2,
                        pad_token_id=tok.pad_token_id or eos, eos_token_id=eos, **gk)
            except torch.cuda.OutOfMemoryError as e:
                if DEV == "cuda":
                    torch.cuda.empty_cache()
                raise HTTPException(503, f"out of GPU memory generating with '{r.model_id}'") from e
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
