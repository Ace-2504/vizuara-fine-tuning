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

import sys
sys.path.insert(0, ROOT)                         # repo root -> teacher.py (Gemini client)

# --- live judging -------------------------------------------------------------------
# Same rubric as the offline harness (evaluations/judge_eval.py): model-blind, pointwise,
# 1-5 correctness. Two variants: with a gold REFERENCE for the frozen eval questions
# (checkable), and reference-free for user-written questions (the judge grades against its
# own knowledge — weaker, and labelled as such in the UI).
JUDGE_SCHEMA = {"type": "object", "properties": {
    "correct": {"type": "integer"},
    "grounded": {"type": "boolean"},
    "reason": {"type": "string"}},
    "required": ["correct", "grounded"]}

# Pinned on purpose. TeacherClient's default list starts at `gemini-3.1-flash`, which this key
# does not serve for generateContent — every run then burned a 404 to discover the fallback
# (bugs/23). Flash-Lite is the judge the offline harness actually used, so scores stay comparable.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-flash-lite")

_teacher = None
_teacher_lock = threading.Lock()


def _get_teacher():
    global _teacher
    with _teacher_lock:
        if _teacher is None:
            from teacher import TeacherClient
            _teacher = TeacherClient(models=(JUDGE_MODEL,))
        return _teacher


def _judge_prompt(question: str, context: Optional[str], reference: Optional[str],
                  candidate: str) -> str:
    head = ("You are a strict evaluator of a question-answering system. Judge ONLY the "
            "CANDIDATE answer; you do not know which system produced it.\n")
    if reference:
        rule = ("The REFERENCE is a correct short answer. Grade the CANDIDATE against it. "
                "Judge meaning, not wording — a correct paraphrase scores high.\n")
        ref_block = f"\nREFERENCE:\n{reference}\n"
    else:
        rule = ("No reference answer is available. Grade the CANDIDATE on factual accuracy and "
                "whether it actually answers the question, using your own knowledge. Be strict: "
                "vague, evasive, repetitive or fabricated answers score low.\n")
        ref_block = ""
    ctx_block = f"\nCONTEXT:\n{context}\n" if context else ""
    return (head + rule +
            '\nReturn JSON:\n- "correct" (1-5): 5 = fully correct and complete, 1 = wrong or '
            'irrelevant.\n- "grounded" (bool): true if the answer avoids fabricated claims'
            + (" and is supported by the CONTEXT.\n" if context else ".\n") +
            '- "reason": one short sentence.\n'
            + ctx_block + f"\nQUESTION:\n{question}\n" + ref_block +
            f"\nCANDIDATE:\n{candidate}")

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

# --- shared Gemma backbone ----------------------------------------------------------
# All five Gemma entries are the SAME 2.6B weights: the base plus four LoRA adapters.
# Loading a private copy per variant cost ~15 GB; one 4-bit base with adapters attached
# costs ~2.4 GB. `gemma-base` is served from the same object with adapters disabled.
# PEFT injects adapter layers INTO the base modules, so entries never cache the model
# object — they resolve `_shared_gemma` at generation time (it gets re-wrapped on first
# adapter attach) and switch adapters under the _gpu lock, which serialises generation.
GEMMA_BASE_ID = "google/gemma-2-2b-it"
_shared_gemma = None            # AutoModel, later wrapped in PeftModel
_gemma_tok = None


def _is_gemma(mid: str) -> bool:
    return mid.startswith("gemma-")


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
    if _is_gemma(mid):
        # once the shared backbone is up, another Gemma variant is just a ~40 MB adapter
        return 0.15 if _shared_gemma is not None else 2.4
    src = CATALOG[mid]["src"]
    if not _is_local(src):                       # HF repo, not on disk yet
        return 1.2 if "500m" in mid else 0.4
    d = model_dir(src)
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
            _drop_shared_gemma_if_unused()      # backbone lives only while a Gemma is resident
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


def _load_gemma(mid: str):
    """Build (or reuse) the one shared 4-bit Gemma backbone and attach this variant's adapter."""
    global _shared_gemma, _gemma_tok
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok_hf = os.environ.get("HF_TOKEN")
    t0 = time.time()

    if _shared_gemma is None:
        print(f"[load] shared gemma backbone <- {GEMMA_BASE_ID} (4-bit)", flush=True)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        _shared_gemma = AutoModelForCausalLM.from_pretrained(
            GEMMA_BASE_ID, quantization_config=bnb, attn_implementation="eager",
            torch_dtype=torch.bfloat16, token=tok_hf).eval()
        _gemma_tok = AutoTokenizer.from_pretrained(GEMMA_BASE_ID, token=tok_hf)
        print(f"[ready] shared gemma backbone in {time.time()-t0:.1f}s  vram {_vram():.2f} GB",
              flush=True)

    adapter = None
    src = CATALOG[mid]["src"]
    if _is_local(src):                                   # a fine-tune -> attach its adapter
        from peft import PeftModel
        d = model_dir(src)
        adapter = mid
        if not isinstance(_shared_gemma, PeftModel):
            _shared_gemma = PeftModel.from_pretrained(_shared_gemma, d, adapter_name=adapter).eval()
        elif adapter not in getattr(_shared_gemma, "peft_config", {}):
            _shared_gemma.load_adapter(d, adapter_name=adapter)
        print(f"[ready] {mid} adapter attached in {time.time()-t0:.1f}s  vram {_vram():.2f} GB",
              flush=True)

    return {"model": None, "shared_gemma": True, "adapter": adapter,
            "tok": _gemma_tok, "family": "gemma"}


def _drop_shared_gemma_if_unused():
    """Free the 4-bit backbone once no Gemma variant is resident. Call holding _cv."""
    global _shared_gemma, _gemma_tok
    if _shared_gemma is None or any(_is_gemma(m) for m in _resident):
        return
    before = _vram()
    _shared_gemma = None
    _gemma_tok = None
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    print(f"[evict] shared gemma backbone  vram {before:.2f} -> {_vram():.2f} GB", flush=True)


def _load_once(mid: str):
        if _is_gemma(mid):
            return _load_gemma(mid)
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
        # _checkout owns registration into _resident
        return {"model": model, "shared_gemma": False, "adapter": None, "tok": tok, "family": family}


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
    with acquire(r.model_id) as entry:
        tok, family = entry["tok"], entry["family"]
        model = _shared_gemma if entry["shared_gemma"] else entry["model"]
        if model is None:
            raise HTTPException(503, f"'{r.model_id}' is no longer loaded; retry")
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
            # Adapter selection mutates shared state, so it happens under the same lock.
            sel = contextlib.nullcontext()
            if entry["shared_gemma"]:
                has_adapters = hasattr(model, "peft_config")
                if entry["adapter"]:
                    model.set_adapter(entry["adapter"])
                elif has_adapters:
                    sel = model.disable_adapter()      # gemma-base off the same backbone
            try:
                with sel, torch.no_grad():
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


class JudgeReq(BaseModel):
    question: str
    context: Optional[str] = None
    reference: Optional[str] = None       # gold answer, when the question came from the eval set
    answers: dict                         # model_id -> candidate answer


@app.post("/judge")
def judge(r: JudgeReq):
    """Score every candidate answer with the blind LLM judge. Answers are judged in parallel
    (network-bound), and each is scored independently — the judge never sees the model name."""
    from concurrent.futures import ThreadPoolExecutor

    teacher = _get_teacher()      # resolve the model (and any fallback) once, before fanning out

    def one(item):
        mid, cand = item
        cand = (cand or "").strip()
        if not cand:
            return mid, {"score": 0.0, "correct": 1, "grounded": False, "reason": "empty answer"}
        prompt = _judge_prompt(r.question, r.context, r.reference, cand)
        try:
            out = teacher.generate_json(prompt, JUDGE_SCHEMA, temperature=0.0)
            if not out:                                   # transient empty/unparseable -> one retry
                out = teacher.generate_json(prompt, JUDGE_SCHEMA, temperature=0.0)
            if not out:
                return mid, {"error": "judge returned nothing"}
            c = max(1, min(5, int(out.get("correct", 1))))
            return mid, {"score": round((c - 1) / 4 * 10, 1), "correct": c,
                         "grounded": bool(out.get("grounded", False)),
                         "reason": (out.get("reason") or "")[:300]}
        except Exception as e:                      # one bad call must not sink the whole run
            return mid, {"error": f"{type(e).__name__}: {str(e)[:160]}"}

    # The first call is made SERIALLY on purpose: TeacherClient only discovers a model fallback
    # (e.g. gemini-3.1-flash -> -lite) when a request 404s, so fanning out first would send every
    # parallel call to a stale model and only one would survive.
    items = list(r.answers.items())
    scored: dict = {}
    if items:
        mid, res = one(items[0])
        scored[mid] = res
        if len(items) > 1:
            with ThreadPoolExecutor(max_workers=6) as ex:
                scored.update(dict(ex.map(one, items[1:])))
    return {"graded": scored, "referenced": bool(r.reference),
            "judge_model": getattr(teacher, "model", "gemini")}


if __name__ == "__main__":
    import uvicorn
    print(f"device={DEV} | max_resident={MAX_RESIDENT} | models={len(CATALOG)}")
    for m in CATALOG:
        print(f"   {'OK ' if available(m) else 'MISSING'} {m}")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
