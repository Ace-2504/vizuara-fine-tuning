"""Run any fine-tuned/aligned version locally on the RTX 3060.

    python serve_local.py --ckpt ../checkpoints/slm-500m-sft --question "..." [--context "..."]
    python serve_local.py --ckpt ../checkpoints/gemma-2-2b-sft-dpo --question "..."   # adapter
    python serve_local.py --ckpt ../checkpoints/slm-125m-sft-rlaif --chat             # interactive

Auto-detects: full-model vs LoRA-adapter, and model family (custom <|role|> vs Gemma template).
Handles the no-BOS / eos-fix for the custom models and the chat template for Gemma. For gated
Gemma adapters you need HF_TOKEN in ../.env (base is downloaded once).
"""
from __future__ import annotations
import argparse, glob, json, os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")   # ignore stale cached token
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

SYS = "You are a precise legal and financial assistant. Answer clearly using the provided context; do not invent facts."


def model_dir(path):
    """Return the dir that actually holds config/adapter files (handles download double-nesting)."""
    for d in (path, *glob.glob(os.path.join(path, "*"))):
        if os.path.isdir(d) and (os.path.exists(os.path.join(d, "config.json"))
                                 or os.path.exists(os.path.join(d, "adapter_config.json"))):
            return d
    return path


def load(ckpt):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = model_dir(ckpt)
    tok_hf = os.environ.get("HF_TOKEN")
    is_adapter = os.path.exists(os.path.join(d, "adapter_config.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if is_adapter:                                            # Gemma LoRA -> base + adapter
        from peft import PeftModel
        base_id = json.load(open(os.path.join(d, "adapter_config.json")))["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(d, token=tok_hf)
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb,
                   attn_implementation="eager", torch_dtype=torch.bfloat16, token=tok_hf)
        except Exception:                                    # no bitsandbytes -> fp16
            base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16,
                   attn_implementation="eager", token=tok_hf).to(dev)
        model = PeftModel.from_pretrained(base, d).eval()
        family = "gemma"
    else:                                                    # full model (125M/500M or merged)
        tok = AutoTokenizer.from_pretrained(d, token=tok_hf)
        model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16,
                                                     token=tok_hf).to(dev).eval()
        family = "gemma" if "gemma" in (getattr(model.config, "model_type", "") or "") else "custom"

    if family == "custom":                                   # eos points at wrong id in some configs
        eid = tok.convert_tokens_to_ids("<|eos|>")
        model.config.eos_token_id = eid
        model.generation_config.eos_token_id = eid
    return model, tok, family, dev


def build_prompt(tok, family, question, context):
    user = f"Context:\n{context}\n\nQuestion: {question}" if context else question
    if family == "gemma":
        conv = [{"role": "user", "content": f"{SYS}\n\n{user}"}]   # Gemma has no system role
        return tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    # custom scheme, NO bos
    return f"<|system|>\n{SYS}<|eos|>\n<|user|>\n{user}<|eos|>\n<|assistant|>\n"


def generate(model, tok, family, dev, question, context=None, max_new=160):
    import torch
    text = build_prompt(tok, family, question, context)
    ids = tok(text, return_tensors="pt", add_special_tokens=(family == "gemma")).input_ids.to(dev)
    eos = tok.eos_token_id if family == "gemma" else tok.convert_tokens_to_ids("<|eos|>")
    gk = {} if family == "custom" else {"use_cache": False}   # Gemma cache dtype bug workaround
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                             top_p=0.9, no_repeat_ngram_size=3, repetition_penalty=1.2,
                             pad_token_id=tok.pad_token_id or eos, eos_token_id=eos, **gk)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--question", default=None)
    ap.add_argument("--context", default=None)
    ap.add_argument("--chat", action="store_true", help="interactive loop")
    ap.add_argument("--max_new", type=int, default=160)
    args = ap.parse_args()

    print(f"loading {args.ckpt} ...")
    model, tok, family, dev = load(args.ckpt)
    print(f"loaded ({family}, {dev}).")
    if args.chat:
        print("Interactive. Blank line to quit. Optional 'CONTEXT: ...' line before your question.")
        ctx = None
        while True:
            q = input("\nyou> ").strip()
            if not q:
                break
            if q.upper().startswith("CONTEXT:"):
                ctx = q[8:].strip(); print("(context set)"); continue
            print("model>", generate(model, tok, family, dev, q, ctx, args.max_new))
    else:
        assert args.question, "pass --question or --chat"
        print("\n" + generate(model, tok, family, dev, args.question, args.context, args.max_new))


if __name__ == "__main__":
    main()
