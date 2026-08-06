"""DPO on the QA-SFT models, on Modal. Hand-rolled DPO loss (no TRL version risk).

    modal run train_dpo.py --model 125m      # slm-125m-sft   -> slm-125m-sft-dpo   (full FT)
    modal run train_dpo.py --model 500m     # slm-500m-sft   -> slm-500m-sft-dpo   (full FT)
    modal run train_dpo.py --model gemma     # gemma-2-2b-sft -> gemma-2-2b-sft-dpo (QLoRA)

Policy = the SFT model (trainable); reference = a frozen copy of the SFT model. DPO raises the
log-prob margin of `chosen` over `rejected`, regularized toward the reference by `beta`.
Reads /rl/preferences.jsonl and /checkpoints/<name>-sft from the ft-data volume.
"""
from __future__ import annotations
import modal

import ft_config as C

app = modal.App("rl-dpo")
vol = modal.Volume.from_name("ft-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate>=0.34",
                 "peft>=0.13", "bitsandbytes>=0.44", "numpy>=1.26,<2.0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("ft_config", "ft_data")
)

PREFS = "/data/rl/preferences.jsonl"
BETA = 0.1
EPOCHS = 3
LR_FULL, LR_LORA = 5e-6, 1e-4

CFG = {
    "125m":  {"gpu": "L4", "mode": "full",  "render": "custom", "max_seq": 1024,
              "name": "slm-125m",
              "ckpt": "/data/checkpoints/slm-125m-sft", "out": "/data/checkpoints/slm-125m-sft-dpo",
              "micro": 4},                                    # 125M is tiny -> larger micro fits
    "500m":  {"gpu": "L4", "mode": "full",  "render": "custom", "max_seq": 1024,
              "name": "slm-500m",
              "ckpt": "/data/checkpoints/slm-500m-sft", "out": "/data/checkpoints/slm-500m-sft-dpo",
              "micro": 2},
    "gemma": {"gpu": "A100-40GB", "mode": "qlora", "render": "gemma", "max_seq": 2048,
              "name": "gemma-2-2b",
              "base": "google/gemma-2-2b-it", "adapter": "/data/checkpoints/gemma-2-2b-sft",
              "out": "/data/checkpoints/gemma-2-2b-sft-dpo", "micro": 2},
}


def _seed_paths(cfg, seed):
    """Multi-seed: init from the SEEDED SFT (*-sft-seed{seed}) and write to *-sft-dpo-seed{seed}.
    Leaves the original published checkpoints untouched. Returns a shallow-copied cfg."""
    name = cfg["name"]
    sft = f"/data/checkpoints/{name}-sft-seed{seed}"
    cfg = {**cfg, "out": f"/data/checkpoints/{name}-sft-dpo-seed{seed}"}
    cfg["ckpt" if cfg["mode"] == "full" else "adapter"] = sft
    return cfg


@app.function(image=image, gpu="L4", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60 * 3)
def train_125m(limit: int = 0, epochs: int = 0, seed: int = C.SEED):
    return _run("125m", limit, epochs, seed)


@app.function(image=image, gpu="L4", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60 * 3)
def train_500m(limit: int = 0, epochs: int = 0, seed: int = C.SEED):
    return _run("500m", limit, epochs, seed)


@app.function(image=image, gpu="A100-40GB", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60 * 4)
def train_gemma(limit: int = 0, epochs: int = 0, seed: int = C.SEED):
    return _run("gemma", limit, epochs, seed)


def _run(model: str, limit: int, epochs: int, seed: int = C.SEED):
    import json, math, os, time
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import ft_data as D

    cfg = _seed_paths(CFG[model], seed)     # seeded-SFT init -> *-sft-dpo-seed{seed}
    n_epochs = epochs or EPOCHS
    torch.manual_seed(seed); dev = "cuda"
    print(f"[dpo/{model}] seed={seed} init={cfg.get('ckpt') or cfg.get('adapter')} out={cfg['out']}", flush=True)
    token = os.environ.get("HF_TOKEN")
    render = D.render_custom if cfg["render"] == "custom" else D.render_gemma

    # ---- load policy + frozen reference ----
    if cfg["mode"] == "full":
        tok = AutoTokenizer.from_pretrained(cfg["ckpt"], token=token)
        policy = AutoModelForCausalLM.from_pretrained(cfg["ckpt"], torch_dtype=torch.bfloat16).to(dev)
        ref = AutoModelForCausalLM.from_pretrained(cfg["ckpt"], torch_dtype=torch.bfloat16).to(dev)
        pad_id = tok.convert_tokens_to_ids("<|pad|>")
        trainable = policy.parameters()
    else:
        from transformers import BitsAndBytesConfig
        from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(cfg["base"], token=token)
        base = AutoModelForCausalLM.from_pretrained(cfg["base"], quantization_config=bnb,
                                                    attn_implementation="eager",
                                                    torch_dtype=torch.bfloat16, token=token)
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        # policy: base + SFT adapter -> continue-train the SFT adapter with DPO
        policy = PeftModel.from_pretrained(base, cfg["adapter"], is_trainable=True)
        for _n, _p in policy.named_parameters():
            if "lora_" in _n:
                _p.requires_grad_(True)
        # reference: a second frozen base + SFT adapter
        base2 = AutoModelForCausalLM.from_pretrained(cfg["base"], quantization_config=bnb,
                                                     attn_implementation="eager",
                                                     torch_dtype=torch.bfloat16, token=token)
        ref = PeftModel.from_pretrained(base2, cfg["adapter"]).eval()
        pad_id = tok.pad_token_id or tok.eos_token_id
        trainable = [p for p in policy.parameters() if p.requires_grad]
        policy.config.use_cache = False
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    # ---- data: build (ids,labels) for prompt+chosen and prompt+rejected ----
    rows = [json.loads(l) for l in open(PREFS, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]

    def enc(r, key):
        msgs = r["prompt"] + [{"role": "assistant", "content": r[key]}]
        ids, labels = render(msgs, tok, cfg["max_seq"])
        return {"input_ids": ids, "labels": labels}

    pairs = [(enc(r, "chosen"), enc(r, "rejected")) for r in rows]
    print(f"[dpo/{model}] {len(pairs)} preference pairs", flush=True)

    def collate(batch):
        flat = [b[0] for b in batch] + [b[1] for b in batch]     # chosen then rejected
        return D.collate(flat, pad_id)

    g = torch.Generator().manual_seed(seed)   # seed the shuffle so each run differs deterministically
    dl = DataLoader(pairs, batch_size=cfg["micro"], shuffle=True, collate_fn=collate, generator=g)

    def seq_logp(model_, batch):
        out = model_(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                     use_cache=False)
        logits = out.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        mask = (labels != -100)
        lp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = lp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (tok_lp * mask).sum(-1)                            # [2*micro]

    steps_total = math.ceil(len(dl)) * n_epochs
    opt = torch.optim.AdamW(trainable, lr=(LR_FULL if cfg["mode"] == "full" else LR_LORA))
    policy.train()
    t0 = time.time(); step = 0
    for epoch in range(n_epochs):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            n = batch["input_ids"].shape[0] // 2
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pol = seq_logp(policy, batch)
                with torch.no_grad():
                    rf = seq_logp(ref, batch)
            pol_ch, pol_rj = pol[:n], pol[n:]
            rf_ch, rf_rj = rf[:n], rf[n:]
            logits = BETA * ((pol_ch - rf_ch) - (pol_rj - rf_rj))
            loss = -F.logsigmoid(logits).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                acc = (logits > 0).float().mean().item()
                print(f"  ep{epoch} step {step}/{steps_total} loss {loss.item():.4f} "
                      f"margin_acc {acc:.2f}", flush=True)

    os.makedirs(cfg["out"], exist_ok=True)
    policy.save_pretrained(cfg["out"]); tok.save_pretrained(cfg["out"])
    vol.commit()
    wall = time.time() - t0
    print(f"[dpo/{model}] saved {cfg['out']} | {step} steps | {wall/60:.1f} min", flush=True)
    return {"model": model, "method": "dpo", "pairs": len(pairs), "steps": step,
            "wall_min": round(wall / 60, 1), "out": cfg["out"]}


@app.local_entrypoint()
def main(model: str = "500m", limit: int = 0, epochs: int = 0, seed: int = C.SEED):
    fn = {"125m": train_125m, "500m": train_500m, "gemma": train_gemma}[model]
    print(fn.remote(limit, epochs, seed))
