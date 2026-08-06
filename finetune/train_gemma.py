"""Fine-tune google/gemma-2-2b-it with QLoRA on Modal. One version per run:

    modal run train_gemma.py --method sft     # -> /data/checkpoints/gemma-2-2b-sft   (adapter)
    modal run train_gemma.py --method raft    # -> /data/checkpoints/gemma-2-2b-raft  (adapter)

Gemma is GATED: needs a Modal secret 'hf-token' holding a valid HF_TOKEN with accepted licence.
Saves only the LoRA adapter (a few hundred MB), not the 2B base.
"""
from __future__ import annotations

import modal

import ft_config as C

app = modal.App("ft-gemma-2b")
vol = modal.Volume.from_name("ft-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate>=0.34",
                 "peft>=0.13", "bitsandbytes>=0.44", "numpy>=1.26,<2.0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})   # curb fragmentation OOM
    .add_local_python_source("ft_config", "ft_data")
)


@app.function(image=image, gpu=C.GEMMA["gpu"], volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60 * 8)
def train(method: str, limit: int = 0, epochs: int = 0, seed: int = C.SEED):
    import math, os, time
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              get_cosine_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import ft_data as D

    cfg = C.GEMMA
    n_epochs = epochs or cfg["epochs"]
    assert method in C.DATA
    torch.manual_seed(seed)                # multi-seed: overrides C.SEED per run
    dev = "cuda"
    token = os.environ["HF_TOKEN"]     # gated -> required

    tok = AutoTokenizer.from_pretrained(cfg["base"], token=token)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base"], quantization_config=bnb, attn_implementation=cfg["attn_impl"],
        torch_dtype=torch.bfloat16, token=token)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=cfg["lora_targets"]))
    model.print_trainable_parameters()
    model.config.use_cache = False

    rows = D.load_jsonl(C.DATA[method])
    if limit:
        rows = rows[:limit]
    ds = D.ChatDataset(rows, tok, D.render_gemma, cfg["max_seq"])
    print(f"[gemma/{method}] {len(ds)} examples | dropped {ds.dropped} | trunc {ds.trunc}", flush=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    g = torch.Generator().manual_seed(seed)   # seed the shuffle so each run differs
    dl = DataLoader(ds, batch_size=cfg["micro_batch"], shuffle=True, generator=g,
                    collate_fn=lambda b: D.collate(b, pad_id))

    steps_per_epoch = math.ceil(len(dl) / cfg["grad_accum"])
    total = steps_per_epoch * n_epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"],
                            weight_decay=C.WEIGHT_DECAY)
    sched = get_cosine_schedule_with_warmup(opt, int(total * C.WARMUP_RATIO), total)

    model.train()
    t0 = time.time(); step = 0; accum = 0; running = 0.0
    for epoch in range(n_epochs):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch); loss = out.loss / cfg["grad_accum"]
            loss.backward(); running += out.loss.item(); accum += 1
            if accum == cfg["grad_accum"]:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               C.GRAD_CLIP)
                opt.step(); sched.step(); opt.zero_grad(); accum = 0; step += 1
                if step % 20 == 0:
                    print(f"  epoch {epoch} step {step}/{total} loss {running/20:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}", flush=True); running = 0.0

    out = f"{C.CKPT_ROOT}/{cfg['name']}-{method}-seed{seed}"   # never clobbers the original
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)          # adapter only
    tok.save_pretrained(out)
    vol.commit()
    wall = time.time() - t0
    print(f"[gemma/{method}] saved adapter {out} | {step} steps | {wall/60:.1f} min", flush=True)
    return {"model": cfg["name"], "method": method, "examples": len(ds),
            "steps": step, "wall_min": round(wall / 60, 1), "out": out}


@app.local_entrypoint()
def main(method: str = "sft", limit: int = 0, epochs: int = 0, seed: int = C.SEED):
    print(train.remote(method, limit, epochs, seed))
