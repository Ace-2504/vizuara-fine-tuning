"""Fine-tune thesreedath/slm-500m-base (full FT) on Modal. Produces one version per run:

    modal run train_500m.py --method sft     # -> /data/checkpoints/slm-500m-sft
    modal run train_500m.py --method raft    # -> /data/checkpoints/slm-500m-raft

Both init from the base checkpoint, so the two versions are directly comparable.
"""
from __future__ import annotations

import modal

import ft_config as C

app = modal.App("ft-slm-500m")
vol = modal.Volume.from_name("ft-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate>=0.34",
                 "numpy>=1.26,<2.0")
    .env({"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})     # kill the stale implicit token
    .add_local_python_source("ft_config", "ft_data")
)


@app.function(image=image, gpu=C.M500["gpu"], volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60 * 4)
def train(method: str, limit: int = 0, epochs: int = 0):
    import math, os, time
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    import ft_data as D

    cfg = C.M500
    n_epochs = epochs or cfg["epochs"]     # smoke tests pass epochs=1
    assert method in C.DATA, f"method must be sft|raft, got {method}"
    torch.manual_seed(C.SEED)
    dev = "cuda"
    token = os.environ.get("HF_TOKEN")     # public model -> None ok; explicit avoids stale cache

    tok = AutoTokenizer.from_pretrained(cfg["base"], token=token)
    model = AutoModelForCausalLM.from_pretrained(cfg["base"], torch_dtype=torch.bfloat16,
                                                 token=token).to(dev)
    # config.json points eos at the wrong id; fix so generation/eos-loss are correct.
    eos_id = tok.convert_tokens_to_ids(cfg["eos_token"])
    model.config.eos_token_id = eos_id
    pad_id = tok.convert_tokens_to_ids("<|pad|>")

    rows = D.load_jsonl(C.DATA[method])
    if limit:
        rows = rows[:limit]
    ds = D.ChatDataset(rows, tok, D.render_custom, cfg["max_seq"])
    print(f"[500m/{method}] {len(ds)} examples | dropped {ds.dropped} | trunc {ds.trunc}", flush=True)
    dl = DataLoader(ds, batch_size=cfg["micro_batch"], shuffle=True,
                    collate_fn=lambda b: D.collate(b, pad_id))

    steps_per_epoch = math.ceil(len(dl) / cfg["grad_accum"])
    total = steps_per_epoch * n_epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=C.WEIGHT_DECAY,
                            betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, int(total * C.WARMUP_RATIO), total)

    model.train()
    t0 = time.time(); step = 0; accum = 0; running = 0.0
    for epoch in range(n_epochs):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / cfg["grad_accum"]
            loss.backward(); running += loss.item() * cfg["grad_accum"]; accum += 1
            if accum == cfg["grad_accum"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                opt.step(); sched.step(); opt.zero_grad(); accum = 0; step += 1
                if step % 20 == 0:
                    print(f"  epoch {epoch} step {step}/{total} loss {running/20/cfg['grad_accum']*cfg['grad_accum']:.4f} "  # noqa
                          f"lr {sched.get_last_lr()[0]:.2e}", flush=True); running = 0.0

    out = f"{C.CKPT_ROOT}/{cfg['name']}-{method}"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    vol.commit()
    wall = time.time() - t0
    print(f"[500m/{method}] saved {out} | {step} steps | {wall/60:.1f} min", flush=True)
    return {"model": cfg["name"], "method": method, "examples": len(ds),
            "steps": step, "wall_min": round(wall / 60, 1), "out": out}


@app.local_entrypoint()
def main(method: str = "sft", limit: int = 0, epochs: int = 0):
    print(train.remote(method, limit, epochs))
