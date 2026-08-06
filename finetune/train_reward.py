"""Bradley-Terry reward model for RLAIF, on Modal.

    modal run train_reward.py

A scalar reward head on the slm-500m-sft backbone, trained on the 500 preferences with the
pairwise BT loss  -log σ(r(chosen) − r(rejected)).  Scores response TEXT, so one shared RM
serves both PPO runs. Reports held-out pairwise accuracy (gates the RLAIF path). Saves to
/checkpoints/reward-500m.
"""
from __future__ import annotations
import modal

import ft_config as C

app = modal.App("rl-reward")
vol = modal.Volume.from_name("ft-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate>=0.34", "numpy>=1.26,<2.0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("ft_config", "ft_data")
)

PREFS = "/data/rl/preferences.jsonl"
EPOCHS, LR, MICRO, HOLDOUT = 3, 1e-5, 4, 60


@app.function(image=image, gpu="L4", volumes={"/data": vol}, secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60)
def train(seed: int = C.SEED):
    import json, math, os, time
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import ft_data as D

    # multi-seed: RM backbone = the seeded 500M SFT; one RM per seed serves both PPO runs.
    SFT = f"/data/checkpoints/slm-500m-sft-seed{seed}"
    OUT = f"/data/checkpoints/reward-500m-seed{seed}"
    torch.manual_seed(seed); dev = "cuda"
    print(f"[reward] seed={seed} backbone={SFT} out={OUT}", flush=True)
    tok = AutoTokenizer.from_pretrained(SFT)
    pad_id = tok.convert_tokens_to_ids("<|pad|>")
    model = AutoModelForSequenceClassification.from_pretrained(
        SFT, num_labels=1, torch_dtype=torch.bfloat16).to(dev)   # scalar reward head (fresh)
    model.config.pad_token_id = pad_id

    rows = [json.loads(l) for l in open(PREFS, encoding="utf-8") if l.strip()]
    train_rows, held = rows[HOLDOUT:], rows[:HOLDOUT]

    def enc(r, key):
        ids, _ = D.render_custom(r["prompt"] + [{"role": "assistant", "content": r[key]}],
                                 tok, 1024)
        return ids

    def make(rs):
        return [(enc(r, "chosen"), enc(r, "rejected")) for r in rs]
    train_pairs, held_pairs = make(train_rows), make(held)
    print(f"[reward] {len(train_pairs)} train / {len(held_pairs)} held-out pairs", flush=True)

    def pad(seqs):
        m = max(len(s) for s in seqs)
        ids = [s + [pad_id] * (m - len(s)) for s in seqs]
        attn = [[1] * len(s) + [0] * (m - len(s)) for s in seqs]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}

    def collate(batch):
        return pad([c for c, _ in batch] + [r for _, r in batch])

    def rewards(batch):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return out.logits.squeeze(-1)                            # [2*micro]

    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(train_pairs, batch_size=MICRO, shuffle=True, collate_fn=collate, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total = math.ceil(len(dl)) * EPOCHS
    model.train(); t0 = time.time(); step = 0
    for epoch in range(EPOCHS):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            n = batch["input_ids"].shape[0] // 2
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = rewards(batch)
            loss = -F.logsigmoid(r[:n] - r[n:]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                print(f"  ep{epoch} step {step}/{total} loss {loss.item():.4f}", flush=True)

    # held-out pairwise accuracy
    model.eval(); correct = 0
    with torch.no_grad():
        for c, rj in held_pairs:
            b = {k: v.to(dev) for k, v in pad([c, rj]).items()}
            rr = rewards(b)
            correct += int(rr[0].item() > rr[1].item())
    acc = correct / max(len(held_pairs), 1)
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    vol.commit()
    wall = time.time() - t0
    print(f"[reward] saved {OUT} | held-out pairwise acc {acc:.3f} | {wall/60:.1f} min", flush=True)
    return {"held_out_accuracy": round(acc, 3), "train_pairs": len(train_pairs),
            "steps": step, "wall_min": round(wall / 60, 1)}


@app.local_entrypoint()
def main(seed: int = C.SEED):
    print(train.remote(seed))
