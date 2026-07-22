"""PPO (RLAIF) against the trained reward model, on Modal. Hand-rolled compact PPO (no TRL
version risk): generate -> reward from RM (minus KL-to-reference penalty) -> clipped
policy-gradient update over the generated tokens, with a batch-mean advantage baseline.

    modal run train_ppo.py --model 500m     # slm-500m-sft   -> slm-500m-sft-rlaif   (full)
    modal run train_ppo.py --model gemma     # gemma-2-2b-sft -> gemma-2-2b-sft-rlaif (QLoRA)

Policy = the SFT model; reference = frozen SFT copy (KL anchor); reward = /checkpoints/reward-500m
(scores response text). Prompts = the preference-set prompts.
"""
from __future__ import annotations
import modal

import ft_config as C

app = modal.App("rl-ppo")
vol = modal.Volume.from_name("ft-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", "transformers==4.46.3", "accelerate>=0.34",
                 "peft>=0.13", "bitsandbytes>=0.44", "numpy>=1.26,<2.0")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("ft_config", "ft_data")
)

PREFS = "/data/rl/preferences.jsonl"
RM = "/data/checkpoints/reward-500m"
GEN_TOK = 96
KL_COEF = 0.15
CLIP = 0.2
PPO_EPOCHS = 2
ROLLOUT_BS = 16       # 500m; gemma overridden to 4 below
LR_FULL, LR_LORA = 1e-5, 1e-4
OUTER = 60                                   # rollout batches

CFG = {
    "500m":  {"gpu": "L4", "mode": "full", "render": "custom", "max_seq": 1024,
              "ckpt": "/data/checkpoints/slm-500m-sft", "out": "/data/checkpoints/slm-500m-sft-rlaif"},
    "gemma": {"gpu": "A100-40GB", "mode": "qlora", "render": "gemma", "max_seq": 1024,
              "base": "google/gemma-2-2b-it", "adapter": "/data/checkpoints/gemma-2-2b-sft",
              "out": "/data/checkpoints/gemma-2-2b-sft-rlaif"},
}


@app.function(image=image, gpu="L4", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60 * 3)
def train_500m(outer: int = 0):
    return _run("500m", outer)


@app.function(image=image, gpu="A100-40GB", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("hf-token")], timeout=60 * 60 * 5)
def train_gemma(outer: int = 0):
    return _run("gemma", outer)


def _run(model: str, outer: int):
    import json, os, time, random
    import torch, torch.nn.functional as F
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              AutoModelForSequenceClassification)
    import ft_data as D

    cfg = CFG[model]; n_outer = outer or OUTER
    rollout_bs = ROLLOUT_BS if cfg["mode"] == "full" else 4       # gemma logits are 256k-wide
    torch.manual_seed(C.SEED); dev = "cuda"
    token = os.environ.get("HF_TOKEN")

    # ---- policy + frozen reference ----
    if cfg["mode"] == "full":
        tok = AutoTokenizer.from_pretrained(cfg["ckpt"], token=token)
        policy = AutoModelForCausalLM.from_pretrained(cfg["ckpt"], torch_dtype=torch.bfloat16).to(dev)
        ref = AutoModelForCausalLM.from_pretrained(cfg["ckpt"], torch_dtype=torch.bfloat16).to(dev)
        trainable = list(policy.parameters()); pad_id = tok.convert_tokens_to_ids("<|pad|>")
    else:
        from transformers import BitsAndBytesConfig
        from peft import PeftModel, prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(cfg["base"], token=token)
        b1 = AutoModelForCausalLM.from_pretrained(cfg["base"], quantization_config=bnb,
             attn_implementation="eager", torch_dtype=torch.bfloat16, token=token)
        b1 = prepare_model_for_kbit_training(b1, use_gradient_checkpointing=True)
        policy = PeftModel.from_pretrained(b1, cfg["adapter"], is_trainable=True)
        for n, p in policy.named_parameters():
            if "lora_" in n: p.requires_grad_(True)
        b2 = AutoModelForCausalLM.from_pretrained(cfg["base"], quantization_config=bnb,
             attn_implementation="eager", torch_dtype=torch.bfloat16, token=token)
        ref = PeftModel.from_pretrained(b2, cfg["adapter"]).eval()
        trainable = [p for p in policy.parameters() if p.requires_grad]
        pad_id = tok.pad_token_id or tok.eos_token_id
    for p in ref.parameters(): p.requires_grad_(False)
    ref.eval()
    eos_id = tok.convert_tokens_to_ids("<|eos|>") if cfg["render"] == "custom" else tok.eos_token_id
    # Gemma-2's HybridCache has a bf16/fp32 bug on this build; disabling the cache during
    # generation avoids it (slower, but correct and fine for short rollouts).
    gen_kw = {} if cfg["mode"] == "full" else {"use_cache": False}

    # ---- reward model (scores response text with its own 500M tokenizer) ----
    rm_tok = AutoTokenizer.from_pretrained(RM)
    rm = AutoModelForSequenceClassification.from_pretrained(RM, num_labels=1,
         torch_dtype=torch.bfloat16).to(dev).eval()
    rm.config.pad_token_id = rm_tok.convert_tokens_to_ids("<|pad|>")

    rows = [json.loads(l) for l in open(PREFS, encoding="utf-8") if l.strip()]
    rng = random.Random(C.SEED)

    def prompt_ids(r):
        # render prompt only (no answer) -> generation prefix
        ids, _ = D.render_custom(r["prompt"], tok, cfg["max_seq"]) if cfg["render"] == "custom" \
            else D.render_gemma(r["prompt"], tok, cfg["max_seq"])
        return ids

    def score(prompt_txt, resp_txt):
        t = rm_tok(prompt_txt + "\n" + resp_txt, return_tensors="pt", truncation=True,
                   max_length=1024)
        t = {k: v.to(dev) for k, v in t.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            return rm(**t).logits.squeeze().item()

    def logps_of(m, ids, gen_mask):
        out = m(input_ids=ids, use_cache=False)
        logits = out.logits[:, :-1]                                # [B, T-1, V] bf16
        gathered = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        lse = torch.logsumexp(logits, dim=-1)                      # no fp32 [B,T,V] copy
        return ((gathered - lse) * gen_mask).float()               # [B, T-1]

    opt = torch.optim.AdamW(trainable, lr=(LR_FULL if cfg["mode"] == "full" else LR_LORA))
    policy.train(); t0 = time.time()
    for it in range(n_outer):
        batch = rng.sample(rows, min(rollout_bs, len(rows)))
        seqs, gmasks, rewards, prompts_txt = [], [], [], []
        # ---- rollout (generate) ----
        policy.eval()
        for r in batch:
            pids = torch.tensor([prompt_ids(r)]).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                g = policy.generate(pids, max_new_tokens=GEN_TOK, do_sample=True, temperature=0.8,
                                    top_p=0.95, pad_token_id=pad_id, eos_token_id=eos_id, **gen_kw)
            full = g[0]
            gen = full[pids.shape[1]:]
            resp = tok.decode(gen, skip_special_tokens=True)
            ptxt = r["prompt"][-1]["content"]
            rew = score(ptxt, resp)
            mask = torch.zeros(full.shape[0] - 1, device=dev)
            mask[pids.shape[1] - 1:] = 1.0                         # supervise generated tokens
            seqs.append(full); gmasks.append(mask); rewards.append(rew); prompts_txt.append(ptxt)
        policy.train()
        # pad rollouts
        m = max(s.shape[0] for s in seqs)
        ids = torch.stack([F.pad(s, (0, m - s.shape[0]), value=pad_id) for s in seqs])
        gm = torch.stack([F.pad(x, (0, m - 1 - x.shape[0]), value=0.0) for x in gmasks])
        rew = torch.tensor(rewards, device=dev)
        adv = (rew - rew.mean()) / (rew.std() + 1e-6)             # batch-mean baseline
        with torch.no_grad():
            old_lp = logps_of(policy, ids, gm)
            ref_lp = logps_of(ref, ids, gm)
        # ---- PPO update epochs ----
        for _ in range(PPO_EPOCHS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                new_lp = logps_of(policy, ids, gm)
            ratio = torch.exp((new_lp - old_lp).clamp(-5, 5))
            a = adv.unsqueeze(1)
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a)
            kl = (new_lp - ref_lp)
            loss = ((pg + KL_COEF * kl) * gm).sum() / gm.sum().clamp(min=1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad()
        if it % 5 == 0:
            print(f"  iter {it}/{n_outer} reward {rew.mean().item():.3f} "
                  f"kl {(old_lp-ref_lp).sum(1).mean().item():.2f} loss {loss.item():.4f}", flush=True)

    os.makedirs(cfg["out"], exist_ok=True)
    policy.save_pretrained(cfg["out"]); tok.save_pretrained(cfg["out"])
    vol.commit()
    wall = time.time() - t0
    print(f"[ppo/{model}] saved {cfg['out']} | {n_outer} iters | {wall/60:.1f} min", flush=True)
    return {"model": model, "method": "rlaif-ppo", "iters": n_outer,
            "final_reward": round(rew.mean().item(), 3), "wall_min": round(wall / 60, 1)}


@app.local_entrypoint()
def main(model: str = "500m", outer: int = 0):
    fn = train_500m if model == "500m" else train_gemma
    print(fn.remote(outer))
