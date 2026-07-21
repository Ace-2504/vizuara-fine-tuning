"""Fine-tuning config for the 500M and Gemma-2B models. One place for every constant.

Two versions per model are produced by running its train script twice with --method:
    --method sft   -> trains on data/sft/qa_sft.jsonl   (the SFT version)
    --method raft  -> trains on data/sft/raft.jsonl     (the RAFT version)
Both initialise from the base checkpoint, so the two versions are directly comparable.
"""
from __future__ import annotations

# NOTE: unlike the dataset-build config, we do NOT disable the implicit HF token here —
# gemma-2-2b-it is GATED and needs the (now-valid, refreshed) cached token to download.

# --- datasets (already built, shared, text-level chat JSONL) -------------------------
DATA = {
    "sft":  "/data/sft/qa_sft.jsonl",     # on the Modal volume
    "raft": "/data/sft/raft.jsonl",
}
EVAL = "/data/sft/eval.jsonl"
# Per-model max sequence length. The 500M's tokenizer was trained ON this corpus so it packs
# legal text efficiently (RAFT max ~936 -> fits 1024). Gemma's general 256k tokenizer needs
# MORE tokens for this domain (+ chat-template turn tokens): RAFT reaches ~1129, so cap higher
# — its native context is 8192, so 2048 is free and eliminates golden-document truncation.

# --- shared training defaults --------------------------------------------------------
SEED = 1337
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0

# --- 500M: full fine-tune ------------------------------------------------------------
M500 = {
    "name": "slm-500m",
    "base": "thesreedath/slm-500m-base",
    "mode": "full",
    "lr": 2e-5,
    "min_lr": 2e-6,
    "epochs": 3,
    "micro_batch": 4,                     # full-FT + long RAFT seqs OOM'd at 8 on L4 (frag)
    "grad_accum": 4,                      # effective batch 16 (unchanged -> still comparable)
    "gpu": "L4",
    "max_seq": 1024,                      # its context ceiling
    "bos": False,                         # <|bos|> untrained -> never prepend
    "eos_token": "<|eos|>",              # config.json points eos at the wrong id; fix explicitly
}

# --- Gemma-2-2b-it: QLoRA ------------------------------------------------------------
GEMMA = {
    "name": "gemma-2-2b",
    "base": "google/gemma-2-2b-it",       # GATED — needs HF_TOKEN + accepted licence
    "mode": "qlora",
    "lr": 2e-4,
    "min_lr": 2e-5,
    "epochs": 2,
    "micro_batch": 2,                     # 256k vocab x 2048 seq -> logits OOM at 4 on L4
    "grad_accum": 8,                      # effective batch 16
    "gpu": "L4",
    "max_seq": 2048,                     # general tokenizer needs more tokens here; 8192 native
    "attn_impl": "eager",                # Gemma-2 soft-capping: eager is the safe choice
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
}

CKPT_ROOT = "/data/checkpoints"          # /data/checkpoints/<name>-<method>/
