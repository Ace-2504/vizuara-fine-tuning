"""Count the tokens actually seen during each fine-tuning run.

Not an estimate: every training example is rendered with the same renderer the trainer used,
tokenised with that model's own tokenizer, truncated at the same max_seq, and multiplied by the
epochs that were actually run. Token counts differ per family because the tokenizers differ
(16,384 / 32,768 / 256,128 vocab), so each family is counted separately.

    ../.venv-cuda/Scripts/python.exe evaluations/count_training_tokens.py
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "finetune"))
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

from ft_data import load_jsonl, render_custom, render_gemma  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints")

# family -> (tokenizer source, renderer, max_seq)
FAMILIES = {
    "125m": (os.path.join(CKPT, "slm-125m-sft", "slm-125m-sft"), render_custom, 1024),
    "500m": (os.path.join(CKPT, "slm-500m-sft", "slm-500m-sft"), render_custom, 1024),
    "gemma": ("google/gemma-2-2b-it", render_gemma, 2048),
}

SFT = os.path.join(ROOT, "data", "sft", "qa_sft.jsonl")
RAFT = os.path.join(ROOT, "data", "sft", "raft.jsonl")
PREF = os.path.join(ROOT, "rl", "data", "preferences.jsonl")

# epochs actually run, from the training logs
EPOCHS = {
    ("125m", "sft"): 3, ("125m", "raft"): 3,
    ("500m", "sft"): 3, ("500m", "raft"): 3,
    ("gemma", "sft"): 2, ("gemma", "raft"): 2,
}


def count(rows, tok, renderer, max_len):
    total = 0
    for r in rows:
        ids, _ = renderer(r["messages"], tok, max_len)
        total += len(ids)
    return total


def pref_rows(path):
    """A preference triplet is two forward passes: prompt+chosen and prompt+rejected.

    Built exactly as train_dpo.py does — `prompt` is already a message list, and the answer is
    appended as the assistant turn.
    """
    out = []
    for r in load_jsonl(path):
        for key in ("chosen", "rejected"):
            msgs = r["prompt"] + [{"role": "assistant", "content": r[key]}]
            out.append({"messages": msgs})
    return out


def human(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    return f"{n/1_000:.0f}K"


def main():
    sft_rows = load_jsonl(SFT)
    raft_rows = load_jsonl(RAFT)
    pr = pref_rows(PREF)
    print(f"examples — sft {len(sft_rows)}  raft {len(raft_rows)}  preference passes {len(pr)}\n")

    results = {}
    for fam, (src, renderer, max_len) in FAMILIES.items():
        tok = AutoTokenizer.from_pretrained(src, token=os.environ.get("HF_TOKEN"))
        s1 = count(sft_rows, tok, renderer, max_len)
        r1 = count(raft_rows, tok, renderer, max_len)
        p1 = count(pr, tok, renderer, max_len)
        e_sft, e_raft = EPOCHS[(fam, "sft")], EPOCHS[(fam, "raft")]
        results[fam] = {
            "sft_per_epoch": s1, "sft_total": s1 * e_sft, "sft_epochs": e_sft,
            "raft_per_epoch": r1, "raft_total": r1 * e_raft, "raft_epochs": e_raft,
            "pref_per_pass": p1,
        }
        print(f"[{fam}]  vocab={len(tok)}  max_seq={max_len}")
        print(f"   QA-SFT : {human(s1)}/epoch x {e_sft} = {human(s1*e_sft)}")
        print(f"   RAFT   : {human(r1)}/epoch x {e_raft} = {human(r1*e_raft)}")
        print(f"   pref   : {human(p1)} per pass over the 500 triplets (chosen + rejected)")
        print()

    out = os.path.join(ROOT, "eval_results", "training_tokens.json")
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
              open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out)
    print("\n--- strings for the frontends ---")
    for fam, v in results.items():
        print(f"  {fam:6s} sft  {human(v['sft_total'])} tokens ({human(v['sft_per_epoch'])} x {v['sft_epochs']} epochs)")
        print(f"  {fam:6s} raft {human(v['raft_total'])} tokens ({human(v['raft_per_epoch'])} x {v['raft_epochs']} epochs)")
        print(f"  {fam:6s} pref {human(v['pref_per_pass'])} tokens per pass")


if __name__ == "__main__":
    main()
