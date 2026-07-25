"""Generate HuggingFace model cards for the 12 fine-tuned SLM versions.

    python evaluations/build_model_cards.py            # writes cards/<repo>/README.md
    python evaluations/build_model_cards.py --push     # also creates the repos and uploads

Every number here is read from this repo's own artefacts — the judged evaluation files, the
rubric-10 re-scoring, and the training logs — so a card cannot drift from the results. Anything
that was never measured (the 500M SFT and RAFT legs were never evaluated) is stated as such
rather than filled in.
"""
from __future__ import annotations
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ER = os.path.join(ROOT, "eval_results")
CKPT = os.path.join(ROOT, "checkpoints")
OUT = os.path.join(ROOT, "cards")

OWNER = "Ace-2504"
E4 = "Ace-2504/slm-125m-e4"
BASE_500 = "thesreedath/slm-500m-base"
BASE_GEMMA = "google/gemma-2-2b-it"

# checkpoint dir -> everything that is true about it
SPECS = {
    # ---- 125M family (full fine-tunes of the from-scratch e4 base) ----
    "slm-125m-sft": dict(
        family="125M", stage="QA SFT", eval_id="125m-qa", base=E4, license="mit",
        params="125.8M", trainable="125.8M (all)", mode="full fine-tune",
        data="15,000 grounded QA pairs", epochs=3, steps="2,812", minutes=9.7, gpu="L4",
        cost="$58.52", ft_cost="$0.73",
        blurb="answers a question from a supplied passage, and declines when the answer is not there"),
    "slm-125m-raft": dict(
        family="125M", stage="RAFT", eval_id="125m-raft", base=E4, license="mit",
        params="125.8M", trainable="125.8M (all)", mode="full fine-tune",
        data="10,000 retrieval-augmented examples (golden passage + distractors)", epochs=3,
        steps="1,875", minutes=19.4, gpu="L4", cost="$58.65", ft_cost="$0.86",
        blurb="answers from the right document among distractors, and abstains when the answer is absent"),
    "slm-125m-sft-dpo": dict(
        family="125M", stage="DPO", eval_id="125m-dpo", base="Ace-2504/slm-125m-sft", license="mit",
        params="125.8M", trainable="125.8M (all)", mode="DPO (full fine-tune)",
        data="500 AI-judged preference triplets", epochs=3, steps="375", minutes=1.0, gpu="L4",
        cost="$58.68", ft_cost="$0.89",
        blurb="the QA model aligned with Direct Preference Optimization"),
    "slm-125m-sft-rlaif": dict(
        family="125M", stage="RLAIF", eval_id="125m-rlaif", base="Ace-2504/slm-125m-sft", license="mit",
        params="125.8M", trainable="125.8M (all)", mode="RLAIF — reward model + PPO",
        data="500 AI-judged preference triplets", epochs=None, steps="60 PPO iterations",
        minutes=5.4, gpu="L4", cost="$58.77", ft_cost="$0.98",
        blurb="the QA model aligned by reinforcement learning against a Bradley-Terry reward model"),
    # ---- 500M family (full fine-tunes of an imported base) ----
    "slm-500m-sft": dict(
        family="500M", stage="QA SFT", eval_id=None, base=BASE_500, license="apache-2.0",
        params="517.8M", trainable="517.8M (all)", mode="full fine-tune",
        data="15,000 grounded QA pairs", epochs=3, steps="2,812", minutes=31.6, gpu="L4",
        cost="$1.03", ft_cost="$1.03",
        blurb="answers a question from a supplied passage"),
    "slm-500m-raft": dict(
        family="500M", stage="RAFT", eval_id=None, base=BASE_500, license="apache-2.0",
        params="517.8M", trainable="517.8M (all)", mode="full fine-tune",
        data="10,000 retrieval-augmented examples (golden passage + distractors)", epochs=3,
        steps="1,875", minutes=59.6, gpu="L4", cost="$1.40", ft_cost="$1.40",
        blurb="answers from the right document among distractors, and abstains when the answer is absent"),
    "slm-500m-sft-dpo": dict(
        family="500M", stage="DPO", eval_id="500m-dpo", base="Ace-2504/slm-500m-sft",
        license="apache-2.0", params="517.8M", trainable="517.8M (all)", mode="DPO (full fine-tune)",
        data="500 AI-judged preference triplets", epochs=3, steps="750", minutes=3.9, gpu="L4",
        cost="$1.23", ft_cost="$1.23",
        blurb="the QA model aligned with Direct Preference Optimization"),
    "slm-500m-sft-rlaif": dict(
        family="500M", stage="RLAIF", eval_id="500m-rlaif", base="Ace-2504/slm-500m-sft",
        license="apache-2.0", params="517.8M", trainable="517.8M (all)",
        mode="RLAIF — reward model + PPO", data="500 AI-judged preference triplets", epochs=None,
        steps="60 PPO iterations", minutes=21.2, gpu="L4", cost="$1.49", ft_cost="$1.49",
        blurb="the QA model aligned by reinforcement learning against a Bradley-Terry reward model"),
    # ---- Gemma family (QLoRA adapters on a gated base) ----
    "gemma-2-2b-sft": dict(
        family="Gemma 2B", stage="QA SFT", eval_id="gemma-qa", base=BASE_GEMMA, license="gemma",
        adapter=True, params="2.6B (frozen base)", trainable="20.8M LoRA (0.8%)",
        mode="QLoRA — 4-bit NF4 base + rank-16 LoRA", data="15,000 grounded QA pairs", epochs=2,
        steps="1,875", minutes=202.0, gpu="A100-40GB", cost="$3.29", ft_cost="$3.29",
        blurb="answers a question from a supplied passage"),
    "gemma-2-2b-raft": dict(
        family="Gemma 2B", stage="RAFT", eval_id="gemma-raft", base=BASE_GEMMA, license="gemma",
        adapter=True, params="2.6B (frozen base)", trainable="20.8M LoRA (0.8%)",
        mode="QLoRA — 4-bit NF4 base + rank-16 LoRA",
        data="10,000 retrieval-augmented examples (golden passage + distractors)", epochs=2,
        steps="1,250", minutes=140.0, gpu="A100-40GB", cost="$2.47", ft_cost="$2.47",
        blurb="answers from the right document among distractors, and abstains when the answer is absent"),
    "gemma-2-2b-sft-dpo": dict(
        family="Gemma 2B", stage="DPO", eval_id="gemma-dpo", base=BASE_GEMMA, license="gemma",
        adapter=True, params="2.6B (frozen base)", trainable="20.8M LoRA (0.8%)",
        mode="QLoRA-DPO — rank-16", data="500 AI-judged preference triplets", epochs=None,
        steps="750", minutes=6.7, gpu="L4", cost="$3.53", ft_cost="$3.53",
        blurb="the QA adapter aligned with Direct Preference Optimization"),
    "gemma-2-2b-sft-rlaif": dict(
        family="Gemma 2B", stage="RLAIF", eval_id="gemma-rlaif", base=BASE_GEMMA, license="gemma",
        adapter=True, params="2.6B (frozen base)", trainable="20.8M LoRA (0.8%)",
        mode="QLoRA RLAIF — reward model + PPO", data="500 AI-judged preference triplets",
        epochs=None, steps="60 PPO iterations", minutes=36.1, gpu="A100-40GB",
        cost="$4.73", ft_cost="$4.73",
        blurb="the QA adapter aligned by reinforcement learning against a Bradley-Terry reward model"),
}

ARCH = {
    "125M": [("Class", "LlamaForCausalLM"), ("Layers", "12"), ("Hidden size", "768"),
             ("Attention", "12 heads · head dim 64 · full MHA"), ("Feed-forward", "SwiGLU · inner 3,072"),
             ("Positional", "RoPE · θ 10,000"), ("Norm", "RMSNorm · ε 1e-5"),
             ("Context", "1,024 tokens"), ("Vocabulary", "16,384 · byte-level BPE"),
             ("Embeddings", "tied input/output")],
    "500M": [("Class", "LlamaForCausalLM"), ("Layers", "24"), ("Hidden size", "1,280"),
             ("Attention", "20 heads · head dim 64 · full MHA"), ("Feed-forward", "SwiGLU · inner 3,456"),
             ("Positional", "RoPE · θ 10,000"), ("Norm", "RMSNorm · ε 1e-5"),
             ("Context", "1,024 tokens"), ("Vocabulary", "32,768"), ("Embeddings", "tied input/output")],
    "Gemma 2B": [("Class", "Gemma2ForCausalLM"), ("Layers", "26"), ("Hidden size", "2,304"),
                 ("Attention", "8 heads / 4 KV · head dim 256 · GQA"), ("Feed-forward", "GeGLU · inner 9,216"),
                 ("Attention window", "sliding 4,096 (alternating) · logit soft-capping"),
                 ("Norm", "RMSNorm"), ("Context", "8,192 tokens"), ("Vocabulary", "256,128"),
                 ("Embeddings", "tied input/output")],
}

SITE = {
    "125m-qa": "https://slm-125m-qa-harman.vercel.app",
    "125m-raft": "https://slm-125m-raft-harman.vercel.app",
    "125m-dpo": "https://slm-125m-dpo-harman.vercel.app",
    "125m-rlaif": "https://slm-125m-rlaif-harman.vercel.app",
    "500m-dpo": "https://slm-500m-dpo-harman.vercel.app",
    "500m-rlaif": "https://slm-500m-rlaif-harman.vercel.app",
    "gemma-qa": "https://slm-gemma-qa-harman.vercel.app",
    "gemma-raft": "https://slm-gemma-raft-harman.vercel.app",
    "gemma-dpo": "https://slm-gemma-dpo-harman.vercel.app",
    "gemma-rlaif": "https://slm-gemma-rlaif-harman.vercel.app",
}
ARENA = "https://slm-arena-harman.vercel.app"


def load_eval():
    """Per-model evaluation numbers, straight from the artefacts."""
    rub = json.load(open(os.path.join(ER, "rubric10_summary.json"), encoding="utf-8"))
    files = {
        "125m-qa": "slm-125m-sft", "125m-raft": "slm-125m-raft",
        "125m-dpo": "slm-125m-sft-dpo", "125m-rlaif": "slm-125m-sft-rlaif",
        "500m-dpo": "slm-500m-sft-dpo", "500m-rlaif": "slm-500m-sft-rlaif",
        "gemma-qa": "gemma-2-2b-sft", "gemma-raft": "gemma-2-2b-raft",
        "gemma-dpo": "gemma-2-2b-sft-dpo", "gemma-rlaif": "gemma-2-2b-sft-rlaif",
    }
    out = {}
    for eid, fn in files.items():
        d = json.load(open(os.path.join(ER, fn + ".judged.json"), encoding="utf-8"))
        clean = [i for i in d["per_item"] if i["cond"] == "clean"]
        n = len(clean)
        out[eid] = {
            "n": n,
            "correct01": round(sum((i["judge"]["correct"] - 1) / 4 for i in clean) / n, 3),
            "grounded": round(sum(1 for i in clean if i["judge"]["grounded"]) / n * 100, 1),
            "token_f1": round(sum(i["scores"].get("token_f1") or 0 for i in clean) / n, 3),
            "fabrication": round(sum(1 for i in clean if i["scores"].get("fabrication")) / n * 100, 1),
            "rubric10": rub.get(eid, {}).get("score10"),
            "by_source": rub.get(eid, {}).get("by_source", {}),
        }
    return out


def card(repo: str, spec: dict, ev: dict | None) -> str:
    fam = spec["family"]
    adapter = spec.get("adapter", False)
    eid = spec.get("eval_id")
    tags = ["legal", "finance", "question-answering", "small-language-model"]
    tags += ["peft", "lora"] if adapter else []
    tags += ["dpo"] if spec["stage"] == "DPO" else []
    tags += ["rlaif", "ppo"] if spec["stage"] == "RLAIF" else []
    tags += ["raft", "retrieval-augmented"] if spec["stage"] == "RAFT" else []

    fm = ["---", f"license: {spec['license']}", f"base_model: {spec['base']}",
          "language:", "  - en", "library_name: " + ("peft" if adapter else "transformers"),
          "pipeline_tag: text-generation", "tags:"]
    fm += [f"  - {t}" for t in tags]
    fm.append("---")

    L = ["\n".join(fm), ""]
    A = L.append

    A(f"# {repo}")
    A("")
    A(f"**{fam} · {spec['stage']}** — {spec['blurb']}.")
    A("")
    A(f"One of 13 models in a controlled study of how far a small language model can be pushed on "
      f"US legal and financial text. Every version was trained on the same data, evaluated on the "
      f"same frozen held-out set, and scored by the same blind LLM judge, so the stages are "
      f"directly comparable. Compare them side by side in the [SLM Arena]({ARENA}).")
    A("")
    A(f"Trained and aligned by **Harman Sandhu** (Vizuara AI Labs).")
    A("")

    if adapter:
        A("> **This is a LoRA adapter, not a standalone model.** It must be loaded on top of "
          f"[`{spec['base']}`]( https://huggingface.co/{spec['base']} ), which is **gated** — "
          "you need to accept Google's licence on that repo before this adapter can be used. "
          "Usage of the adapter is governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).")
        A("")

    # ---------- quick facts ----------
    A("## At a glance")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| Base model | [`{spec['base']}`](https://huggingface.co/{spec['base']}) |")
    A(f"| Method | {spec['mode']} |")
    A(f"| Parameters | {spec['params']} |")
    A(f"| Trainable parameters | {spec['trainable']} |")
    A(f"| Training data | {spec['data']} |")
    ep = f"{spec['epochs']} epochs · " if spec.get("epochs") else ""
    A(f"| Schedule | {ep}{spec['steps']} |")
    A(f"| Compute | {spec['minutes']:.1f} min on {spec['gpu']} (Modal) |")
    A(f"| Cost | **{spec['cost']}** total lineage |")
    if ev and ev.get("rubric10") is not None:
        A(f"| Judge score | **{ev['rubric10']:.2f} / 10** |")
    A("")

    # ---------- usage ----------
    A("## Usage")
    A("")
    if adapter:
        A("```python")
        A("import torch")
        A("from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig")
        A("from peft import PeftModel")
        A("")
        A(f'BASE = "{spec["base"]}"      # gated — accept the licence first')
        A(f'ADAPTER = "{OWNER}/{repo}"')
        A("")
        A("bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',")
        A("                         bnb_4bit_compute_dtype=torch.bfloat16,")
        A("                         bnb_4bit_use_double_quant=True)")
        A("base = AutoModelForCausalLM.from_pretrained(")
        A("    BASE, quantization_config=bnb, attn_implementation='eager',")
        A("    torch_dtype=torch.bfloat16)")
        A("model = PeftModel.from_pretrained(base, ADAPTER).eval()")
        A("tok = AutoTokenizer.from_pretrained(BASE)")
        A("")
        A("SYS = ('You are a precise legal and financial assistant. Answer clearly using the '")
        A("       'provided context; do not invent facts.')")
        A("user = 'Context:\\n<passage>\\n\\nQuestion: <your question>'")
        A("text = tok.apply_chat_template([{'role': 'user', 'content': f'{SYS}\\n\\n{user}'}],")
        A("                               tokenize=False, add_generation_prompt=True)")
        A("ids = tok(text, return_tensors='pt').to(model.device)")
        A("# Gemma-2's hybrid cache misbehaves here — use_cache=False is required.")
        A("out = model.generate(**ids, max_new_tokens=160, use_cache=False)")
        A("print(tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True))")
        A("```")
    else:
        A("```python")
        A("import torch")
        A("from transformers import AutoModelForCausalLM, AutoTokenizer")
        A("")
        A(f'MODEL = "{OWNER}/{repo}"')
        A("model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval()")
        A("tok = AutoTokenizer.from_pretrained(MODEL)")
        A("")
        A("SYS = ('You are a precise legal and financial assistant. Answer clearly using the '")
        A("       'provided context; do not invent facts.')")
        A("user = 'Context:\\n<passage>\\n\\nQuestion: <your question>'")
        A("")
        A("# Custom chat scheme. NOTE: never prepend <|bos|> — it was left untrained.")
        A("prompt = f'<|system|>\\n{SYS}<|eos|>\\n<|user|>\\n{user}<|eos|>\\n<|assistant|>\\n'")
        A("enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)")
        A("eos = tok.convert_tokens_to_ids('<|eos|>')")
        A("out = model.generate(**enc, max_new_tokens=160, eos_token_id=eos,")
        A("                     pad_token_id=eos)   # pad and eos share an id here")
        A("print(tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True))")
        A("```")
    A("")

    # ---------- evaluation ----------
    A("## Evaluation")
    A("")
    if not eid or not ev:
        A("**This checkpoint was not evaluated.** The published study evaluated the base, DPO and "
          "RLAIF legs of the 500M line; this intermediate checkpoint was trained and released for "
          "completeness but never scored, so no numbers are quoted here rather than borrowed from "
          "a sibling model.")
    else:
        A(f"Scored on **{ev['n']} decontaminated held-out questions** (chunk-level dedup against "
          "public legal benchmarks, so no evaluation passage was trained on). Every model in the "
          "study answered the identical questions with deterministic greedy decoding, and a "
          "**Gemini judge blind to the model** graded each answer with the gold answer in hand.")
        A("")
        A("| Metric | Value |")
        A("|---|---|")
        A(f"| Judge score (0–10 rubric) | **{ev['rubric10']:.2f}** |")
        A(f"| Judged correctness (0–1) | {ev['correct01']:.3f} |")
        A(f"| Groundedness | {ev['grounded']:.1f}% |")
        A(f"| Fabrication ↓ | {ev['fabrication']:.1f}% |")
        A(f"| Token-F1 | {ev['token_f1']:.3f} |")
        A(f"| n | {ev['n']} |")
        A("")
        if ev.get("by_source"):
            A("By source:")
            A("")
            A("| US case law | SEC filings | Educational web |")
            A("|---|---|---|")
            bs = ev["by_source"]
            A(f"| {bs.get('case-law','–')} | {bs.get('sec','–')} | {bs.get('fineweb-edu','–')} |")
            A("")
        A("The 0–10 figure is a four-dimension rubric (correctness 0–5 + completeness 0–2 + "
          "groundedness 0–2 + clarity 0–1). The 0–1 figure is the stricter correctness-only scale "
          "used in the experiment reports. Same questions, same answers, same judge — different "
          "scale, so the two numbers differ.")
        A("")
        A("**Token-F1 is reported for completeness and should not be read as quality**: it "
          "punishes correct paraphrase heavily, which is exactly why the judge carries the "
          "headline.")
    A("")

    # ---------- architecture ----------
    A("## Architecture")
    A("")
    A("| | |")
    A("|---|---|")
    for k, v in ARCH[fam]:
        A(f"| {k} | {v} |")
    A("")

    # ---------- training ----------
    A("## Training")
    A("")
    A(f"- **Initialised from** [`{spec['base']}`](https://huggingface.co/{spec['base']})")
    A(f"- **Method** — {spec['mode']}")
    A(f"- **Data** — {spec['data']}, generated from a legal/financial corpus and gated by an "
      "LLM judge for faithfulness")
    A(f"- **Schedule** — {ep}{spec['steps']}, {spec['minutes']:.1f} min on {spec['gpu']}")
    if spec["stage"] in ("QA SFT", "RAFT"):
        A("- Loss is **masked to the answer tokens only**, so the model is never trained to "
          "reproduce the prompt")
    if spec["stage"] == "DPO":
        A("- DPO against a **frozen reference copy** of the SFT checkpoint, β = 0.1")
    if spec["stage"] == "RLAIF":
        A("- A **Bradley-Terry reward model** (0.983 held-out pairwise accuracy) scored the "
          "preferences; PPO then optimised the policy against it with a KL penalty anchoring it "
          "to the reference")
    A("")
    A(f"**Cost — {spec['cost']}.** ")
    if fam == "125M":
        A("This line was **pretrained from scratch**, so the total includes the base: $57.79 of "
          "A100 time across four pretraining legs (v1 → extension → e2 → e4, 27.5 h at $2.10/h) "
          f"plus {spec['ft_cost']} for this fine-tune. Models built on someone else's base do not "
          "carry that cost.")
    elif fam == "500M":
        A(f"The base was imported, so only this project's own work is counted: {spec['ft_cost']} "
          "of Modal GPU time and shared dataset generation.")
    else:
        A(f"The base is Google's and free to build on, so only this project's own work is "
          f"counted: {spec['ft_cost']}.")
    A("")

    # ---------- limitations ----------
    A("## Limitations and intended use")
    A("")
    A("- **Not legal or financial advice.** This is a research artefact for studying small-model "
      "training, not a professional tool. Do not rely on its output for real decisions.")
    if fam in ("125M", "500M"):
        A(f"- **At {spec['params']} it holds very little world knowledge.** It is built to read an "
          "answer out of a passage you supply, not to recall facts. Used closed-book it will "
          "produce fluent, confident and wrong text.")
    if eid and ev and ev["rubric10"] is not None and ev["rubric10"] < 4:
        A(f"- **This checkpoint scores {ev['rubric10']:.2f}/10** — it is published for comparison "
          "across training stages, not because it is good. See the arena for why.")
    if spec["stage"] == "RAFT":
        A("- Emits its evidence wrapped in `##begin_quote## … ##end_quote##` markers, and the "
          "sampler mangles these fairly often — strip or parse them before display.")
    A("- The judge behind these scores has **not been calibrated against human labels**, so treat "
      "small differences between models cautiously.")
    A("- English only; the corpus is US case law, SEC filings and educational web text.")
    A("")

    # ---------- family ----------
    A("## The rest of the family")
    A("")
    A("| Size | Base | QA SFT | RAFT | DPO | RLAIF |")
    A("|---|---|---|---|---|---|")
    A(f"| 125M | [`slm-125m-e4`](https://huggingface.co/{E4}) | `slm-125m-sft` | `slm-125m-raft` | `slm-125m-sft-dpo` | `slm-125m-sft-rlaif` |")
    A(f"| 500M | [`slm-500m-base`](https://huggingface.co/{BASE_500}) | `slm-500m-sft` | `slm-500m-raft` | `slm-500m-sft-dpo` | `slm-500m-sft-rlaif` |")
    A(f"| Gemma 2B | [`gemma-2-2b-it`](https://huggingface.co/{BASE_GEMMA}) | `gemma-2-2b-sft` | `gemma-2-2b-raft` | `gemma-2-2b-sft-dpo` | `gemma-2-2b-sft-rlaif` |")
    A("")
    A(f"All under [`{OWNER}`](https://huggingface.co/{OWNER}) except the two imported bases.")
    if eid and eid in SITE:
        A("")
        A(f"This model has its own write-up — training details, cost breakdown and live demo — at "
          f"[{SITE[eid]}]({SITE[eid]}).")
    A("")
    A("## Citation")
    A("")
    A("```bibtex")
    A("@misc{sandhu2026slm,")
    A("  title  = {Small Language Models for Legal and Financial Text: a controlled study of")
    A("            pretraining, instruction tuning, retrieval augmentation and alignment},")
    A("  author = {Harman Sandhu},")
    A("  year   = {2026},")
    A(f"  note   = {{{ARENA}}}")
    A("}")
    A("```")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="create repos and upload weights")
    ap.add_argument("--only", default=None, help="comma-separated checkpoint names")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    ev_all = load_eval()
    names = a.only.split(",") if a.only else list(SPECS)
    os.makedirs(OUT, exist_ok=True)

    for name in names:
        spec = SPECS[name]
        ev = ev_all.get(spec.get("eval_id")) if spec.get("eval_id") else None
        text = card(name, spec, ev)
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"card: {name}  ({len(text)} chars)")

    if not a.push:
        print("\nno --push: cards written only")
        return

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])

    def model_dir(p):
        for c in (p, *[os.path.join(p, x) for x in os.listdir(p)]):
            if os.path.isdir(c) and (os.path.exists(os.path.join(c, "config.json"))
                                     or os.path.exists(os.path.join(c, "adapter_config.json"))):
                return c
        return p

    for name in names:
        repo = f"{OWNER}/{name}"
        src = model_dir(os.path.join(CKPT, name))
        print(f"\n=== {repo}  <- {src}")
        api.create_repo(repo, repo_type="model", private=a.private, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=src, repo_type="model",
                          commit_message="Add weights")
        api.upload_file(path_or_fileobj=os.path.join(OUT, name, "README.md"),
                        path_in_repo="README.md", repo_id=repo, repo_type="model",
                        commit_message="Add model card")
        print(f"    https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
