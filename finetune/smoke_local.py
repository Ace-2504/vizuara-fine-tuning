"""Local sanity checks for the 125M fine-tune, no Modal/GPU needed.

Uses the corpus tokenizer (data/tokenizer/) — identical to e4's — to verify, over the FULL
datasets:
  1. loss masking is assistant-only (decode one batch's supervised tokens),
  2. RAFT truncation at max_seq=1024 is ~0 (guide P.7 / Track-B),
  3. sequence-length distribution is sane.
"""
import sys, statistics
from transformers import AutoTokenizer
import ft_data as D

TOK_DIR = "../data/tokenizer"
MAX = 1024

tok = AutoTokenizer.from_pretrained(TOK_DIR)
print(f"tokenizer: vocab={len(tok)} eos={tok.convert_tokens_to_ids('<|eos|>')} "
      f"pad={tok.convert_tokens_to_ids('<|pad|>')}", flush=True)

for method, path in [("sft", "../data/sft/qa_sft.jsonl"), ("raft", "../data/sft/raft.jsonl")]:
    rows = D.load_jsonl(path)
    ds = D.ChatDataset(rows, tok, D.render_custom, MAX)
    lens = [len(ds.ex[i][0]) for i in range(len(ds))]
    print(f"\n[{method}] rows={len(rows)} kept={len(ds)} dropped={ds.dropped} "
          f"trunc(>{MAX})={ds.trunc}", flush=True)
    print(f"[{method}] seqlen: median={int(statistics.median(lens))} "
          f"p95={sorted(lens)[int(len(lens)*0.95)]} max={max(lens)}", flush=True)

    # Masking check: decode ONLY the supervised (label != -100) tokens of the first example.
    ids, labels = ds.ex[0]
    sup = [i for i, l in zip(ids, labels) if l != D.IGNORE]
    masked_txt = tok.decode(sup)
    print(f"[{method}] supervised tokens ({len(sup)}) decode -> {masked_txt[:220]!r}", flush=True)
    # The last assistant content should appear; no <|user|>/<|system|> role text should.
    assert "<|user|>" not in masked_txt and "<|system|>" not in masked_txt, \
        "MASKING BUG: prompt tokens are being supervised"
    print(f"[{method}] OK: only assistant text supervised", flush=True)

print("\nALL LOCAL CHECKS PASSED", flush=True)
