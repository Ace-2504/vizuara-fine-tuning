"""Render the shared chat-JSONL datasets into tokenized, loss-masked training examples.

The datasets are TEXT (system/user/assistant). Each model renders them with its OWN
tokenizer + chat template, and loss is computed ONLY on the assistant turn.

Two renderers:
- render_custom  : the 500M's scheme  <|role|>\n{content}<|eos|>\n , NO bos (untrained).
- render_gemma   : Gemma's chat template, system folded into the first user turn, bos kept.
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

IGNORE = -100


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ---------------- 500M (custom template, no BOS) ----------------
def render_custom(messages, tok, max_len):
    """Returns (input_ids, labels). Loss only on the assistant content + its <|eos|>."""
    eos = tok.convert_tokens_to_ids("<|eos|>")
    ids, labels = [], []
    for m in messages:
        role = m["role"]
        head = tok(f"<|{role}|>\n", add_special_tokens=False)["input_ids"]
        body = tok(m["content"], add_special_tokens=False)["input_ids"]
        ids += head + body + [eos]
        if role == "assistant":
            labels += [IGNORE] * len(head) + body + [eos]      # supervise answer + eos
        else:
            labels += [IGNORE] * (len(head) + len(body) + 1)
        ids.append(tok.convert_tokens_to_ids("\n") if "\n" in tok.get_vocab() else eos)
        labels.append(IGNORE)
    return ids[:max_len], labels[:max_len]


# ---------------- Gemma (built-in chat template, system -> user) ----------------
def _to_gemma_conv(messages):
    sys = next((m["content"] for m in messages if m["role"] == "system"), "")
    conv = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user" and sys:
            conv.append({"role": "user", "content": f"{sys}\n\n{m['content']}"}); sys = ""
        else:
            conv.append({"role": "model" if m["role"] == "assistant" else m["role"],
                         "content": m["content"]})
    return conv


def render_gemma(messages, tok, max_len):
    """Mask everything up to and including the final '<start_of_turn>model\\n'."""
    conv = _to_gemma_conv(messages)
    full = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    # prompt = everything except the model's answer; find the last model turn opener
    marker = "<start_of_turn>model\n"
    cut = full.rfind(marker) + len(marker)
    prompt_text, answer_text = full[:cut], full[cut:]
    p = tok(prompt_text, add_special_tokens=False)["input_ids"]
    a = tok(answer_text, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:max_len]
    labels = ([IGNORE] * len(p) + a)[:max_len]
    return ids, labels


class ChatDataset(Dataset):
    def __init__(self, rows, tok, renderer, max_len):
        self.ex = []
        dropped = 0
        for r in rows:
            ids, labels = renderer(r["messages"], tok, max_len)
            if len(ids) < 4 or all(l == IGNORE for l in labels):
                dropped += 1
                continue
            self.ex.append((ids, labels))
        self.trunc = sum(1 for r in rows
                         if len(renderer(r["messages"], tok, 10 ** 9)[0]) > max_len)
        self.dropped = dropped

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, labels = self.ex[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = m - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [IGNORE] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}
