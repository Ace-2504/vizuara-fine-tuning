# 21 — No `attention_mask` passed, on models where pad and eos share a token id

**Symptom.** Every generation on the custom (125M / 500M) models emitted:

```
The attention mask is not set and cannot be inferred from input because pad token is same as
eos token. As a consequence, you may observe unexpected behavior.
```

Output still looked plausible, so it was easy to dismiss as noise — which is precisely why it
was worth fixing before wiring 13 frontends to it.

## Why it arose

Only `input_ids` were taken from the tokenizer; the mask was discarded:

```python
ids = tok(full, return_tensors="pt", add_special_tokens=...).input_ids.to(DEV)
out = model.generate(ids, ...)          # no attention_mask
```

With no mask, `generate()` infers one by treating `pad_token_id` positions as padding. On the
custom models `<|eos|>` **is** the pad token (`pad_token_id or eos` is also what gets passed
through), so any legitimate end-of-turn token inside the prompt — the `<|eos|>` that terminates
the system and user turns in the `<|role|>` template — is indistinguishable from padding:

```
<|system|>\n…<|eos|>\n<|user|>\n…<|eos|>\n<|assistant|>\n
```

Every one of those separators risked being masked out, silently degrading the model's view of its
own prompt. The template makes this *systematic* rather than an edge case.

## How it was fixed

Keep the encoding and pass its mask explicitly:

```python
enc  = tok(full, return_tensors="pt", add_special_tokens=(family == "gemma" and kind != "completion"))
ids  = enc.input_ids.to(DEV)
attn = enc.attention_mask.to(DEV)   # explicit: pad and eos share an id on the custom models
out  = model.generate(ids, attention_mask=attn, ...)
```

The tokenizer's own mask is all-ones for a single unpadded sequence, which is the correct
intent — no real padding exists here.

## Alternatives considered

- **Give the custom models a distinct `<|pad|>` id.** Cleaner in principle, and `<|pad|>` exists
  in the vocabulary — but changing pad semantics at serving time diverges from how the models
  were trained and fine-tuned. Not worth the risk for a serving-side fix.
- **Suppress the warning.** Would have hidden a real correctness issue.

## Lesson

A warning that says "you may observe unexpected behavior" and still produces readable output is
the most dangerous kind: nothing crashes, and the damage is a quietly degraded prompt.
