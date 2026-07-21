# 07 — Self-reference filter deleted valid aux examples

**Symptom.** After fixes 04–06, aux tasks recovered only to ~1,384 total — still far below the
~11,700 in the raw data. Something else was still deleting them.

## Why it arose

`rule_ok` dropped any pair whose question+answer contained a self-referential phrase:

```python
if any(p in low for p in ("the passage", "the text", "the document", "as stated above",
                          "here are", "provided text")):
    return False
```

Applied to **all** tasks. That rule is correct for **QA**: a closed-book answer that says "as
stated in the document" is broken, because at training time the passage is absent — the referent
does not exist. But for the aux tasks the passage **is** in the prompt, so an answer like *"The
document outlines three risk factors…"* is perfectly valid. The filter deleted most faithful
summaries and rewrites for using ordinary referential language.

## How it was fixed

Scope the self-reference filter to `task == "qa"`, and keep only the task-agnostic template-echo
drop ("here are") for everything:

```python
if "here are" in low:            # template echo — drop for any task
    return False
if r["task"] == "qa":
    if any(p in low for p in ("the passage", "the text", "the document", "as stated above")):
        return False
    ...quote checks...
```

After this, `rule_ok` survival went from 21,098 → 31,047, aux recovered fully, and the assembly
hit the 15,000 QA-SFT target with a balanced task mix.

## Alternatives considered

- **Drop the self-reference filter entirely.**
- **Strip the offending phrases** from answers instead of dropping the pair.
- **Whitelist** specific phrases as acceptable for aux.

## Why they were not chosen

- **Dropping it entirely** loses a genuine QA quality guard — a closed-book QA answer that refers
  to an absent passage is a real defect that would teach the model to hallucinate a referent.
- **Stripping phrases** rewrites the teacher's output, which risks corrupting meaning ("as stated
  above, revenue rose" → "revenue rose" can change scope) and hides quality problems instead of
  surfacing them.
- **A whitelist** is fragile and grows without end.

Scoping the rule to the task where it is actually valid (QA) is precise: it keeps the guarantee
where it matters and stops punishing aux for normal language.
