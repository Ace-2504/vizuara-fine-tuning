# 24 — Model output rendered markdown literally (second occurrence)

**Symptom.** Answers in the demo and arena boxes printed their own formatting as characters:

```
**Merger:**
* **Definition:** A merger occurs when two or more companies combine...
* **Control Structure:** Both merging entities maintain separate identities...
```

Gemma writes light markdown by habit, so nearly every one of its answers was affected. The RAFT
fine-tunes had the same class of problem with their `##begin_quote## … ##end_quote##` evidence
markers.

## Why it arose — and why it came back

The output box interpolated the raw string:

```tsx
{err ? `Request failed: ${err}` : out || c.empty}
```

React escapes text, which is correct for safety, but nothing ever *interpreted* the markup. Every
`**`, `*` and `##begin_quote##` reached the reader as literal characters.

This is the **second time** this class of bug shipped. [Bug 16](16-published-report-markup-leaks.md)
was the same failure in the published evaluation report, and it was fixed **there and only there** —
by cleaning the strings at that one call site. The underlying fact was never addressed:

> **Model output is lightly-formatted text, not plain text.** Any surface that displays it must
> render that formatting.

Patching one surface left every future surface exposed. When the live demo and the arena were
built months later, they reproduced the bug exactly.

## How it was fixed

A single shared renderer, `components/ModelText.tsx`, now owns the problem, and every surface that
displays model output uses it:

| Surface | File |
| --- | --- |
| Model-site demo box | `slm-frontends/components/Demo.tsx` |
| Arena per-model answers | `slm-arena/components/ArenaLive.tsx` |

It handles what these models actually emit — `**bold**`, `__bold__`, `*italic*`, `` `code` ``,
`*`/`-` bullet lines — and turns the RAFT `##begin_quote## … ##end_quote##` pair into a real
quotation block instead of stray tokens.

Two deliberate constraints:

- **No markdown library.** The subset these models produce is tiny; a dependency would be more
  surface area than the feature.
- **No `dangerouslySetInnerHTML`.** The renderer builds React elements, so untrusted model output
  can never inject markup. This matters more than usual: the text is generated, not authored.

## Alternatives considered

- **Strip the markers.** One regex, and it was tempting — but `**Definition:**` carries emphasis the
  author meant, and deleting it loses information. Rendering keeps it.
- **Render markdown at the server**, in `serve_api`. Wrong layer: the API returns what the model
  said, and presentation belongs to whatever displays it. It would also have to be undone for any
  non-HTML consumer.
- **Fix only the arena** (where it was reported). Exactly the mistake that caused the recurrence.

## Lesson

When a bug is a property of the *data* rather than of one screen, fixing the screen guarantees a
recurrence. Bug 16 fixed a symptom; this entry fixes the cause, and the fix lives in one component
so the next surface inherits it for free.
