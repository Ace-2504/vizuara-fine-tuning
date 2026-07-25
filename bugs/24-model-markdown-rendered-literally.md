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
`*`/`-` bullet lines — and turns the RAFT evidence markers into a real quotation block instead of
stray tokens. Getting those markers right took a second pass; see below.

Two deliberate constraints:

- **No markdown library.** The subset these models produce is tiny; a dependency would be more
  surface area than the feature.
- **No `dangerouslySetInnerHTML`.** The renderer builds React elements, so untrusted model output
  can never inject markup. This matters more than usual: the text is generated, not authored.

## The second lesson: don't enumerate variants

The first version of the renderer matched the markers exactly — `##begin_quote##` and
`##end_quote##`. Testing it against live output immediately turned up debris it did not catch,
because the sampler mangles the token constantly:

```
##end_ quote##        ##end_quotechloro##       ##end_of_quote###       ##batchmode##
```

Each fix-the-variant round caught one form and missed the next. `##batchmode##` is not even a
quote marker — it is a hallucinated control token of the same shape.

The working approach was to stop enumerating and **match the shape once, then classify**:

```ts
const HASH_MARKER   = /#{2,}(?![#\s])[^#\n]{0,40}#{1,4}/g;
const HASH_TRAILING = /#{2,}(?![#\s])[^#\n]{0,40}$/gm;   // closing hashes dropped by the model

const classify = (m: string) => (/quote/i.test(m) ? "␟" : "");   // ␟ toggles an evidence block
```

Anything of that shape mentioning "quote" toggles a quotation; any other `##…##` is sampler noise
and is dropped. `##…##` never occurs in real prose, so the shape itself is the signal.

`(?![#\s])` is load-bearing: it forces the whole run of hashes to be consumed and requires a
non-space next character, which is exactly what separates a control token from a markdown
`## Heading`. Without it, `#{2,}` backtracks — matching `##` of `###` — and eats real headings.

Verified against the shipped source (the regexes are extracted from the file and run, rather than
retyped into a test, after an earlier edit silently failed to apply):

| input | output |
| --- | --- |
| `##begin_quote## e ##end_quote##` | quotation around `e` |
| `##end_quotechloro##` / `##end_ quote##` / `##end_of_quote###` | quote boundary |
| `##batchmode##` | removed |
| `text ##end_quote` | quote boundary |
| `## Heading` / `### Heading` | **unchanged** |

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
