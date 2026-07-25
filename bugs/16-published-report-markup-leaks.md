# 16 — Published report leaked raw markdown and RAFT quote-markers

**Symptom.** Two rendering defects on the public Vercel report site (not the local `.md` reports):

1. The **Key-findings bullets showed literal `**`** instead of bold — e.g.
   `**Best model:** Gemma-2B RAFT — 0.975 …` rendered with the asterisks visible.
2. The **example answers contained `##begin_quote## … ##end_quote##`** markers (and mid-word
   truncations), e.g. *"##begin_quote## One type of surfactant comes from cows … ##end_quote## Cows
   and synthetic materials."*

## Why it arose

1. `key_findings()` returns bullet strings written in **markdown** (`**bold**`, `*italic*`). The
   publisher built the Key-findings block by dropping those strings **straight into HTML** (`<li>{b}</li>`)
   without converting inline markdown, so the raw `**` survived.
2. The RAFT-trained models emit `##begin_quote## … ##end_quote##` in their **actual output** — it's
   their trained "quote the source, then answer" format. The examples table printed the raw `resp`
   field verbatim, so the training-format markers showed up on the page.

## How it was fixed

1. Convert inline markdown in each bullet before inserting:
   ```python
   b = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", b)
   b = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", b)
   ```
2. A `clean_response()` that renders `##begin_quote##/##end_quote##` as normal quotation marks,
   collapses whitespace, and truncates on a **word boundary**. The models' genuine quirks
   (repetition, `*Note:` asides, two-word non-answers) are deliberately kept — only the training-format
   markers are cleaned.

## Alternatives considered

- **Render the whole report with a markdown library** instead of building HTML by hand. The site is
  built from data for full layout control (custom charts, dropped columns, example tables), so
  targeted inline-markdown conversion was the smaller change.
- **Strip the quoted span entirely**, showing only the final answer. Rejected: showing the quote (as
  real quotation marks) is honest and reveals *why* RAFT models behave as they do.
