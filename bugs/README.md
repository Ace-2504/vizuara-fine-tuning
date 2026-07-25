# Bugs & Issues

One file per issue hit on this project. Each records: why it arose, how it was fixed, what
alternatives existed, and why those were not chosen.

## Phase 1 — dataset build (2026-07-21)

Issues hit while building the QA-SFT + RAFT datasets.

| # | Issue | Kind | Where |
| --- | --- | --- | --- |
| [01](01-domain-interleave-closure.md) | All chunks came from one source (100% fineweb-edu) | logic bug | `gen.py` |
| [02](02-orphaned-background-process.md) | Full run died silently after ~250 calls | process/launch | shell |
| [03](03-stale-hf-token-401.md) | Public embedder download 401'd | environment | `config.py` |
| [04](04-grounding-filter-paraphrastic.md) | Summaries/rewrites deleted as "ungrounded" | logic bug | `build.py` |
| [05](05-eval-quarantine-doc-level.md) | Eval holdout ate ~9,400 training pairs | logic bug | `build.py` |
| [06](06-embedding-dedup-aux-collapse.md) | Aux tasks collapsed to a handful | logic bug | `build.py` |
| [07](07-self-reference-filter-aux.md) | Self-ref filter deleted valid aux | logic bug | `build.py` |
| [08](08-torn-jsonl-lines.md) | 27 corrupt lines crashed the reader | robustness | `build.py` |
| [09](09-cost-estimate-error.md) | Cost under-estimated ~6× ($5-6 vs ~$25-30) | estimation | analysis |
| [10](10-raft-duplicate-documents.md) | 15.3% of RAFT prompts had duplicate documents | logic bug | `build.py` |
| [11](11-answerable-absent-topic-mismatch.md) | RAFT "golden-present-but-absent" was off-topic | logic bug | `build.py` |

Bugs 01–03 were caught during validation before the full run. Bugs 04–07 were caught by reading
the assembly output counts (aux tasks vanishing, train pool shrinking) rather than crashes —
they would have silently produced a skewed dataset. Bug 09 had real consequences: the $5 balance
depleted after generation, before the judge stage. Bugs 10–11 were caught by **inspecting actual
dataset examples** — neither crashed nor showed in aggregate counts; only reading the rows
revealed them. Both are assembly-only fixes (no re-generation, no API cost).

## Phase 2 — evaluation & serving (2026-07-23 → 07-24)

Issues hit while running the set1/set2 evaluation (`evaluations/`), finishing it locally after the
cloud budget ran out, and publishing the results.

| # | Issue | Kind | Where |
| --- | --- | --- | --- |
| [12](12-gemma-eval-too-slow-for-timeout.md) | Gemma-2B eval blew the 2 h per-call timeout (4-bit → A100 capacity → bf16 → local) | perf / infra | `evaluations/eval.py` |
| [13](13-modal-spend-limit-mid-run.md) | Modal spend limit stopped the run mid-flight; finished locally on an RTX 3060 | infra / billing | Modal · `eval_local.py` |
| [14](14-fewshot-base-context-overflow.md) | Few-shot base prompts overflowed the 1,024-token context | correctness | `evaluations/eval.py` |
| [15](15-modal-volume-get-dir-to-file.md) | `modal volume get` collapsed a directory into one file | environment | download |
| [16](16-published-report-markup-leaks.md) | Published report leaked raw `**` and RAFT `##begin_quote##` markers | rendering | `evaluations/publish_reports.py` |

Phase-2 bugs 12–13 were infrastructure, not code: the evaluation logic was correct, but the
cheapest/most-available way to *run* it kept shifting (L4 → A100 → bf16 → local GPU) as timeouts,
capacity, and finally the spend limit forced the hand. 14 was caught by reading the first log lines
(the "exceed max length 1024" warning), 15–16 only by **looking at the actual files/rendered page**,
not at exit codes.
