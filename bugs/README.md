# Bugs & Issues — dataset build (2026-07-21)

One file per issue hit while building the QA-SFT + RAFT datasets. Each records: why it arose,
how it was fixed, what alternatives existed, and why those were not chosen.

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

Bugs 01–03 were caught during validation before the full run. Bugs 04–07 were caught by reading
the assembly output counts (aux tasks vanishing, train pool shrinking) rather than crashes —
they would have silently produced a skewed dataset. Bug 09 had real consequences: the $5 balance
depleted after generation, before the judge stage.
