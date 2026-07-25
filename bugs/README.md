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

## Phase 3 — local serving (2026-07-25)

Issues hit while standing up `finetune/serve_api.py` — one endpoint serving all 13 versions on a
single RTX 3060 (12 GB) — and wiring it to the 13 model sites plus the arena.

| # | Issue | Kind | Where |
| --- | --- | --- | --- |
| [17](17-zero-byte-weights-passed-availability.md) | 0-byte `model.safetensors` passed the availability check | robustness | download · `serve_api.py` |
| [18](18-eviction-during-generation-freed-nothing.md) | LRU eviction during generation freed no VRAM, over-committed the GPU | concurrency | `serve_api.py` |
| [19](19-native-oom-kills-process-uncatchable.md) | 4-bit load OOM killed the process; `except OutOfMemoryError` never fired | infra / memory | `serve_api.py` |
| [20](20-gemma-variants-duplicated-the-base.md) | Each Gemma variant loaded a private copy of the same base (~15 GB) | memory | `serve_api.py` |
| [21](21-missing-attention-mask-pad-equals-eos.md) | No `attention_mask` where pad and eos share a token id | correctness | `serve_api.py` |
| [22](22-launch-json-cwd-cannot-cross-drives.md) | Dev-server `cwd` cannot cross drives (C: → D:) | environment | `.claude/launch.json` |

Bug 15 also **recurred** here (same `modal volume get` collapse, different call site) — its entry
now carries the recurrence and the sturdier `mkdir -p` form.

The through-line of this phase is that **exit codes and bookkeeping both lied**. 17 downloaded
"successfully" while the weights were empty; 18's pool reported 2 resident models while 3 sat on
the card; 19 didn't even leave a traceback. Each was caught only by measuring the physical thing —
bytes on disk, `torch.cuda.memory_allocated()` before/after an eviction, VRAM per load — rather
than trusting the layer above it. 19 in particular reframed the concurrency work: a *reactive*
`try/except` is worthless against a native abort, so admission has to be decided before the
allocation, not repaired after it.

Together 19 + 20 moved the ceiling from "dies at 11 models" to **all 13 resident in 6.55 GB**,
verified under a 20-thread / 60-request load with 0 failures and 0 evictions.
