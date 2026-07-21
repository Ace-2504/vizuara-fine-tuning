# 01 — All chunks came from one source (late-binding closure)

**Symptom.** The first validation run produced **254 pairs, 100% `fineweb-edu`**, instead of the
intended `sec` / `case-law` / `fineweb-edu` mix.

## Why it arose

`gen.py::iter_chunks` built one chunk-generator per source with a dict comprehension:

```python
gens = {s: (chunks_of_shard(s, sh) for sh in shards(s)) for s in C.SOURCES}
```

The inner **generator expression** `(chunks_of_shard(s, sh) for sh in shards(s))` captures `s`
as a *free variable*, resolved lazily when the generator is consumed — not when it is created.
By the time any generator was iterated, the dict comprehension had finished and `s` was bound to
its final value, `"fineweb-edu"`. So all three "per-source" generators called
`chunks_of_shard("fineweb-edu", …)` and read the same shards. (The outer `shards(s)` iterable
*is* evaluated eagerly with the right `s`, which is why it failed silently rather than erroring —
the shard *names* were per-source, only the `chunks_of_shard` source argument was wrong.)

This is the classic Python late-binding closure trap.

## How it was fixed

Replaced the generator expression with a named function whose `source` is a **bound argument**
(arguments bind at call time, eliminating late binding):

```python
def _source_chunks(source):
    for sh in shards(source):
        yield from chunks_of_shard(source, sh)

active = {s: _source_chunks(s) for s in C.SOURCES}
```

Re-validated: `sec 51 / case-law 44 / fineweb-edu 35` — the target mix. Only ~$0.03 of skewed
data was generated; it was cleared and regenerated.

## Alternatives considered

- **Default-argument capture** — `(chunks_of_shard(s2, sh) for s2 in [s] for sh in shards(s2))`
  or a `lambda s=s:` binding trick.
- **Eager flat list** — materialize `[(source, shard) for source in SOURCES for shard in
  shards(source)]` up front and index into it.
- **`functools.partial`** — `partial(chunks_of_shard, s)` per source.

## Why they were not chosen

- The **default-arg trick works but is exactly the kind of subtle idiom that invites this same
  bug back on the next edit** — a maintainer who "simplifies" it reintroduces late binding. A
  named function makes the binding obvious and self-documenting.
- An **eager flat list** would work but throws away the streaming property and complicates the
  weighted per-source round-robin (which needs one live cursor per source). The named generator
  keeps per-source state trivial.
- **`partial`** is heavier and less readable than a three-line generator, for no benefit.
