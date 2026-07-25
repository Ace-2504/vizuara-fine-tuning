# 23 — Parallel judge calls all 404'd racing the model fallback

**Symptom.** The arena's live `/judge` endpoint graded only **1 of 4** answers. The survivor scored
correctly; the rest came back with:

```
ClientError: 404 NOT_FOUND. {'error': {'code': 404, 'message':
'models/gemini-3.1-flash is not found for API version v1beta,
 or is not supported for generateContent.'}}
```

Judged **one at a time** the same answers scored fine (10.0 / 0.0). Only the parallel path failed,
and the number that succeeded was always exactly one.

## Why it arose

Two things combined.

**1. `gemini-3.1-flash` is not available on this key.** Not a billing problem — listing the models
the key can see returns 56 entries including `gemini-3.1-flash-lite`, `-lite-preview`, `-image`,
`-tts-preview`, `-live-preview`, but **no plain text `gemini-3.1-flash`**. So the primary model in
`TeacherClient.models` can never serve `generateContent` here; every run must fall back.

**2. The fallback is discovered lazily, and only by failing.** `TeacherClient` holds an ordered
tuple and advances an index when a call returns 404/403:

```python
models: tuple[str, ...] = ("gemini-3.1-flash", "gemini-3.1-flash-lite")

@property
def model(self):
    return self.models[self._idx]        # advances only after a request fails
```

Constructing the client makes **no request**, so `_idx` is still 0 when the first batch fans out.
The judge dispatched all four answers at once:

```python
with ThreadPoolExecutor(max_workers=6) as ex:
    scored = dict(ex.map(one, list(r.answers.items())))
```

All four threads read `self.model` → `gemini-3.1-flash` → all four 404. One thread's failure
advanced `_idx` and its internal retry then succeeded on `-lite`; the other three had already
consumed their retry budget on the dead model. Hence exactly one survivor.

The offline harness never hit this because `judge_eval.py` is a **sequential** loop — the very
first call pays the 404, the fallback sticks, and every later call uses `-lite`.

## How it was fixed

Resolve the model **serially** before fanning out — one real call absorbs the fallback, then the
rest parallelise against a settled `_idx`:

```python
teacher = _get_teacher()            # construct once (no request yet)
items = list(r.answers.items())
if items:
    mid, res = one(items[0])        # SERIAL: this call discovers the fallback
    scored[mid] = res
    if len(items) > 1:
        with ThreadPoolExecutor(max_workers=6) as ex:
            scored.update(dict(ex.map(one, items[1:])))
```

A single retry was also added inside `one()` for transient empty/unparseable responses. Result:
**4/4 graded**, and the judge discriminates correctly — a right answer 10.0, a wrong one 0.0,
gibberish 0.0.

## Was it a credits problem?

**No.** Three independent signs:

- The error is `404 NOT_FOUND` naming the model. Quota exhaustion surfaces as
  **429 `RESOURCE_EXHAUSTED`**, not 404.
- The model list proves plain `gemini-3.1-flash` is absent from the key's catalogue entirely.
- Calls on `gemini-3.1-flash-lite` succeeded immediately before and after, returning real scores —
  a keyed, funded client.

## Cost note (why the fallback is harmless)

Measured usage for one judge call: **125 input / 36 output tokens** (a realistic arena call with a
120-token candidate is ~280 in / ~40 out). Against the project's assumed rate table in `smoke.py`:

| Model | $/1M in | $/1M out | 1,300 judge calls |
| --- | --- | --- | --- |
| `gemini-3.1-flash` | 0.30 | 1.20 | ~$0.17 |
| `gemini-3.1-flash-lite` | 0.10 | 0.40 | ~$0.06 |

So Flash is roughly **3× Flash-Lite**, and the whole difference over 1,300 calls is a few cents.
⚠️ Those rates are the repo's *assumed* figures and are explicitly flagged "VERIFY against current
pricing" — [bug 09](09-cost-estimate-error.md) found the real blended Flash-Lite cost to be
**~$1 / 1M tokens**, ~6× the assumption. On that empirical figure 1,300 calls (~0.42M tokens) is
**~$0.42 on Flash-Lite** and roughly **$1.25 on Flash** if the 3× card ratio holds. Either way the
choice is worth well under a dollar per full arena sweep — and moot here, since Flash is not
offered on this key.

## Follow-up — the arena judge is now pinned

The concurrency fix made the fallback *safe*, but every server still burned one 404 to discover it.
The arena judge is now pinned to Flash-Lite at construction, so the dead model is never called:

```python
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-flash-lite")
_teacher = TeacherClient(models=(JUDGE_MODEL,))
```

This also keeps the live arena on **the same judge the offline harness used**, so live scores stay
comparable with the frozen evaluation rather than drifting because a different model graded them.
Verified: 4/4 graded and **zero `[teacher]` fallback lines** in the server log. The serial-first
call is deliberately kept — it costs nothing and still protects the parallel path if anyone
re-introduces a multi-model list or overrides `JUDGE_MODEL`.

`teacher.py`'s shared default tuple is left untouched, since the dataset-build and offline-judge
paths use it on keys that may legitimately have Flash.

## Alternatives considered

- **Drop `gemini-3.1-flash` from the shared tuple in `teacher.py`.** Removes the 404 for every
  caller — but that tuple is shared with the dataset-build and offline-judge paths, and a key on a
  different tier may well have Flash. Pinning per call-site (above) is the narrower change.
- **Probe the model list on construction** (`client.models.list()`). One extra API round-trip on
  every server start, and it still would not settle `_idx` without changing `TeacherClient`.
- **A lock around the first call.** Equivalent to the serial-first approach but more machinery for
  the same effect.

## Lesson

A lazily-discovered, failure-driven fallback is safe under sequential use and quietly broken under
concurrency: every parallel caller races the same stale state and burns its retry budget on it.
Resolve shared client state **once, serially**, before fanning out.
