# 22 — `launch.json` dev-server entries failed because `cwd` cannot cross drives

**Symptom.** Per-site dev-server entries pointing at the frontend repo were rejected before
anything started:

```
Failed to start server: cwd must be a relative path within the project root.
```

## Why it arose

The launch config lives with the pipeline repo on **C:**, while the frontends live on **D:**:

```
project root : C:\Users\…\Replicate-the-125M-SLM-Data-Pipeline
frontends    : D:\slm-frontends
```

`cwd` must be **relative to the project root**, and on Windows there is no relative path from a
location on `C:` to one on `D:` — separate drives are separate roots, so no amount of `..`
segments connects them. An absolute `"cwd": "D:\\slm-frontends"` is rejected by the same rule.

This is why the sibling `e4-site` entry works: `../pre-training-v1-extended/web-v1-extended` is a
genuine relative path *within the same drive*.

## How it was fixed

Stop using `cwd` and let the package manager take the path instead — `npm --prefix` accepts an
absolute path and is unaffected by the working directory:

```json
{
  "name": "site-gemma-qa",
  "runtimeExecutable": "npm",
  "runtimeArgs": ["--prefix", "D:/slm-frontends", "run", "dev", "--", "-p", "3020"],
  "env": { "NEXT_PUBLIC_MODEL_ID": "gemma-qa" },
  "port": 3020
}
```

Forward slashes are used so the JSON needs no escaping. One entry per model (ports 3011–3023)
plus the arena (3030), each pinning its model through `env`.

## Alternatives considered

- **Move the frontends onto C:.** C: had ~38 GB free against D:'s ~361 GB, and the checkpoints
  live on D: — not worth relocating a working layout for a config limitation.
- **A directory junction from C: into D:** (`mklink /J`). Would satisfy the relative-path rule,
  but adds an invisible filesystem indirection that anyone reading the repo would have to
  discover.
- **A wrapper script per site that `cd`s first.** 13 extra shell files to maintain for something
  one flag solves.
