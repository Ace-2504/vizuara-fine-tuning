# 02 — Full generation died silently after ~250 calls

**Symptom.** The full generation run reported "completed, exit code 0" almost immediately, having
made only ~250 of 23,093 calls (635 raw pairs). No error, no budget message — it just stopped.

## Why it arose

It was launched as a shell-backgrounded command **and** through the harness's own
background mechanism at the same time:

```bash
python gen.py > data/sft/gen_run.log 2>&1 &     # + run_in_background=true
```

The trailing `&` backgrounds the Python process *inside the bash subshell* and makes the bash
command return immediately with exit 0. The harness saw the command "complete," and when the
transient subshell exited it took the orphaned child Python process with it. The job was killed,
not finished.

## How it was fixed

Relaunched **without** the shell `&`, letting the harness own and track the actual Python process
(and notify on its real completion hours later):

```bash
python gen.py 2>&1 | tee data/sft/gen_run.log     # run_in_background=true, no &
```

Because generation is resumable via `gen_state.json`, the ~250 already-done chunks were skipped;
nothing was lost.

## Alternatives considered

- **`nohup python gen.py & disown`** — detach from the shell explicitly.
- **Windows detached process** — `start /b` / a `CREATE_NEW_PROCESS_GROUP` spawn.
- **Run in the foreground** — block until it finishes.

## Why they were not chosen

- **`nohup`/`disown` and `start /b`** both create a process the harness does not own — it can't
  track it, stream its output, or notify on completion, and a second background layer is exactly
  what caused the bug. Fighting the harness's backgrounding with a second mechanism is the
  anti-pattern here.
- **Foreground** would block the session for the multi-hour run, defeating the point of doing
  other work (writing `build.py`) while it ran.

Letting the harness manage the single long-running process is the supported path: one owner, one
completion signal, plus the resumability that made the botched first launch harmless.
