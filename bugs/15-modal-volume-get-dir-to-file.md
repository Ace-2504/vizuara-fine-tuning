# 15 — `modal volume get` collapsed a directory into a single file

**Symptom.** Downloading the Gemma adapters for the local run produced, for each adapter, a single
**83 MB file** named e.g. `gemma-2-2b-sft` (detected as UTF-8 text) instead of a directory
containing `adapter_config.json` + `adapter_model.safetensors` + tokenizer files. The `is_adapter`
check (`os.path.exists(.../adapter_config.json)`) then failed for all four.

## Why it arose

The command was given a **non-existent target *path*** instead of the parent *directory*:

```bash
# wrong: target is a path that doesn't exist -> the dir's files were written into one file
modal volume get ft-data /checkpoints/gemma-2-2b-sft ./migrate/checkpoints/gemma-2-2b-sft
```

`modal volume get <remote_dir> <local>` expects `<local>` to be an **existing directory**, into
which it recreates the remote basename. Given a bare non-existent path it does not create the
directory tree; the multi-file download degenerates into a single output file. (The same command
worked earlier for `/eval` precisely because its target `./eval_results` already existed.)

## How it was fixed

Point the target at the **existing parent directory** and let Modal recreate the subfolder:

```bash
modal volume get ft-data /checkpoints/gemma-2-2b-sft ./migrate/checkpoints   # -> .../gemma-2-2b-sft/
```

After re-downloading, each adapter had its `adapter_config.json` + `adapter_model.safetensors`
(113 MB total) and loaded correctly.

## Alternatives considered

- **Download file-by-file** (`.../adapter_config.json`, `.../adapter_model.safetensors`, …).
  Works, but the parent-dir form is one command per adapter and less error-prone.

## Recurrence (2026-07-25, local serving)

Hit again while pulling the remaining 8 checkpoints for the local inference server — three came
down as single ~240 MB files. The blob is confirmed to be the directory concatenated: it begins
with `tokenizer.json`'s text (`{"version": "1.0", …`) and ends in raw tensor bytes.

Equivalent fix, stated as a rule that is harder to get wrong: **create the destination directory
first**, then pass it as the target.

```bash
mkdir -p checkpoints/slm-125m-sft
modal volume get ft-data checkpoints/slm-125m-sft checkpoints/slm-125m-sft
# -> checkpoints/slm-125m-sft/slm-125m-sft/{config.json,model.safetensors,…}
```

This produces the doubly-nested `x/x/` layout, which `serve_local.model_dir()` already resolves
("handles download double-nesting"). Two related notes from the recurrence:

- **Set `PYTHONIOENCODING=utf-8` on Windows.** Modal's completion banner crashes the console with
  `'charmap' codec can't encode…` / `maps to <undefined>`, which makes a *successful* download
  look like a failure and invites a pointless retry.
- **A silent partial download is possible** even with the correct form — see
  [bug 17](17-zero-byte-weights-passed-availability.md), where the weights file arrived as 0 bytes
  and still passed the availability check.
