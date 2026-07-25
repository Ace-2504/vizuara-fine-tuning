# 17 — A 0-byte `model.safetensors` passed the availability check

**Symptom.** After pulling the 12 fine-tuned checkpoints off the Modal volume, every directory
looked healthy — 6 files each, `config.json` present, `available()` returning `True` — but
`slm-500m-sft-rlaif` totalled **2 MB** instead of ~990 MB. Its `model.safetensors` was
**0 bytes**. Nothing failed at download time and nothing failed at check time; the model would
only have exploded at first inference, on the one site nobody had opened yet.

## Why it arose

Two independent weaknesses lined up.

1. **The download reported success.** `modal volume get` exited 0 and printed
   `✓ Finished downloading files to local!` while leaving the largest file empty. The transfer of
   the ~1 GB weights file failed silently; the small JSON/tokenizer files all arrived.

2. **The health check only looked for a config file.** `available()` asked "does this directory
   contain `config.json` or `adapter_config.json`?" — a *presence* test with no notion of size:

```python
def available(mid: str) -> bool:
    d = model_dir(src)
    return os.path.exists(os.path.join(d, "config.json")) or \
           os.path.exists(os.path.join(d, "adapter_config.json"))
```

Config files are ~1 KB and had transferred fine, so the broken checkpoint looked identical to a
good one. A per-directory file *count* would not have caught it either — the count was 6, same as
every healthy checkpoint.

## How it was fixed

Caught by **summing actual file sizes** rather than trusting exit codes or file presence — a
sweep printing `files=N  <MB> MB` per checkpoint made the outlier obvious at a glance:

```
slm-500m-sft-dpo         files=6     990 MB  config=True
slm-500m-sft-rlaif       files=6       2 MB  config=True   <-- weights are 0 bytes
```

The directory was deleted and re-pulled, recovering the full 990 MB. Note the re-download also had
to wait on a **file handle held by the dead downloader** (`Device or resource busy` on Windows)
— the stale `python.exe` had to be killed before the path could be removed.

## Alternatives considered

- **Verify the safetensors header** (read the 8-byte length prefix + JSON header and compare the
  declared tensor bytes against file size). Strongest check and cheap, but overkill for a
  one-machine setup; a size floor catches the realistic failure (truncated/empty transfer).
- **Checksum against the remote.** Modal exposes no per-file hash through `volume get`, so this
  would mean re-downloading to compare — self-defeating.
- **Let it fail at load time.** Rejected: the error would surface as a confusing
  `safetensors` parse error on whichever frontend was opened first, long after the cause.

## Lesson

An availability check that tests for *presence* is not a check for *integrity*. Where a download
can partially succeed, assert on **bytes**, not on file existence or exit status.
