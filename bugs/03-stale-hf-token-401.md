# 03 — Public embedder download failed with 401

**Symptom.** `build.py`'s dedup stage crashed loading the embedding model:

```
401 Client Error: Unauthorized for url:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json
OAuth token signature verification failed: signature verification failed
```

`all-MiniLM-L6-v2` is a **public** model that needs no authentication.

## Why it arose

This machine has a cached `hf_oauth_*` token that is stale/invalid (noted previously in project
memory). `huggingface_hub` sends that token **implicitly** on every request, even for anonymous
public downloads. The server rejects the bad token with a hard 401 instead of falling back to
unauthenticated access — so a public download that would succeed anonymously fails because a
broken credential is attached to it.

## How it was fixed

In `config.py` (imported by every stage), stop `hf_hub` from sending any implicit/cached token,
and drop the token env vars for the process:

```python
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
```

With no token attached, the public model downloads anonymously. Verified: embedder loads and
encodes on CPU.

## Alternatives considered

- **`huggingface-cli logout`** / delete the cached token file.
- **Pass `token=False`** to `SentenceTransformer(...)`.
- **Refresh the token** with a valid one.
- **Pre-download** the model out-of-band and load from a local path.

## Why they were not chosen

- **`logout`/deleting the token mutates the user's global HF state** — a side effect beyond this
  project. The same machine uses a valid `HF_TOKEN` for the **gated** `google/gemma-2-2b-it`
  download in the fine-tuning stage; clobbering the global login could break that.
- **`token=False`** is not reliably threaded through every nested download that
  `sentence-transformers`/`transformers` make (config, adapter probe, weights), so it fixes some
  calls and not others.
- **Refreshing the token** is unnecessary work for a public model that needs no auth at all.
- **Pre-downloading** is brittle and non-portable across machines/clones.

The env-var disable is **process-scoped, non-destructive, and self-documenting** — it changes
nothing on disk or in the user's global HF login, and it lives in the one file every stage
imports.
