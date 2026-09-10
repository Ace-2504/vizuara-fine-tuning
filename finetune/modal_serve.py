"""Serve all 15 SLM models on Modal — one L4, scale-to-zero, stable public URL.

Wraps the existing serve_api.py (same CATALOG, shared-Gemma backbone + adapters,
/health /generate /judge). Modal supplies the GPU, a persistent weights Volume, and a
permanent https://*.modal.run URL — replacing the RTX 3060 + Cloudflare tunnel.

Deploy (from this finetune/ dir, so serve_api.py + serve_local.py are importable):
    MODAL_PROFILE=singh1621 modal deploy modal_serve.py

Layout on the Volume:
    /weights/checkpoints/<name>/<name>/...   (the 12 local fine-tunes; double-nested as on disk)
    /weights/hf-cache/...                    (HF downloads for the 3 base models, persisted)
"""
import os
import modal

app = modal.App("slm-serve")

WEIGHTS = modal.Volume.from_name("slm-weights", create_if_missing=True)
VOL = "/weights"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # EXACT versions from the working local CUDA venv — the checkpoints' tokenizer.json
        # and the gemma 4-bit path only parse/load cleanly on these.
        "torch==2.4.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "accelerate==1.14.0",
        "peft==0.19.1",
        "bitsandbytes==0.50.0",      # 4-bit shared Gemma backbone
        "sentencepiece",
        "fastapi[standard]",
        "pydantic>=2",
        "hf-transfer",               # faster base-model downloads
        "google-genai",              # the /judge Gemini client (teacher.py)
        "python-dotenv",
    )
    .env({
        "CKPT_DIR": f"{VOL}/checkpoints",     # serve_api reads local fine-tunes from here
        "HF_HOME": f"{VOL}/hf-cache",         # persist base-model downloads on the volume
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "MAX_RESIDENT": "15",
        "PYTHONPATH": "/judgelib",            # so `from teacher import ...` resolves in /judge
    })
    .add_local_python_source("serve_api", "serve_local")              # local files LAST
    .add_local_file("teacher.py", "/judgelib/teacher.py")            # the Gemini judge client
)


@app.cls(
    gpu="L4",                          # 24 GB, fits all 15 with headroom
    image=image,
    volumes={VOL: WEIGHTS},
    scaledown_window=300,              # scale to zero 5 min after the last request
    min_containers=0,                  # pure scale-to-zero
    timeout=900,
    secrets=[
        modal.Secret.from_name("hf-token"),      # HF_TOKEN for gated gemma-2-2b-it
        modal.Secret.from_name("gemini-key"),    # GEMINI_API_KEY for the /judge client
    ],
)
@modal.concurrent(max_inputs=20)       # queue requests; generation is GPU-lock serialized
class Serve:
    @modal.enter()
    def start(self):
        """Once per container start: pre-load every model so the first real request is
        fast, then commit the volume so HF base-model downloads persist across restarts."""
        import sys, os
        sys.path.insert(0, "/judgelib")          # make `from teacher import ...` (used by /judge) resolvable
        try:
            print(f"[judge] /judgelib = {os.listdir('/judgelib')}", flush=True)
            import teacher  # noqa: F401
            print("[judge] teacher import OK", flush=True)
        except Exception as e:
            print(f"[judge] teacher import FAILED: {type(e).__name__}: {e}", flush=True)
        import serve_api
        for mid in serve_api.CATALOG:
            try:
                with serve_api.acquire(mid):   # loads + REGISTERS in _resident, then leaves it idle
                    pass
                print(f"[warm] {mid}", flush=True)
            except Exception as e:
                print(f"[warm-fail] {mid}: {e}", flush=True)
        try:
            WEIGHTS.commit()           # persist newly-downloaded base weights into hf-cache
        except Exception as e:
            print(f"[commit-skip] {e}", flush=True)
        self.fastapi = serve_api.app

    @modal.asgi_app()
    def web(self):
        return self.fastapi
