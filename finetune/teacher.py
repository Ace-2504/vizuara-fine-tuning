"""Gemini teacher client — resilient, resumable, budget-aware.

Design goals for THIS build (a $5-at-a-time budget on a possibly-flaky key):
- Try a primary model, fall back to a cheaper/verified one on hard model errors or
  repeated 503s (full gemini-3.1-flash 503'd on this key before; -lite is verified).
- Retry transient blips (429 rate / 5xx); skip a single empty or unparseable response
  rather than aborting a multi-hour chain.
- Raise **BudgetExhausted** (clean, catchable) when the balance/quota is spent, so the
  driver checkpoints and exits — a re-run then RESUMES from disk, never restarts.
- Track exact token usage per model for costing.

Standalone: depends only on google-genai + python-dotenv.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class BudgetExhausted(Exception):
    """Balance/quota spent and backoff cannot clear it. Driver should checkpoint + exit."""


@dataclass
class Usage:
    requests: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    retries: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, in_t: int, out_t: int) -> None:
        self.requests += 1
        self.in_tokens += in_t or 0
        self.out_tokens += out_t or 0
        m = self.by_model.setdefault(model, {"req": 0, "in": 0, "out": 0})
        m["req"] += 1; m["in"] += in_t or 0; m["out"] += out_t or 0


@dataclass
class TeacherClient:
    # Primary first; on a hard model error or repeated 503 we advance down the list.
    models: tuple[str, ...] = ("gemini-3.1-flash", "gemini-3.1-flash-lite")
    min_interval_s: float = 0.5          # ~120 rpm; raise if the tier is slower
    max_retries: int = 5
    env_path: Path = Path(__file__).resolve().parent / ".env"
    usage: Usage = field(default_factory=Usage)
    _idx: int = 0
    _last_call: float = 0.0

    def __post_init__(self) -> None:
        load_dotenv(self.env_path)
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise SystemExit(f"GEMINI_API_KEY not found in {self.env_path}")
        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=key)

    @property
    def model(self) -> str:
        return self.models[self._idx]

    def _advance_model(self, why: str) -> bool:
        if self._idx < len(self.models) - 1:
            dead = self.models[self._idx]
            self._idx += 1
            print(f"  [teacher] {dead} -> {self.model} ({why})", flush=True)
            return True
        return False

    def _pace(self) -> None:
        wait = self.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def generate_json(self, prompt: str, schema, *, temperature: float = 0.9,
                      thinking: bool = False):
        """Parsed JSON (list/dict), or None if this single call was skipped.
        Raises BudgetExhausted on a persistent quota/billing wall."""
        import json as _json
        from google.genai import types

        cfg = dict(response_mime_type="application/json", response_schema=schema,
                   temperature=temperature)
        if not thinking:
            try:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        config = types.GenerateContentConfig(**cfg)

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                r = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config)
                self._last_call = time.time()
                um = getattr(r, "usage_metadata", None)
                self.usage.add(self.model,
                               getattr(um, "prompt_token_count", 0) if um else 0,
                               getattr(um, "candidates_token_count", 0) if um else 0)
                text = getattr(r, "text", None)
                if not text:
                    if attempt < self.max_retries:
                        self.usage.retries += 1
                        time.sleep(delay + random.uniform(0, delay)); delay = min(delay*2, 60)
                        continue
                    print("  [teacher] empty response after retries — skipping call", flush=True)
                    return None
                return _json.loads(text)
            except _json.JSONDecodeError:
                self._last_call = time.time()
                if attempt < self.max_retries:
                    self.usage.retries += 1
                    time.sleep(delay + random.uniform(0, delay)); delay = min(delay*2, 60)
                    continue
                print("  [teacher] unparseable JSON after retries — skipping call", flush=True)
                return None
            except Exception as e:  # noqa: BLE001 — classify by status/message
                self._last_call = time.time()
                status, msg = _status_of(e), str(e).lower()
                billing = ("resource_exhausted" in msg or "quota" in msg or "billing" in msg
                           or "insufficient" in msg or "exceeded your current quota" in msg)
                # Hard model error (model not available / no access on this key): fall back.
                if status in (404, 403) or "not_found" in msg or "not found" in msg:
                    if self._advance_model(f"status {status}"):
                        delay = 2.0; continue
                    raise
                # Persistent 503 on the active model: fall back to the next model.
                if status == 503 and attempt >= 2 and self._advance_model("503"):
                    delay = 2.0; continue
                transient = status in (429, 500, 502, 503, 504)
                if transient and attempt < self.max_retries and not billing:
                    self.usage.retries += 1
                    time.sleep(delay + random.uniform(0, delay)); delay = min(delay*2, 60)
                    continue
                # Out of retries on 429, or an explicit billing/quota error -> clean halt.
                if status == 429 or billing:
                    raise BudgetExhausted(str(e)[:200]) from e
                raise
        raise BudgetExhausted("retries exhausted")


def _status_of(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    s = str(exc)
    for code in (429, 500, 502, 503, 504, 404, 401, 403):
        if str(code) in s:
            return code
    return None
