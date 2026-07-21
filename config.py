"""Single source of truth for the dataset build. All constants live here."""
from __future__ import annotations
import os
from pathlib import Path

# This machine has a stale hf_oauth token that 401s even for public models. Stop hf_hub
# from sending any implicit/cached token so anonymous public downloads (the embedder) work.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "corpus"
TOKENIZER = ROOT / "data" / "tokenizer"
OUT = ROOT / "data" / "sft"
PROV = OUT / "provenance"
RAW = OUT / "raw.jsonl"                 # all raw teacher output (append-only, resumable)
GEN_STATE = OUT / "gen_state.json"      # which (source,shard,chunk,task) are done
JUDGE_CACHE = PROV / "judge.jsonl"      # per-pair judge verdicts (append-only, resumable)

SOURCES = ("sec", "case-law", "fineweb-edu")
DOMAIN_MIX = {"sec": 0.347, "case-law": 0.342, "fineweb-edu": 0.311}

CHUNK_TOKENS = 256
MIN_CHUNK_TOKENS = 80
MAX_CHUNKS_PER_DOC = 12
QA_PER_CHUNK = 5

# --- target KEPT sizes (post-gauntlet) ---
QA_SFT_SIZE = 15_000
RAFT_SIZE = 10_000
EVAL_SIZE = 500
# QA-SFT task mix (of QA_SFT_SIZE)
TASK_MIX = {"qa": 0.40, "summarize": 0.25, "extract": 0.20, "rewrite": 0.15}
CLOSED_BOOK_FRAC = 0.15                 # of qa task; fineweb-edu only
DIFFICULTY_MIX = {"easy": 0.4, "medium": 0.4, "hard": 0.2}
EVOL_FRAC = 0.20                        # of qa evolved to harder
# RAFT
RAFT_K = 2                              # distractors per example (1024-ctx budget)
RAFT_P_GOLDEN = 0.80                    # golden-present fraction
RAFT_ABSTAIN_FRAC = 0.20               # distractors-only (of RAFT_SIZE)
ANSWERABLE_ABSENT_FRAC = 0.10          # golden present but answer absent (of golden-present)
ABSTAIN_STRING = "not stated in the context"
RAFT_DOC_TOKENS = 256

# raw over-generation to survive the gauntlet (~55-60% keep)
KEEP_ASSUMED = 0.55

# --- teacher / judge ---
GEN_MODELS = ("gemini-3.1-flash", "gemini-3.1-flash-lite")   # tries first, falls back
GEN_TEMPERATURE = 0.9
JUDGE_TEMPERATURE = 0.0
JUDGE_KEEP_CORRECT = 4                  # keep if correct >= 4 AND grounded

# --- filters / dedup ---
MIN_ANSWER_CHARS, MAX_ANSWER_CHARS = 8, 1200
MIN_QUESTION_CHARS = 8
GROUNDING_MIN = 0.55                    # answer content-word overlap w/ passage
MAX_QUOTE_TOKENS = 60
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEDUP_COSINE = 0.90
DECONTAM_NGRAM = 13

SEED = 1337
