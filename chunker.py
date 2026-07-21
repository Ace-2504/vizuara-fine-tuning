"""Corpus -> passages. One document per line; chunk with the project tokenizer."""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache

import config as C


@dataclass
class Chunk:
    chunk_id: str
    text: str
    token_len: int
    source: str
    doc_id: str          # f"{source}/{shard}:{line_no}"


@lru_cache(maxsize=1)
def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(C.TOKENIZER))


def shards(source: str) -> list[str]:
    d = C.CORPUS / source
    return sorted(f for f in os.listdir(d) if f.endswith(".txt"))


def chunks_of_shard(source: str, shard: str):
    """Yield Chunks from one shard, deterministically."""
    tok = _tok()
    path = C.CORPUS / source / shard
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            text = line.rstrip("\n")
            if not text:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            doc_id = f"{source}/{shard}:{line_no}"
            for n, i in enumerate(range(0, len(ids), C.CHUNK_TOKENS)):
                if n >= C.MAX_CHUNKS_PER_DOC:
                    break
                piece = ids[i:i + C.CHUNK_TOKENS]
                if len(piece) < C.MIN_CHUNK_TOKENS:
                    continue
                body = tok.decode(piece)
                cid = hashlib.sha256(body.encode()).hexdigest()[:16]
                yield Chunk(cid, body, len(piece), source, doc_id)


def token_len(text: str) -> int:
    return len(_tok()(text, add_special_tokens=False)["input_ids"])
