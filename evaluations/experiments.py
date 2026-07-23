"""The two evaluation experiments — set1 and set2 — as first-class objects.

These are TWO SEPARATE experiments. set1 compares set1 models against each other;
set2 compares set2 models against each other. There is deliberately NO comparison
across the two sets — `eval_report.py` builds one independent pairwise matrix per set.

Each version's metadata drives the report:
  - family        : which base model it descends from (for grouping / colour)
  - is_base       : an untuned zero-point (cannot follow the chat template; a floor)
  - is_raft       : trained on RAFT data -> gets the four-condition breakdown
  - reward_circular: its alignment optimised the eval's reward model, so the reward
                     metric is NOT an independent signal for it (RLAIF). Excluded from
                     any reward-based headline; F1 / judge remain valid.
  - sft_parent    : the SFT checkpoint it was aligned from (context only; the parent
                    lives in the other set, so it is never pulled into this set's matrix).

Version keys match `eval.py`'s ALL_VERSIONS exactly.
"""
from __future__ import annotations

# ---- set 1: base vs SFT vs RAFT, on the 125M and Gemma families --------------------
SET1 = [
    "base-125m",
    "slm-125m-sft",
    "slm-125m-raft",
    "base-gemma",
    "gemma-2-2b-sft",
    "gemma-2-2b-raft",
]

# ---- set 2: the alignment experiment — DPO vs RLAIF, across families ----------------
SET2 = [
    "base-500m",
    "slm-125m-sft-dpo",
    "slm-125m-sft-rlaif",
    "slm-500m-sft-dpo",
    "slm-500m-sft-rlaif",
    "gemma-2-2b-sft-dpo",
    "gemma-2-2b-sft-rlaif",
]

EXPERIMENTS = {"set1": SET1, "set2": SET2}

# Per-version metadata. `family` also fixes a stable display order within a set.
META = {
    # set 1
    "base-125m":          {"family": "125m",  "is_base": True,  "is_raft": False, "reward_circular": False, "sft_parent": None,          "label": "SLM-125M base"},
    "slm-125m-sft":       {"family": "125m",  "is_base": False, "is_raft": False, "reward_circular": False, "sft_parent": None,          "label": "SLM-125M SFT"},
    "slm-125m-raft":      {"family": "125m",  "is_base": False, "is_raft": True,  "reward_circular": False, "sft_parent": None,          "label": "SLM-125M RAFT"},
    "base-gemma":         {"family": "gemma", "is_base": True,  "is_raft": False, "reward_circular": False, "sft_parent": None,          "label": "Gemma-2B base"},
    "gemma-2-2b-sft":     {"family": "gemma", "is_base": False, "is_raft": False, "reward_circular": False, "sft_parent": None,          "label": "Gemma-2B SFT"},
    "gemma-2-2b-raft":    {"family": "gemma", "is_base": False, "is_raft": True,  "reward_circular": False, "sft_parent": None,          "label": "Gemma-2B RAFT"},
    # set 2
    "base-500m":          {"family": "500m",  "is_base": True,  "is_raft": False, "reward_circular": False, "sft_parent": None,             "label": "SLM-500M base"},
    "slm-125m-sft-dpo":   {"family": "125m",  "is_base": False, "is_raft": False, "reward_circular": False, "sft_parent": "slm-125m-sft",   "label": "SLM-125M SFT+DPO"},
    "slm-125m-sft-rlaif": {"family": "125m",  "is_base": False, "is_raft": False, "reward_circular": True,  "sft_parent": "slm-125m-sft",   "label": "SLM-125M SFT+RLAIF"},
    "slm-500m-sft-dpo":   {"family": "500m",  "is_base": False, "is_raft": False, "reward_circular": False, "sft_parent": "slm-500m-sft",   "label": "SLM-500M SFT+DPO"},
    "slm-500m-sft-rlaif": {"family": "500m",  "is_base": False, "is_raft": False, "reward_circular": True,  "sft_parent": "slm-500m-sft",   "label": "SLM-500M SFT+RLAIF"},
    "gemma-2-2b-sft-dpo":   {"family": "gemma", "is_base": False, "is_raft": False, "reward_circular": False, "sft_parent": "gemma-2-2b-sft", "label": "Gemma-2B SFT+DPO"},
    "gemma-2-2b-sft-rlaif": {"family": "gemma", "is_base": False, "is_raft": False, "reward_circular": True,  "sft_parent": "gemma-2-2b-sft", "label": "Gemma-2B SFT+RLAIF"},
}

# SFT parents referenced by set2 but not members of any experiment. eval.py can still
# score them (they are needed as alignment references anyway); the report only uses them
# if a future reference-line option is enabled — never inside a set's own matrix.
REFERENCE_ONLY = ["slm-500m-sft"]


def versions_for(experiment: str) -> list[str]:
    if experiment not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {experiment!r}; choose from {list(EXPERIMENTS)}")
    return list(EXPERIMENTS[experiment])


def experiment_of(version: str) -> str | None:
    for name, members in EXPERIMENTS.items():
        if version in members:
            return name
    return None
