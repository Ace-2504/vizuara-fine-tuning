"""Shared statistics for the eval report — paired bootstrap done right.

The old report compared two models by eye: "their 95% CIs overlap -> not resolved".
Overlapping per-model CIs is NOT a significance test — it is both too conservative
(can miss real differences) and, when people read a gap as significant because the
intervals *don't* touch, subtly wrong about the error rate. The correct object is the
CI on the PAIRED DIFFERENCE, resampling the same eval items for both models with one
shared set of indices, so per-item difficulty cancels.

    mean_ci(values)                 -> (point, lo, hi)          one model, one metric
    paired_delta_ci(a_by_key, b_by_key) -> {...}               model A minus model B

Both take plain Python; numpy is used internally. Deterministic given `seed`.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 5000
SEED = 0


def mean_ci(values, n: int = N_BOOT, seed: int = SEED):
    """Point estimate + 95% percentile bootstrap CI for one metric on one model."""
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(a, (n, a.size), replace=True).mean(axis=1)
    return (float(a.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def paired_delta_ci(a_by_key: dict, b_by_key: dict, n: int = N_BOOT, seed: int = SEED):
    """95% CI on (mean(A) - mean(B)) over the items both models were scored on.

    a_by_key / b_by_key map an item key (e.g. (cond, pair_id)) -> scalar score. Only
    the intersection of keys is used, and the SAME bootstrap indices resample both
    models, so the difference is paired. 'significant' == the delta CI excludes 0.
    """
    keys = sorted(set(a_by_key) & set(b_by_key))
    if not keys:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "significant": False,
                "mean_a": 0.0, "mean_b": 0.0}
    a = np.array([a_by_key[k] for k in keys], dtype=float)
    b = np.array([b_by_key[k] for k in keys], dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, (n, diff.size))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(diff.size),
        "significant": bool(lo > 0 or hi < 0),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
    }


def paired_gap_ci(minuend_by_key: dict, subtrahend_by_key: dict, n: int = N_BOOT, seed: int = SEED):
    """A within-model condition gap (e.g. realistic F1 minus closed_book F1), paired on
    pair_id across the two conditions. Same machinery as paired_delta_ci, named for the
    RAFT gap use-case. Returns a CI so 'grounding_gap' is no longer a bare point estimate.
    """
    return paired_delta_ci(minuend_by_key, subtrahend_by_key, n=n, seed=seed)
