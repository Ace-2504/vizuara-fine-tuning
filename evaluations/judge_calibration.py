"""Calibrate the LLM judge against human labels (caveat 1).

The judge is the headline metric, so its trustworthiness must be measured, not assumed.
This samples a small set of already-judged items, lets a HUMAN label the same items
blind to the judge, and reports judge-vs-human agreement. Do this once; if agreement is
poor, the judged numbers should be treated with caution (or the judge prompt revised).

    # 1) emit a labelling sheet (after judge_eval.py has produced *.judged.json):
    python evaluations/judge_calibration.py emit ./eval_results --n 50 --out calib.csv
    # 2) open calib.csv, fill human_correct (1-5) and human_grounded (0/1) for each row,
    #    WITHOUT looking at the judge_* columns (hide them). Save.
    # 3) score the agreement:
    python evaluations/judge_calibration.py score calib.csv

Reports: exact & within-1 agreement and Pearson r on correctness (1-5), and agreement +
Cohen's kappa on groundedness (bool). No external deps beyond the stdlib.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys


def _load_judged(results_dir: str):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "*.judged.json")):
        blob = json.load(open(path, encoding="utf-8"))
        version = blob.get("result", {}).get("version", os.path.basename(path))
        for it in blob.get("per_item", []):
            if it.get("judge"):
                rows.append((version, it))
    return rows


def emit(a):
    rows = _load_judged(a.results_dir)
    if not rows:
        sys.exit(f"no *.judged.json with judge scores in {a.results_dir} — run judge_eval.py first.")
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    picked = rows[: a.n]
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "version", "cond", "context_and_question", "reference", "candidate",
                    "judge_correct", "judge_grounded", "human_correct", "human_grounded"])
        for i, (version, it) in enumerate(picked):
            j = it["judge"]
            w.writerow([i, version, it["cond"], it["user"], it["ref"], it["resp"],
                        j.get("correct", ""), int(bool(j.get("grounded"))), "", ""])
    print(f"wrote {a.out} with {len(picked)} items to label.\n"
          f"Fill human_correct (1-5) and human_grounded (0/1) WITHOUT reading the judge_* "
          f"columns, then: python {os.path.basename(__file__)} score {a.out}", flush=True)


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def _cohen_kappa(a_labels, b_labels):
    """Binary Cohen's kappa."""
    n = len(a_labels)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a_labels, b_labels)) / n
    pa1 = sum(a_labels) / n
    pb1 = sum(b_labels) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def score(a):
    with open(a.sheet, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("human_correct", "").strip() != "" and r.get("human_grounded", "").strip() != ""]
    if not rows:
        sys.exit("no labelled rows found — fill human_correct and human_grounded first.")
    jc = [int(float(r["judge_correct"])) for r in rows]
    hc = [int(float(r["human_correct"])) for r in rows]
    jg = [int(float(r["judge_grounded"])) for r in rows]
    hg = [int(float(r["human_grounded"])) for r in rows]
    n = len(rows)

    exact = sum(a == b for a, b in zip(jc, hc)) / n
    within1 = sum(abs(a - b) <= 1 for a, b in zip(jc, hc)) / n
    r = _pearson([float(x) for x in jc], [float(x) for x in hc])
    # binarised correctness (correct if >=4), the way the report thresholds it
    jc_bin = [int(x >= 4) for x in jc]
    hc_bin = [int(x >= 4) for x in hc]
    corr_bin_agree = sum(a == b for a, b in zip(jc_bin, hc_bin)) / n
    corr_kappa = _cohen_kappa(jc_bin, hc_bin)
    g_agree = sum(a == b for a, b in zip(jg, hg)) / n
    g_kappa = _cohen_kappa(jg, hg)

    print(f"\njudge-vs-human calibration on {n} items\n{'-'*44}")
    print("correctness (1-5):")
    print(f"  exact agreement      {exact:.3f}")
    print(f"  within-1 agreement   {within1:.3f}")
    print(f"  Pearson r            {r:.3f}")
    print(f"  binary(>=4) agree    {corr_bin_agree:.3f}   Cohen kappa {corr_kappa:.3f}")
    print("groundedness (bool):")
    print(f"  agreement            {g_agree:.3f}   Cohen kappa {g_kappa:.3f}")
    print("\nrule of thumb: kappa >0.6 substantial, 0.4-0.6 moderate, <0.4 weak. Within-1 "
          "agreement >0.8 on the 1-5 scale is a usable judge; below that, treat judged "
          "numbers cautiously and consider revising the judge prompt.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="write a human-labelling CSV")
    e.add_argument("results_dir", nargs="?", default="./eval_results")
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--out", default="calib.csv")
    e.set_defaults(func=emit)
    s = sub.add_parser("score", help="score a filled labelling CSV")
    s.add_argument("sheet")
    s.set_defaults(func=score)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
