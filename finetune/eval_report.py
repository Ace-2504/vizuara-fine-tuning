"""Aggregate eval.py's per-version results into a comparison table.

    # 1) download results from the volume (after `modal run eval.py --all`):
    modal volume get ft-data /eval ./eval_results
    # 2) build the report:
    python eval_report.py ./eval_results

Prints, per base model, base -> SFT -> RAFT -> DPO -> RLAIF with clean-condition token-F1,
reward, and fabrication (each with 95% CI). A difference whose CI overlaps the base is flagged
'not resolved'. Also prints the RAFT grounding gaps.
"""
from __future__ import annotations
import glob, json, os, sys

GROUPS = {"125m": "slm-125m", "500m": "slm-500m", "gemma": "gemma-2-2b"}
ORDER = ["base", "sft", "raft", "sft-dpo", "sft-rlaif"]


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        j = json.load(open(f, encoding="utf-8"))
        r = j.get("result", j)
        out[r["version"]] = r
    return out


def ci(triple):
    return f"{triple[0]:.3f} [{triple[1]:.3f},{triple[2]:.3f}]"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "./eval_results"
    res = load(d)
    if not res:
        print(f"no eval json found in {d}"); return

    for gkey, prefix in GROUPS.items():
        base_name = f"base-{gkey}"
        rows = []
        for variant in ORDER:
            name = base_name if variant == "base" else f"{prefix}-{variant}"
            if name in res:
                rows.append((variant, res[name]))
        if not rows:
            continue
        print(f"\n### {prefix}")
        base = dict(rows).get("base")
        base_f1 = base["conditions"]["clean"]["token_f1"] if base else None
        print(f"{'variant':10} {'clean token-F1 [95% CI]':30} {'reward':22} {'fabrication':20} vs-base")
        for variant, r in rows:
            c = r["conditions"]["clean"]
            verdict = ""
            if base_f1 and variant != "base":
                # resolved if this F1's CI-low > base F1's CI-high (or vice versa)
                if c["token_f1"][1] > base_f1[2]:
                    verdict = "better ✓"
                elif c["token_f1"][2] < base_f1[1]:
                    verdict = "worse ✗"
                else:
                    verdict = "not resolved"
            print(f"{variant:10} {ci(c['token_f1']):30} {ci(c['reward']):22} "
                  f"{ci(c['fabrication']):20} {verdict}")
            if r.get("is_raft"):
                print(f"           RAFT: grounding_gap {r['grounding_gap']:+.3f} | "
                      f"distractor_gap {r['distractor_gap']:+.3f} | "
                      f"correct_abstention {r['correct_abstention']:.3f}")

    print("\nNotes: token-F1/reward higher is better; fabrication lower is better. "
          "'reward' is the shared reward model's score (also a DPO/RLAIF win signal). "
          "A variant is only 'better'/'worse' if its 95% CI does not overlap the base's.")


if __name__ == "__main__":
    main()
