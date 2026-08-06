"""Evaluate ONE seeded checkpoint with full parity to the original eval + rubric-10 judge.

    python evaluations/eval_seed.py --seeded slm-500m-sft-seed1 --template slm-500m-sft \
           --profile aceaynon2504 --family custom

Steps: (1) download the checkpoint from the profile's ft-data volume (mkdir first — the
volume-get concat bug needs an existing dir), (2) generate greedily on the SAME 500 clean
items as the template model (identical system/user/ref/pair_id/source), with eval.py-parity
render per family, (3) token_f1 + fabrication, (4) rubric-10 judge via the local server,
(5) save eval_results/seed-evals/<seeded>.judged.json + append a summary row.

Nothing is published.
"""
from __future__ import annotations
import argparse, json, io, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "finetune"))
ER = os.path.join(ROOT, "eval_results")
SEED_DIR = os.path.join(ER, "seed-evals")
DL_ROOT = os.path.join(ROOT, "checkpoints", "seeded")   # on D:, plenty of space
BASE = "http://127.0.0.1:8000"
ABSTAIN = "not stated in the context"
SYS_FALLBACK = "You are a precise legal and financial assistant. Answer clearly using the provided context; do not invent facts."


def _words(s): return "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower()).split()
def token_f1(pred, ref):
    p, r = _words(pred), _words(ref)
    if not p or not r: return 0.0
    common = {}
    for w in p: common[w] = common.get(w, 0) + 1
    ov = sum(min(c, r.count(w)) for w, c in common.items())
    if not ov: return 0.0
    pr, rc = ov/len(p), ov/len(r); return 2*pr*rc/(pr+rc)
def _nn(s): return s.replace(",", "").rstrip("%").rstrip(".")
def fabrication(pred, ctx):
    cn = _nn(ctx.replace(",", ""))
    np_ = {_nn(x) for x in re.findall(r"\d[\d,\.]*", pred)}
    nc = {_nn(x) for x in re.findall(r"\d[\d,\.]*", ctx)}
    return float(len([x for x in np_ if x and x not in nc and x not in cn]) > 0)


def _has_ckpt(d):
    return os.path.isdir(d) and (os.path.exists(os.path.join(d, "config.json"))
                                 or os.path.exists(os.path.join(d, "adapter_config.json")))

def _resolve(dst, seeded):
    for c in (dst, os.path.join(dst, seeded), os.path.join(dst, "checkpoints", seeded)):
        if _has_ckpt(c):
            return c
    return None

def _safetensors_ok(d):
    """Integrity check: every .safetensors is fully written (file >= 8 + header + tensor bytes).
    Catches truncated/corrupt downloads that otherwise produce a gibberish model."""
    import glob, struct
    fs = glob.glob(os.path.join(d, "*.safetensors"))
    if not fs:   # LoRA adapters ship adapter_model.safetensors/.bin
        return (os.path.exists(os.path.join(d, "adapter_model.safetensors"))
                or os.path.exists(os.path.join(d, "adapter_model.bin")))
    for f in fs:
        sz = os.path.getsize(f)
        if sz < 16:
            return False
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if sz < 8 + n:
                    return False
                hdr = json.loads(fh.read(n).decode("utf-8"))
            need = 8 + n + max((v["data_offsets"][1] for k, v in hdr.items()
                                if k != "__metadata__" and isinstance(v, dict) and "data_offsets" in v),
                               default=0)
            if sz < need:
                return False
        except Exception:
            return False
    return True

def dl_checkpoint(seeded, profile):
    dst = os.path.join(DL_ROOT, seeded)
    found = _resolve(dst, seeded)
    if found and _safetensors_ok(found):
        print(f"  [dl] {seeded} already present + intact at {found} -> skip", flush=True)
        return found
    os.makedirs(dst, exist_ok=True)     # MUST exist first (volume-get concat bug)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MSYS_NO_PATHCONV": "1", "MODAL_PROFILE": profile}
    cmd = [sys.executable, "-m", "modal", "volume", "get", "--force", "ft-data",
           f"/checkpoints/{seeded}", dst]
    print(f"  [dl] {seeded} from {profile} (waiting for full completion)...", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)  # ~25s typical; NEVER kill early
    found = _resolve(dst, seeded)
    if not found or not _safetensors_ok(found):
        print((r.stdout or "")[-400:], (r.stderr or "")[-400:], flush=True)
        raise SystemExit(f"download failed/corrupt for {seeded} (rc={r.returncode})")
    print(f"  [dl] {seeded} downloaded + integrity-verified", flush=True)
    return found


def render(family, system, user):
    if family == "gemma":
        return None  # gemma uses chat template applied in generate()
    return f"<|system|>\n{system}<|eos|>\n<|user|>\n{user}<|eos|>\n<|assistant|>\n"


def post(path, body, timeout=120):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f: return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeded", required=True)      # e.g. slm-500m-sft-seed1
    ap.add_argument("--template", required=True)     # e.g. slm-500m-sft (original, for the 500 items)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--family", required=True, choices=["custom", "gemma"])
    a = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    import torch
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    from serve_local import load
    os.makedirs(SEED_DIR, exist_ok=True)

    tmpl = json.load(io.open(os.path.join(ER, f"{a.template}.judged.json"), encoding="utf-8"))
    items = [it for it in tmpl["per_item"] if it["cond"] == "clean"]
    print(f"[{a.seeded}] {len(items)} clean items from {a.template}", flush=True)

    ckpt = dl_checkpoint(a.seeded, a.profile)
    print(f"[{a.seeded}] loading {ckpt}", flush=True)
    model, tok, family, dev = load(ckpt)
    eos = tok.convert_tokens_to_ids("<|eos|>") if family == "custom" else tok.eos_token_id

    def gen(system, user):
        if family == "gemma":
            text = tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to(dev)
            gk = {"use_cache": False}
        else:
            ids = tok(render("custom", system, user), return_tensors="pt",
                      add_special_tokens=False).input_ids.to(dev); gk = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tok.pad_token_id or eos, eos_token_id=eos, **gk)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    per_item, t0 = [], time.time()
    for n, it in enumerate(items, 1):
        system = it.get("system") or SYS_FALLBACK
        resp = gen(system, it["user"])
        per_item.append({"cond": "clean", "pair_id": it["pair_id"], "source": it.get("source"),
                         "system": system, "user": it["user"], "ref": it["ref"], "resp": resp,
                         "scores": {"token_f1": token_f1(resp, it["ref"]),
                                    "fabrication": fabrication(resp, it["user"])}})
        if n % 100 == 0:
            print(f"  gen {n}/{len(items)} ({(time.time()-t0)/n:.2f}s/item)", flush=True)
    del model
    torch.cuda.empty_cache()

    # rubric-10 judge via the running server. First call SERIAL (lets TeacherClient resolve any
    # model fallback), then fan out — otherwise parallel calls race the stale model. (known gotcha)
    from concurrent.futures import ThreadPoolExecutor
    print(f"[{a.seeded}] judging {len(per_item)} answers...", flush=True)
    def judge_one(it):
        u = it["user"]; i = u.rfind("Question:")
        q = u[i+len("Question:"):].strip() if i >= 0 else u
        ctx = u[:i].strip() if i >= 0 else ""
        try:
            g = post("/judge", {"question": q, "context": ctx, "reference": it["ref"],
                                "answers": {a.seeded: it["resp"]}})["graded"][a.seeded]
            return {"score": g.get("score"), "parts": g.get("parts"), "grounded": g.get("grounded")}
        except Exception as e:
            return {"error": str(e)[:120]}
    if per_item:
        per_item[0]["judge"] = judge_one(per_item[0])           # warm up serially
        with ThreadPoolExecutor(max_workers=6) as ex:
            for it, res in zip(per_item[1:], ex.map(judge_one, per_item[1:])):
                it["judge"] = res
    vals, gr, bysrc, f1s, fabs = [], [], {}, [], []
    for it in per_item:
        g = it.get("judge") or {}
        if g.get("score") is not None:
            vals.append(g["score"]); gr.append(1 if g.get("grounded") else 0)
            bysrc.setdefault(it["source"], []).append(g["score"])
            f1s.append(it["scores"]["token_f1"]); fabs.append(it["scores"]["fabrication"])

    summary = {"seeded": a.seeded, "template": a.template, "n": len(vals),
               "score": round(sum(vals)/len(vals), 2) if vals else None,
               "grounded": round(sum(gr)/len(gr)*100, 1) if gr else None,
               "by_source": {k: round(sum(v)/len(v), 2) for k, v in sorted(bysrc.items())},
               "token_f1": round(sum(f1s)/len(f1s), 3) if f1s else None,
               "fabrication": round(sum(fabs)/len(fabs)*100, 1) if fabs else None}
    json.dump({"summary": summary, "per_item": per_item},
              io.open(os.path.join(SEED_DIR, f"{a.seeded}.judged.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # append to a running summary table
    sp = os.path.join(SEED_DIR, "seed_eval_summary.jsonl")
    io.open(sp, "a", encoding="utf-8").write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[{a.seeded}] DONE  score {summary['score']}/10  grounded {summary['grounded']}%  "
          f"F1 {summary['token_f1']}  fab {summary['fabrication']}%  -> seed-evals/{a.seeded}.judged.json", flush=True)


if __name__ == "__main__":
    main()
