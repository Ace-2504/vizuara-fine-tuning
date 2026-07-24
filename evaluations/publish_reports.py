"""Render REPORT_SET1.md / REPORT_SET2.md into a public, general-audience static site
with charts, for Vercel. The LOCAL .md files are left untouched — this only rewrites
wording and adds visuals for the published version.

    python evaluations/publish_reports.py            # -> report_site/{index,set1,set2}.html
Then deploy report_site/ with the Vercel CLI.
"""
from __future__ import annotations

import html
import os
import re
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from stats import mean_ci                       # noqa: E402
from experiments import EXPERIMENTS, META       # noqa: E402

import json  # noqa: E402

RESULTS = os.path.join(ROOT, "eval_results")
SRC = {"set1": os.path.join(HERE, "REPORT_SET1.md"), "set2": os.path.join(HERE, "REPORT_SET2.md")}
OUT = os.path.join(ROOT, "report_site")
TITLES = {"set1": "Set 1 · base vs SFT vs RAFT", "set2": "Set 2 · DPO vs RLAIF"}
FAM_COLOR = {"125m": "#f5b342", "500m": "#5b93ff", "gemma": "#38d9c4"}

# General-audience rewording — applied ONLY to the published copy. Order matters.
REWRITES = [
    (r"\s*\(caveat \d+\)", ""),                                    # strip internal caveat refs
    (r"caveat \d+", "a known limitation"),
    ("Lexical-F1 vs judge disagreement", "Where word-overlap scoring disagrees with the AI judge"),
    ("circular — omitted", "omitted — this model was trained on this reward"),
    ("token-F1", "word-overlap"),
    ("matches-ref", "matches reference"),
    ("false-abstain↓", "wrongly refused↓"),
    ("false-abstain", "wrongly refused"),
    ("abstain rate", "refusal rate"),
    ("correct abstention", "correct refusal"),
    ("over-abstains", "wrongly refuses"),
    ("retrieval_failure (abstain expected)", "answer is absent (correct action: refuse)"),
    ("closed_book (parametric recall — contamination-sensitive)", "no documents given (memory only)"),
    ("realistic (with distractors)", "with distractor documents"),
    ("Reproducibility", "Consistency check"),
]

INTRO = {
    "set1": ("<b>What this measures.</b> Each model is shown one or more documents and a question, "
             "and must answer <i>using only those documents</i>. An independent AI judge (Gemini), "
             "blind to which model wrote each answer, rates its correctness — shown here on a 0–1 "
             "scale. The three versions of each model: <b>base</b> = not fine-tuned; <b>SFT</b> = "
             "taught to answer from documents; <b>RAFT</b> = also taught to say “not in the "
             "documents” when the answer is missing. Bars show judged correctness with a 95% "
             "confidence range; the ◆ marker is the older automatic “word-overlap” score, shown "
             "to highlight where it misleads."),
    "set2": ("<b>What this measures.</b> Each model is shown documents and a question and must "
             "answer <i>using only those documents</i>. An independent AI judge (Gemini), blind to "
             "which model wrote each answer, rates its correctness on a 0–1 scale. Every model here "
             "was first taught to answer (SFT), then <b>aligned</b> to human-style preferences two "
             "different ways: <b>DPO</b> (learns directly from preferred-vs-rejected answers) and "
             "<b>RLAIF</b> (learns via a reward model). Bars show judged correctness with a 95% "
             "confidence range; the ◆ marker is the older “word-overlap” score."),
}

CSS = """
:root{--bg:#0b0e14;--card:#12161f;--fg:#dfe5ee;--muted:#8b95a7;--accent:#f5b342;
--border:#232a38;--head:#1a2130}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:28px 20px 80px}
nav{position:sticky;top:0;background:rgba(11,14,20,.92);backdrop-filter:blur(6px);
border-bottom:1px solid var(--border);padding:12px 20px;margin:-28px -20px 24px;z-index:5}
nav a{color:var(--muted);text-decoration:none;margin-right:18px;font-size:13.5px;font-weight:500}
nav a:hover,nav a.on{color:var(--accent)}
h1{font-size:26px;line-height:1.25;margin:.3em 0 .5em}
h2{font-size:19px;margin:1.8em 0 .6em;padding-top:.4em;border-top:1px solid var(--border)}
h3{font-size:15.5px;color:var(--accent);margin:1.5em 0 .5em;font-weight:600}
p,li{color:var(--fg)}em{color:var(--muted);font-style:normal}strong{color:#fff}
code{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:#0e1220;
border:1px solid var(--border);border-radius:4px;padding:1px 5px;color:var(--accent)}
ul{padding-left:20px}li{margin:5px 0}
hr{border:0;border-top:1px solid var(--border);margin:1.5em 0}
.tw{overflow-x:auto;margin:12px 0;border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
th{background:var(--head);color:#c7cfdd;font-weight:600}
tr:last-child td{border-bottom:0}
td:not(:first-child){font-family:ui-monospace,Menlo,monospace;color:#c7cfdd}
tbody tr:hover td{background:rgba(245,179,66,.04)}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 22px;margin:18px 0}
.intro{background:#0e1220;border-left:3px solid var(--accent);border-radius:8px;
padding:14px 18px;margin:16px 0;font-size:14px;line-height:1.7;color:#cdd5e2}
.chart{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin:18px 0}
.cap{color:var(--muted);font-size:12px;margin-top:8px}
.legend{font-size:11.5px;fill:var(--muted)}
.foot{color:var(--muted);font-size:12px;margin-top:40px;border-top:1px solid var(--border);padding-top:14px}
a{color:var(--accent)}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--card:#fff;--fg:#1a1f2b;--muted:#5b6478;
--border:#e2e6ee;--head:#eef1f6}code{background:#eef1f6}}
"""


def esc(s):
    return html.escape(str(s))


# ---------- data for charts ----------
def clean_vals(version, metric):
    """List of per-item clean-condition values for a metric (judged 0-1, or token_f1)."""
    path = os.path.join(RESULTS, f"{version}.judged.json")
    if not os.path.exists(path):
        path = os.path.join(RESULTS, f"{version}.json")
    if not os.path.exists(path):
        return []
    blob = json.load(open(path, encoding="utf-8"))
    out = []
    for it in blob.get("per_item", []):
        if it.get("cond") != "clean":
            continue
        if metric == "judged":
            j = it.get("judge")
            if j:
                out.append((float(j.get("correct", 0)) - 1.0) / 4.0)
        else:
            out.append(it.get("scores", {}).get(metric, 0.0))
    return out


def series(name):
    rows = []
    for v in EXPERIMENTS[name]:
        jv = clean_vals(v, "judged")
        if not jv:
            continue
        pt, lo, hi = mean_ci(jv)
        f1 = sum(clean_vals(v, "token_f1")) / max(1, len(clean_vals(v, "token_f1")))
        rows.append({"label": META[v]["label"], "fam": META[v]["family"],
                     "v": pt, "lo": lo, "hi": hi, "f1": f1})
    return rows


# ---------- SVG charts ----------
def hbar_chart(rows, title, note):
    W, rowh, padL, padR, padT, padB = 760, 34, 200, 66, 52, 20
    H = padT + padB + rowh * len(rows)
    def X(v):
        return padL + (W - padL - padR) * v
    p = [f'<text x="14" y="24" font-size="15" font-weight="600" fill="var(--fg)">{esc(title)}</text>']
    # legend
    p.append(f'<rect x="{W-250}" y="12" width="16" height="10" rx="2" fill="#8b95a7"/>'
             f'<text class="legend" x="{W-230}" y="21">AI-judged correctness</text>'
             f'<path d="M{W-96} 12 l6 6 -6 6 -6 -6 z" fill="var(--fg)"/>'
             f'<text class="legend" x="{W-84}" y="21">word-overlap</text>')
    for g in (0, .25, .5, .75, 1.0):
        gx = X(g)
        p.append(f'<line x1="{gx:.1f}" y1="{padT-4}" x2="{gx:.1f}" y2="{H-padB}" stroke="var(--border)"/>')
        p.append(f'<text x="{gx:.1f}" y="{padT-8}" text-anchor="middle" class="legend">{g:.2f}</text>')
    for i, r in enumerate(rows):
        y = padT + i * rowh + rowh / 2
        p.append(f'<text x="{padL-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="var(--fg)">{esc(r["label"])}</text>')
        p.append(f'<rect x="{padL}" y="{y-9:.1f}" width="{max(0,X(r["v"])-padL):.1f}" height="18" rx="3" fill="{FAM_COLOR.get(r["fam"],"#888")}"/>')
        p.append(f'<line x1="{X(r["lo"]):.1f}" y1="{y:.1f}" x2="{X(r["hi"]):.1f}" y2="{y:.1f}" stroke="var(--fg)" stroke-width="1.4" opacity=".55"/>')
        fx = X(r["f1"])                                    # word-overlap marker (diamond)
        p.append(f'<path d="M{fx:.1f} {y-5:.1f} l5 5 -5 5 -5 -5 z" fill="var(--fg)" opacity=".85"/>')
        p.append(f'<text x="{X(r["v"])+8:.1f}" y="{y+4:.1f}" font-size="11.5" font-family="monospace" fill="var(--fg)">{r["v"]:.3f}</text>')
    svg = f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">{"".join(p)}</svg>'
    return f'<div class="chart">{svg}<div class=cap>{esc(note)}</div></div>'


def delta_chart(name):
    """set2 only: DPO vs RLAIF difference per family (RLAIF − DPO)."""
    fams = [("125m", "SLM-125M"), ("500m", "SLM-500M"), ("gemma", "Gemma-2B")]
    rows = []
    for fam, lab in fams:
        dpo = [v for v in EXPERIMENTS[name] if META[v]["family"] == fam and v.endswith("-dpo")]
        rl = [v for v in EXPERIMENTS[name] if META[v]["family"] == fam and v.endswith("-rlaif")]
        if not dpo or not rl:
            continue
        d = sum(clean_vals(dpo[0], "judged")) / max(1, len(clean_vals(dpo[0], "judged")))
        r = sum(clean_vals(rl[0], "judged")) / max(1, len(clean_vals(rl[0], "judged")))
        rows.append((lab, r - d))
    if not rows:
        return ""
    W, rowh, padL, padT, padB = 760, 40, 120, 54, 26
    mid = padL + (W - padL - 30) / 2
    H = padT + padB + rowh * len(rows)
    span = (W - padL - 30) / 2
    m = max(abs(d) for _, d in rows) or 0.3
    p = [f'<text x="14" y="24" font-size="15" font-weight="600" fill="var(--fg)">Which alignment method wins, by model size</text>',
         f'<line x1="{mid}" y1="{padT-4}" x2="{mid}" y2="{H-padB}" stroke="var(--border)"/>',
         f'<text class="legend" x="{padL}" y="{padT-8}">← DPO better</text>',
         f'<text class="legend" x="{W-30}" y="{padT-8}" text-anchor="end">RLAIF better →</text>']
    for i, (lab, d) in enumerate(rows):
        y = padT + i * rowh + rowh / 2
        w = span * (d / m) * 0.92
        col = "#38d9c4" if d > 0 else "#e2663b"
        p.append(f'<text x="{padL-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="var(--fg)">{esc(lab)}</text>')
        x0 = mid if d >= 0 else mid + w
        p.append(f'<rect x="{x0:.1f}" y="{y-9:.1f}" width="{abs(w):.1f}" height="18" rx="3" fill="{col}"/>')
        tx = mid + w + (8 if d >= 0 else -8)
        anc = "start" if d >= 0 else "end"
        p.append(f'<text x="{tx:.1f}" y="{y+4:.1f}" text-anchor="{anc}" font-size="11.5" font-family="monospace" fill="var(--fg)">{d:+.3f}</text>')
    svg = f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">{"".join(p)}</svg>'
    return (f'<div class="chart">{svg}<div class=cap>Gap in AI-judged correctness between the two '
            f'alignment methods for each model family. The winner flips with model size.</div></div>')


# ---------- markdown -> cleaned html ----------
def md_to_html(name):
    raw = open(SRC[name], encoding="utf-8").read()
    raw = "\n".join("" if re.fullmatch(r"=+\s*", ln) else ln for ln in raw.splitlines())
    for pat, rep in REWRITES:
        raw = re.sub(pat, rep, raw) if pat.startswith(("\\", "(", "[")) or "\\" in pat else raw.replace(pat, rep)
    html_body = markdown.markdown(raw, extensions=["tables", "sane_lists", "fenced_code"])
    html_body = re.sub(r"(<table>.*?</table>)", r'<div class="tw">\1</div>', html_body, flags=re.S)
    return html_body


def page(body, active, title):
    nav = "".join(f'<a href="{href}"{" class=on" if active==k else ""}>{lab}</a>'
                  for k, href, lab in [("index", "/", "Overview"), ("set1", "/set1", "Set 1"),
                                       ("set2", "/set2", "Set 2")])
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><div class=wrap>"
            f"<nav>{nav}</nav>{body}"
            f"<div class=foot>Grounded question-answering evaluation · 2,000 held-out questions · "
            f"independent AI judge with 95% confidence ranges and paired significance testing · "
            f"two separate experiments, not compared against each other.</div></div></body></html>")


def report_page(name):
    rows = series(name)
    title_h1 = f"<h1>{'Set 1 — base vs SFT vs RAFT' if name=='set1' else 'Set 2 — alignment: DPO vs RLAIF'}</h1>"
    intro = f'<div class="intro">{INTRO[name]}</div>'
    chart = hbar_chart(rows, "AI-judged correctness by model",
                       "Longer bar = more often correct. Thin line = 95% confidence range. "
                       "The ◆ shows the older word-overlap score — where it sits far left of the "
                       "bar, that automatic metric badly under-rates a model that is actually correct.")
    extra = delta_chart(name) if name == "set2" else ""
    body = md_to_html(name)
    # drop the now-duplicate h1 that came from the markdown (we render our own)
    body = re.sub(r"^<h1>.*?</h1>", "", body, count=1, flags=re.S)
    return title_h1 + intro + chart + extra + body


def index_body():
    return (
        "<h1>Small-language-model fine-tuning — evaluation</h1>"
        "<p><em>Two independent experiments on grounded question-answering over US legal and "
        "financial text. Every score is from an independent AI judge, with confidence ranges and "
        "significance testing. The two experiments are separate — no numbers are compared across "
        "them.</em></p>"
        "<div class=card><h3><a href='/set1'>Set 1 — base vs SFT vs RAFT</a></h3>"
        "<p>Does fine-tuning a small model help? <strong>Model size dominates</strong> — the 2B "
        "model is near-perfect while the 125M lags far behind. Teaching the small model to answer "
        "(SFT) helps, but teaching it to refuse when unsure (RAFT) <strong>backfires</strong> — it "
        "starts refusing questions it could answer.</p></div>"
        "<div class=card><h3><a href='/set2'>Set 2 — alignment: DPO vs RLAIF</a></h3>"
        "<p>Two ways of aligning a model to preferences. <strong>The better method flips with "
        "size</strong> — RLAIF wins for the small models, DPO for the 2B model. One method even "
        "<strong>broke the smallest model</strong>, collapsing it to two-word answers.</p></div>")


def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(
        page(index_body(), "index", "SLM fine-tuning evaluation"))
    for name in ("set1", "set2"):
        open(os.path.join(OUT, f"{name}.html"), "w", encoding="utf-8").write(
            page(report_page(name), name, TITLES[name]))
    open(os.path.join(OUT, "vercel.json"), "w", encoding="utf-8").write('{\n  "cleanUrls": true\n}\n')
    print("wrote", OUT, "->", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
