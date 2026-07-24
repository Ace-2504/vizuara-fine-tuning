"""Render REPORT_SET1.md / REPORT_SET2.md into a small static site for Vercel.

    python evaluations/publish_reports.py            # -> report_site/{index,set1,set2}.html
Then deploy report_site/ with the Vercel CLI. Pure static HTML, no JS deps.
"""
from __future__ import annotations

import os
import re
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = {"set1": os.path.join(HERE, "REPORT_SET1.md"),
       "set2": os.path.join(HERE, "REPORT_SET2.md")}
OUT = os.path.join(ROOT, "report_site")
TITLES = {"set1": "Set 1 · base vs SFT vs RAFT", "set2": "Set 2 · DPO vs RLAIF"}

CSS = """
:root{--bg:#0b0e14;--card:#12161f;--fg:#dfe5ee;--muted:#8b95a7;--accent:#f5b342;
--border:#232a38;--head:#1a2130;--good:#4ade80;--warn:#f5b342}
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
p,li{color:var(--fg)}em{color:var(--muted);font-style:normal}
strong{color:#fff}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
background:#0e1220;border:1px solid var(--border);border-radius:4px;padding:1px 5px;color:var(--accent)}
ul{padding-left:20px}li{margin:5px 0}
hr{border:0;border-top:1px solid var(--border);margin:1.5em 0}
.tw{overflow-x:auto;margin:12px 0;border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
th{background:var(--head);color:#c7cfdd;font-weight:600;position:sticky;top:0}
tr:last-child td{border-bottom:0}
td:not(:first-child){font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#c7cfdd}
tbody tr:hover td{background:rgba(245,179,66,.04)}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 22px;margin:18px 0}
.foot{color:var(--muted);font-size:12px;margin-top:40px;border-top:1px solid var(--border);padding-top:14px}
a{color:var(--accent)}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--card:#fff;--fg:#1a1f2b;--muted:#5b6478;
--border:#e2e6ee;--head:#eef1f6}code{background:#eef1f6}}
"""


def md_to_html(path: str) -> str:
    raw = open(path, encoding="utf-8").read()
    # drop the '====' separator rules (they aren't valid md headings)
    raw = "\n".join("" if re.fullmatch(r"=+\s*", ln) else ln for ln in raw.splitlines())
    html = markdown.markdown(raw, extensions=["tables", "sane_lists", "fenced_code"])
    # wrap every table in a horizontal-scroll container (mobile-safe)
    html = re.sub(r"(<table>.*?</table>)", r'<div class="tw">\1</div>', html, flags=re.S)
    return html


def page(body: str, active: str, title: str) -> str:
    nav = "".join(
        f'<a href="{href}"{" class=on" if active==key else ""}>{label}</a>'
        for key, href, label in [("index", "/", "Overview"),
                                  ("set1", "/set1", "Set 1"), ("set2", "/set2", "Set 2")])
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{CSS}</style></head><body><div class=wrap>"
            f"<nav>{nav}</nav>{body}"
            f"<div class=foot>SLM fine-tuning evaluation · grounded-QA, 500 pair_ids × 4 RAFT "
            f"conditions · LLM-judge headline + paired-bootstrap significance · generated from "
            f"the eval harness in <code>evaluations/</code>.</div>"
            f"</div></body></html>")


def index_body() -> str:
    return (
        "<h1>SLM fine-tuning — evaluation</h1>"
        "<p><em>Two independent grounded-QA experiments on US legal/financial text. Headline "
        "metric is an independent LLM judge; every model-vs-model claim is a paired-bootstrap "
        "significant result on one decontaminated held-out set.</em></p>"
        "<div class=card><h3><a href='/set1'>Set 1 — base vs SFT vs RAFT</a></h3>"
        "<p>SLM-125M and Gemma-2B, each as base / SFT / RAFT. <strong>Scale dominates</strong> "
        "(Gemma ~0.95–0.98 ≫ 125M); SFT lifts the 125M but <strong>RAFT hurts it</strong> "
        "(over-abstains on 84% of answerable questions); token-F1 inverts Gemma's ranking.</p></div>"
        "<div class=card><h3><a href='/set2'>Set 2 — alignment: DPO vs RLAIF</a></h3>"
        "<p>125M / 500M / Gemma-2B, aligned with DPO or RLAIF. The <strong>winning method flips "
        "with scale</strong> — RLAIF wins on the small SLMs, DPO on Gemma (all significant); DPO "
        "<strong>collapsed the 125M</strong> to 2-word answers.</p></div>"
        "<p class=foot style='border:0'>These are two separate experiments; no numbers are "
        "compared across them.</p>")


def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(
        page(index_body(), "index", "SLM fine-tuning evaluation"))
    for key in ("set1", "set2"):
        body = md_to_html(SRC[key])
        open(os.path.join(OUT, f"{key}.html"), "w", encoding="utf-8").write(
            page(body, key, TITLES[key]))
    open(os.path.join(OUT, "vercel.json"), "w", encoding="utf-8").write('{\n  "cleanUrls": true\n}\n')
    print("wrote", OUT, "->", os.listdir(OUT))


if __name__ == "__main__":
    main()
