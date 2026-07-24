"""Build the public Vercel site for the set1 / set2 evaluations — general-audience,
charted, and self-contained per experiment. Built directly from the judged results
(not from the .md), so the LOCAL REPORT_SET*.md files are never touched.

Pages: /  /set1  /set2  /scores (hidden)  /word-overlap (hidden)

    python evaluations/publish_reports.py       # -> report_site/*.html
Then deploy report_site/ with the Vercel CLI.
"""
from __future__ import annotations

import html
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from experiments import EXPERIMENTS, META            # noqa: E402
from stats import mean_ci, paired_delta_ci           # noqa: E402
from eval_report import load_items, by_key, rank_and_matrix, key_findings  # noqa: E402

RESULTS = os.path.join(ROOT, "eval_results")
OUT = os.path.join(ROOT, "report_site")
DATA = load_items(RESULTS)

COND_PLAIN = {"clean": "one correct document", "realistic": "correct document + distractors",
              "retrieval_failure": "answer is absent (correct action: refuse)",
              "closed_book": "no documents given (memory only)"}
# (label, metric-key, kind) — kind: rate | len | reward
COLS = [("AI-judged correctness", "judge_correct", "rate"),
        ("groundedness", "grounded", "rate"),
        ("matches reference", "matches_ref", "rate"),
        ("word-overlap", "token_f1", "rate"),
        ("wrongly refused", "false_abstain", "rate"),
        ("typical length", "resp_len_words", "len")]
REWARD_COL = ("reward", "reward", "reward")


def esc(s):
    return html.escape(str(s))


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def clean_terms(s):
    import re
    s = re.sub(r"\(F1↔judge disagreement ([\d.]+)\)",
               lambda m: f"(word-overlap and the judge disagree on {round(float(m.group(1))*100)}% of its answers)", s)
    for a, b in [("token-F1", "word-overlap"), ("over-abstains", "wrongly refuses"),
                 ("judged correctness", "AI-judged correctness"),
                 ("statistically resolved (paired bootstrap, 95% CI excludes 0)",
                  "confirmed as real by a significance test"),
                 ("paired bootstrap, 95% CI excludes 0", "confirmed by a significance test"),
                 (" — trust the judge, not word-overlap", " — so we trust the judge, not word-overlap")]:
        s = s.replace(a, b)
    return s


def q_of(user):
    return user.split("Question:")[-1].strip() if "Question:" in user else user[-160:]


# ---------------- charts ----------------
def grouped_chart(name):
    JUD, WO = "#f5b342", "#8b95a7"
    rows = []
    for v in EXPERIMENTS[name]:
        if v not in DATA:
            continue
        jv = list(by_key(DATA[v]["per_item"], "judge_correct").values())
        wv = list(by_key(DATA[v]["per_item"], "token_f1").values())
        if not jv:
            continue
        rows.append((META[v]["label"], sum(jv) / len(jv), sum(wv) / max(1, len(wv))))
    W, padL, padR, padT, padB, gh = 760, 215, 62, 60, 18, 46
    H = padT + padB + gh * len(rows)
    X = lambda v: padL + (W - padL - padR) * v
    p = [f'<text x="14" y="22" font-size="15" font-weight="600" fill="var(--fg)">AI-judged correctness vs word-overlap</text>',
         f'<rect x="{W-330}" y="11" width="12" height="12" rx="2" fill="{JUD}"/><text class="legend" x="{W-313}" y="21">AI-judged correctness</text>',
         f'<rect x="{W-152}" y="11" width="12" height="12" rx="2" fill="{WO}"/><text class="legend" x="{W-135}" y="21">word-overlap</text>']
    for g in (0, .25, .5, .75, 1.0):
        gx = X(g)
        p.append(f'<line x1="{gx:.1f}" y1="{padT-4}" x2="{gx:.1f}" y2="{H-padB}" stroke="var(--border)"/>')
        p.append(f'<text x="{gx:.1f}" y="{padT-8}" text-anchor="middle" class="legend">{g:.2f}</text>')
    for i, (label, j, w) in enumerate(rows):
        gy = padT + i * gh
        p.append(f'<text x="{padL-10}" y="{gy+gh/2+4:.1f}" text-anchor="end" font-size="12" fill="var(--fg)">{esc(label)}</text>')
        p.append(f'<rect x="{padL}" y="{gy+6}" width="{max(0,X(j)-padL):.1f}" height="15" rx="2.5" fill="{JUD}"/>')
        p.append(f'<text x="{X(j)+6:.1f}" y="{gy+18:.1f}" font-size="11" font-family="monospace" fill="var(--fg)">{j:.2f}</text>')
        p.append(f'<rect x="{padL}" y="{gy+24}" width="{max(0,X(w)-padL):.1f}" height="15" rx="2.5" fill="{WO}"/>')
        p.append(f'<text x="{X(w)+6:.1f}" y="{gy+36:.1f}" font-size="11" font-family="monospace" fill="var(--muted)">{w:.2f}</text>')
    svg = f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">{"".join(p)}</svg>'
    return (f'<div class="chart">{svg}<div class=cap>Each model has two bars: the amber bar is how '
            f'often an independent AI judge rated its answer correct; the grey bar is the older '
            f'“word-overlap” score. Where grey is far shorter than amber, word-overlap is '
            f'under-rating a model that is actually answering correctly.</div></div>')


def delta_chart(name):
    fams = [("125m", "SLM-125M"), ("500m", "SLM-500M"), ("gemma", "Gemma-2B")]
    rows = []
    for fam, lab in fams:
        dpo = [v for v in EXPERIMENTS[name] if META[v]["family"] == fam and v.endswith("-dpo")]
        rl = [v for v in EXPERIMENTS[name] if META[v]["family"] == fam and v.endswith("-rlaif")]
        if not dpo or not rl or dpo[0] not in DATA or rl[0] not in DATA:
            continue
        d = mean_ci(list(by_key(DATA[dpo[0]]["per_item"], "judge_correct").values()))[0]
        r = mean_ci(list(by_key(DATA[rl[0]]["per_item"], "judge_correct").values()))[0]
        rows.append((lab, r - d))
    if not rows:
        return ""
    W, rowh, padL, padT, padB = 760, 42, 130, 56, 26
    mid = padL + (W - padL - 30) / 2
    span = (W - padL - 30) / 2
    H = padT + padB + rowh * len(rows)
    m = max(abs(d) for _, d in rows) or 0.3
    p = ['<text x="14" y="22" font-size="15" font-weight="600" fill="var(--fg)">Which alignment method wins, by model size</text>',
         f'<line x1="{mid}" y1="{padT-4}" x2="{mid}" y2="{H-padB}" stroke="var(--border)"/>',
         f'<text class="legend" x="{padL}" y="{padT-8}">← DPO better</text>',
         f'<text class="legend" x="{W-30}" y="{padT-8}" text-anchor="end">RLAIF better →</text>']
    for i, (lab, d) in enumerate(rows):
        y = padT + i * rowh + rowh / 2
        w = span * (d / m) * 0.9
        col = "#38d9c4" if d > 0 else "#e2663b"
        p.append(f'<text x="{padL-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="var(--fg)">{esc(lab)}</text>')
        p.append(f'<rect x="{(mid if d>=0 else mid+w):.1f}" y="{y-9:.1f}" width="{abs(w):.1f}" height="18" rx="3" fill="{col}"/>')
        tx, anc = (mid + w + 8, "start") if d >= 0 else (mid + w - 8, "end")
        p.append(f'<text x="{tx:.1f}" y="{y+4:.1f}" text-anchor="{anc}" font-size="11.5" font-family="monospace" fill="var(--fg)">{d:+.3f}</text>')
    svg = f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">{"".join(p)}</svg>'
    return (f'<div class="chart">{svg}<div class=cap>The gap in AI-judged correctness between the two '
            f'alignment methods for each model size. RLAIF wins for the small models; DPO wins for '
            f'the 2B model — the winner flips with size.</div></div>')


# ---------------- tables ----------------
def cell(items, metric, kind):
    if kind == "len":
        v = list(by_key(items, "resp_len_words").values())
        return f"{median(v):.0f}w" if v else "—"
    vals = list(by_key(items, metric).values())
    if not vals:
        return "n/a"
    t = mean_ci(vals)
    return f"{t[0]:.3f} <span class=ci>[{t[1]:.3f}, {t[2]:.3f}]</span>"


def per_version_table(name):
    cols = COLS + ([REWARD_COL] if name == "set2" else [])
    th = "".join(f"<th>{esc(c[0])}</th>" for c in cols)
    body = []
    for v in EXPERIMENTS[name]:
        if v not in DATA:
            continue
        it = DATA[v]["per_item"]
        tds = [f'<td class="model">{esc(META[v]["label"])}{" · base" if META[v]["is_base"] else ""}</td>']
        for lbl, key, kind in cols:
            if kind == "reward":
                if META[v].get("reward_circular"):
                    tds.append("<td>omitted*</td>")
                else:
                    rv = list(by_key(it, "reward").values())
                    tds.append(f"<td>{mean_ci(rv)[0]:+.2f}</td>" if rv else "<td>n/a</td>")
            else:
                tds.append(f"<td>{cell(it, key, kind)}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    links = ('<p class="links"><a href="/scores">What these scores mean &rarr;</a>'
             '&nbsp;&nbsp;·&nbsp;&nbsp;<a href="/word-overlap">What is the word-overlap score? &rarr;</a></p>')
    note = ""
    if name == "set2":
        note = ('<p class="note"><b>About the reward column.</b> A secondary score from a separate '
                '“reward model” (built on the 500M model). <b>DPO</b> models show their score; '
                '<b>RLAIF</b> models show <b>“omitted*”</b> because they were <i>trained</i> to '
                'maximise this exact reward, so scoring them with it would be circular; <b>Gemma</b> '
                'shows <b>“n/a”</b> because the reward model is a different model family and its '
                'scores are not comparable. It is a side-signal only — the AI judge is the headline.</p>')
    return (f"<h2>Full scores</h2><div class=tw><table><thead><tr><th>model</th>{th}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>{note}{links}")


def ranking_html(name):
    models = [v for v in EXPERIMENTS[name] if v in DATA and not META[v]["is_base"]]
    rm = rank_and_matrix(DATA, models, "judge_correct")
    if not rm:
        return ""
    L = ["<h2>Ranking &amp; which differences are real</h2>",
         "<p><em>Ranked by AI-judged correctness. A difference between two models counts as real "
         "only when a statistical significance test is confident it is not just chance — otherwise "
         "it is marked “not resolved.”</em></p>",
         "<ol>"]
    for v in rm["order"]:
        t = rm["means"][v]
        L.append(f"<li><b>{esc(META[v]['label'])}</b> — {t[0]:.3f} <span class=ci>[{t[1]:.3f}, {t[2]:.3f}]</span></li>")
    L.append("</ol><div class=tw><table><thead><tr><th>comparison</th><th>difference</th><th>real?</th></tr></thead><tbody>")
    for a, b, d in rm["pairs"]:
        sig = "yes ✓" if d["significant"] else "not resolved"
        L.append(f"<tr><td class=model>{esc(META[a]['label'])} vs {esc(META[b]['label'])}</td>"
                 f"<td>{d['delta']:+.3f}</td><td>{sig}</td></tr>")
    L.append("</tbody></table></div>")
    return "".join(L)


def raft_html(name):
    rafts = [v for v in EXPERIMENTS[name] if v in DATA and META[v].get("is_raft")]
    if not rafts:
        return ""
    L = ["<h2>How the RAFT models behave in four situations</h2>",
         "<p><em>RAFT models are trained to answer when the document has the answer and to refuse "
         "when it doesn't. These four rows show whether they get that right.</em></p>"]
    for v in rafts:
        it = DATA[v]["per_item"]
        L.append(f"<h3>{esc(META[v]['label'])}</h3><div class=tw><table>"
                 "<thead><tr><th>situation</th><th>AI-judged</th><th>word-overlap</th><th>refusal rate</th></tr></thead><tbody>")
        for c in ["clean", "realistic", "retrieval_failure", "closed_book"]:
            jc = list(by_key(it, "judge_correct", c).values())
            f1 = list(by_key(it, "token_f1", c).values())
            ab = list(by_key(it, "abstain", c).values())
            L.append(f"<tr><td class=model>{esc(COND_PLAIN[c])}</td>"
                     f"<td>{mean_ci(jc)[0]:.3f}</td><td>{mean_ci(f1)[0]:.3f}</td>"
                     f"<td>{mean_ci(ab)[0]:.3f}</td></tr>")
        L.append("</tbody></table></div>")
    return "".join(L)


def pick_examples(name, n=6):
    tuned = [v for v in EXPERIMENTS[name] if v in DATA and not META[v]["is_base"]]
    byq = defaultdict(dict)
    for v in tuned:
        for it in DATA[v]["per_item"]:
            if it.get("cond") == "clean" and it.get("answerable"):
                byq[it["pair_id"]][v] = it
    full = {pid: d for pid, d in byq.items() if len(d) == len(tuned)}

    def variance(pid):
        dd = full[pid]
        vals = [((dd[v]["judge"]["correct"] - 1) / 4 if dd[v].get("judge") else 0) for v in tuned]
        m = sum(vals) / len(vals)
        return sum((x - m) ** 2 for x in vals) / len(vals)
    return tuned, sorted(full, key=variance, reverse=True)[:n], full


def examples_html(name):
    tuned, pids, full = pick_examples(name)
    if not pids:
        return ""
    L = ["<h2>Real examples</h2>",
         "<p><em>Six real questions from the test set, chosen because the models disagree most on "
         "them — so you can see the differences directly. Each table shows every fine-tuned "
         "model's actual answer and the AI judge's rating.</em></p>"]
    for i, pid in enumerate(pids, 1):
        d = full[pid]
        any_it = next(iter(d.values()))
        L.append(f'<div class="ex"><p class="exq"><b>Q{i}.</b> {esc(q_of(any_it["user"]))}</p>')
        L.append(f'<p class="exr"><b>Correct answer:</b> {esc(any_it["ref"][:220])}</p>')
        L.append('<div class=tw><table><thead><tr><th>model</th><th>its answer</th>'
                 '<th>AI-judged</th><th>grounded</th></tr></thead><tbody>')
        for v in tuned:
            it = d[v]
            j = it.get("judge", {})
            resp = it["resp"].strip().replace("\n", " ")[:200] or "(empty)"
            g = "yes" if j.get("grounded") else "no"
            L.append(f'<tr><td class=model>{esc(META[v]["label"])}</td>'
                     f'<td class=resp>{esc(resp)}</td>'
                     f'<td>{j.get("correct","?")}/5</td><td>{g}</td></tr>')
        L.append("</tbody></table></div></div>")
    return "".join(L)


def key_findings_html(name):
    bullets = key_findings(name, EXPERIMENTS[name], DATA, "judge_correct")
    lis = "".join(f"<li>{clean_terms(b)}</li>" for b in bullets)
    return f'<div class="card"><h3 style="margin-top:0">Key findings</h3><ul>{lis}</ul></div>'


def consistency_html(name):
    hs = {DATA[v]["result"].get("manifest", {}).get("eval_sha256") for v in EXPERIMENTS[name] if v in DATA}
    hs.discard(None)
    ok = len(hs) == 1
    return ('<h2>Consistency check</h2><p>' + ('All models were scored on the exact same set of '
            'held-out questions, so the comparison is fair. ✓' if ok else
            '⚠️ Models were scored on different question sets — comparisons may not be fair.') + '</p>')


# ---------------- page shell + content ----------------
INTRO = {
    "set1": ("<b>The question.</b> A small model has been shown one or more documents and asked a "
             "question it must answer <i>using only those documents</i>. Does fine-tuning actually "
             "make a tiny model good at this — and does teaching it to <i>refuse</i> when the answer "
             "isn't there (the “RAFT” recipe) help or hurt? Each model appears in three forms: "
             "<b>base</b> (not fine-tuned), <b>SFT</b> (taught to answer from documents), and "
             "<b>RAFT</b> (also taught to refuse when the answer is missing)."),
    "set2": ("<b>The question.</b> Once a model can answer from documents, can we make it <i>better</i> "
             "by aligning it to preferred answers — and which alignment method is better, <b>DPO</b> "
             "(learns straight from preferred-vs-rejected answer pairs) or <b>RLAIF</b> (learns through "
             "a reward model)? Every model here was first taught to answer (SFT), then aligned each of "
             "the two ways, across three model sizes."),
}


def report_page(name):
    h1 = ("Set 1 — does fine-tuning a small model help?" if name == "set1"
          else "Set 2 — which alignment method is better?")
    parts = [f"<h1>{h1}</h1>", f'<div class="intro">{INTRO[name]}</div>',
             key_findings_html(name), grouped_chart(name)]
    if name == "set2":
        parts.append(delta_chart(name))
    parts += [per_version_table(name), ranking_html(name), raft_html(name),
              examples_html(name), consistency_html(name)]
    return "".join(parts)


def page(body, active, title):
    nav = "".join(f'<a href="{h}"{" class=on" if active==k else ""}>{l}</a>'
                  for k, h, l in [("index", "/", "Overview"), ("set1", "/set1", "Set 1"),
                                  ("set2", "/set2", "Set 2")])
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><div class=wrap>"
            f"<nav>{nav}</nav>{body}"
            f"<div class=foot>Grounded question-answering evaluation · held-out questions · "
            f"independent AI judge with 95% confidence ranges and paired significance testing · "
            f"two separate experiments, not compared against each other.</div></div></body></html>")


def overview_body():
    return (
        "<h1>Can a small language model answer questions from documents — and how should we train it?</h1>"
        "<div class=intro><b>What we set out to do.</b> Large language models are expensive to run. "
        "We wanted to know how far <i>small</i> models (from 125 million up to 2 billion parameters) "
        "can be pushed on a practical, high-stakes task: <b>grounded question-answering</b> over US "
        "legal and financial documents. “Grounded” means the model must answer <i>using only the "
        "documents it is given</i> — and say so when the answer isn't there — rather than making "
        "things up. For legal and financial use, a confident wrong answer is worse than no answer.</div>"
        "<p>Getting a small model to do this well is really two separate questions, so we ran "
        "<b>two independent experiments</b>:</p>"
        "<div class=card><h3><a href='/set1'>Set 1 — does fine-tuning help, and does teaching refusal help?</a></h3>"
        "<p>First, the basics: can fine-tuning turn a small, general model into a competent document "
        "reader? And a subtler question — if we teach it to <i>refuse</i> when the documents don't "
        "contain the answer (a recipe called RAFT), does that make it safer, or too timid? We compare "
        "each model as base, SFT, and RAFT. <b>Finding:</b> size dominates, and teaching refusal "
        "<i>backfires</i> on the smallest model.</p></div>"
        "<div class=card><h3><a href='/set2'>Set 2 — once it can answer, does alignment make it better?</a></h3>"
        "<p>Second, polish: after a model can answer, we can <i>align</i> it toward better answers. "
        "There are two popular methods — DPO and RLAIF — and it isn't obvious which is better for "
        "small models. We test both across three sizes. <b>Finding:</b> the winner flips with size, "
        "and one method breaks the smallest model.</p></div>"
        "<div class=intro><b>How we judged fairly.</b> Every answer is scored by an independent AI "
        "judge that never sees which model wrote it, on the same held-out questions, with confidence "
        "ranges and statistical significance tests — so the rankings are trustworthy, not cherry-picked. "
        "<a href='/scores'>What the scores mean →</a></div>"
        "<p class=foot style='border:0'>The two experiments are independent; no numbers are compared "
        "between them.</p>")


def scores_body():
    return (
        "<h1>What the scores mean</h1>"
        "<p><em>A plain-language guide to every column in the tables. No background needed.</em></p>"
        "<div class=intro>Picture handing someone a few documents and asking a question. They write "
        "an answer. How do we grade it? That's what these scores do — and here's the story of each, "
        "in the order they matter.</div>"
        "<h3>AI-judged correctness — the main score</h3>"
        "<p>An independent AI (Google's Gemini), which <b>never sees which model wrote the answer</b>, "
        "reads the documents, the question, the known-correct answer, and the model's answer, and rates "
        "how correct it is from 1 to 5. We rescale that to a 0–1 number (1.0 = perfect). This is our "
        "headline score because, unlike counting words, it understands <i>meaning</i> — a correct "
        "answer phrased differently still counts as correct.</p>"
        "<h3>Groundedness — is it made up?</h3>"
        "<p>A separate check: does every fact in the answer actually come from the documents? A fluent, "
        "confident answer that invents a number or a name is <b>not grounded</b>. For legal and "
        "financial text this matters as much as being correct — 1.0 means fully supported by the "
        "documents.</p>"
        "<h3>Matches reference — is it basically the known answer?</h3>"
        "<p>A stricter yes/no: is the model's answer essentially the same as the one correct reference "
        "answer we already have? Reported as the fraction of times it matched.</p>"
        "<h3>Word-overlap — the old automatic score</h3>"
        "<p>Before AI judges, answers were graded automatically by counting shared words with the "
        "reference. It's fast but blunt — it punishes a correct answer worded differently. We still "
        "show it so you can see where it disagrees with the judge. "
        "<a href='/word-overlap'>Full explanation, with an example →</a></p>"
        "<h3>Wrongly refused — is it too timid?</h3>"
        "<p>How often the model said “not in the documents” <b>when the answer actually was there</b>. "
        "Lower is better; a high number means the model is over-cautious and unhelpful.</p>"
        "<h3>Refusal rate — how often it declines</h3>"
        "<p>How often the model refuses to answer at all. This is <i>good</i> when the answer is "
        "genuinely absent, and <i>bad</i> when it isn't — read it together with the situation.</p>"
        "<h3>Typical length</h3>"
        "<p>The typical answer length in words. Useful because some training makes models ramble, and "
        "longer isn't better — it can hide padding or drift.</p>"
        "<h3>Reward <span style='color:var(--muted);font-weight:400'>(alignment experiment only)</span></h3>"
        "<p>A side-score from a separate “reward model” used during one of the training methods. It's "
        "shown only as a secondary signal, never as the headline, and it's left out for models that "
        "were trained on that very reward (scoring them with it would be circular). The AI judge is "
        "always the score that decides rankings.</p>"
        "<p style='margin-top:28px'><a href='/set1'>← Back to Set 1</a> &nbsp;·&nbsp; <a href='/set2'>Back to Set 2 →</a></p>")


def word_overlap_body():
    return (
        "<h1>What is the “word-overlap” score?</h1>"
        "<p><em>Also called F1. It's the old, automatic way of grading an answer — and it's why we "
        "added an AI judge.</em></p>"
        "<div class=intro>Before AI models were good enough to grade other models, we graded answers "
        "with a simple trick: <b>count how many words the answer shares with the known-correct "
        "reference</b>, and balance that against how many extra words it added. That balance is the "
        "“F1” or word-overlap score, from 0 to 1.</div>"
        "<h3>Why it's handy</h3>"
        "<p>It needs no AI, costs nothing, and is perfectly repeatable. For decades it was the standard "
        "automatic score for question-answering.</p>"
        "<h3>Why it misleads — a real example</h3>"
        "<p>Take this real question from our test set:</p>"
        "<div class=card><p><b>Question:</b> What are the two primary sources from which surfactant "
        "used in RDS treatment is derived?</p>"
        "<p><b>Correct answer:</b> One type is cow-derived and the other is synthetic.</p>"
        "<p><b>A model answered:</b> “one type of surfactant originates from cows as a natural product, "
        "while the other is synthetically produced.”</p>"
        "<p>That answer is <b>completely correct</b> — but it shares almost no exact words with the "
        "reference, so word-overlap scores it around <b>0.14</b> (looks terrible), while the AI judge "
        "correctly rates it <b>5/5</b>.</p></div>"
        "<p>The opposite also happens: an answer can repeat the reference's words while being wrong or "
        "rambling, and word-overlap rewards it anyway.</p>"
        "<h3>What we do about it</h3>"
        "<p>We make the <b>AI judge</b> the headline score, because it reads for meaning. We still show "
        "word-overlap next to it — when the two disagree a lot (you'll see it in the charts), that's a "
        "sign word-overlap is being fooled, usually by a model that paraphrases or writes at length.</p>"
        "<p style='margin-top:28px'><a href='/set1'>← Back to Set 1</a> &nbsp;·&nbsp; <a href='/set2'>Back to Set 2 →</a> &nbsp;·&nbsp; <a href='/scores'>All scores explained →</a></p>")


CSS = """
:root{--bg:#0b0e14;--card:#12161f;--fg:#dfe5ee;--muted:#8b95a7;--accent:#f5b342;--border:#232a38;--head:#1a2130}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:28px 20px 80px}
nav{position:sticky;top:0;background:rgba(11,14,20,.92);backdrop-filter:blur(6px);border-bottom:1px solid var(--border);padding:12px 20px;margin:-28px -20px 24px;z-index:5}
nav a{color:var(--muted);text-decoration:none;margin-right:18px;font-size:13.5px;font-weight:500}
nav a:hover,nav a.on{color:var(--accent)}
h1{font-size:26px;line-height:1.25;margin:.3em 0 .5em}
h2{font-size:19px;margin:1.9em 0 .5em;padding-top:.5em;border-top:1px solid var(--border)}
h3{font-size:15.5px;color:var(--accent);margin:1.4em 0 .4em;font-weight:600}
p,li{color:var(--fg)}em{color:var(--muted);font-style:normal}strong,b{color:#fff}
ul,ol{padding-left:20px}li{margin:6px 0}
a{color:var(--accent)}
.intro{background:#0e1220;border-left:3px solid var(--accent);border-radius:8px;padding:14px 18px;margin:16px 0;font-size:14px;line-height:1.75;color:#cdd5e2}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin:16px 0}
.card h3 a{text-decoration:none}
.chart{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin:18px 0}
.cap{color:var(--muted);font-size:12px;margin-top:8px;line-height:1.5}
.legend{fill:var(--muted);font-size:11px}
.tw{overflow-x:auto;margin:12px 0;border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top}
th{background:var(--head);color:#c7cfdd;font-weight:600;white-space:nowrap}
tr:last-child td{border-bottom:0}
td{white-space:nowrap;font-family:ui-monospace,Menlo,monospace;color:#c7cfdd}
td.model{white-space:nowrap;font-family:inherit;color:var(--fg);font-weight:500}
td.resp{white-space:normal;font-family:inherit;color:#c7cfdd;min-width:280px;line-height:1.5}
.ci{color:var(--muted);font-size:11px}
tbody tr:hover td{background:rgba(245,179,66,.04)}
.links{font-size:13.5px;margin:10px 2px 4px}
.note{font-size:13px;color:#cdd5e2;background:#0e1220;border-radius:8px;padding:12px 15px;margin:12px 0;line-height:1.6}
.ex{margin:16px 0}
.exq{margin:0 0 4px}.exr{margin:0 0 8px;color:var(--muted)}.exr b{color:var(--fg)}
.foot{color:var(--muted);font-size:12px;margin-top:44px;border-top:1px solid var(--border);padding-top:14px}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--card:#fff;--fg:#1a1f2b;--muted:#5b6478;--border:#e2e6ee;--head:#eef1f6}}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    pages = {"index.html": page(overview_body(), "index", "SLM fine-tuning evaluation"),
             "set1.html": page(report_page("set1"), "set1", "Set 1 · fine-tuning a small model"),
             "set2.html": page(report_page("set2"), "set2", "Set 2 · alignment methods"),
             "scores.html": page(scores_body(), None, "What the scores mean"),
             "word-overlap.html": page(word_overlap_body(), None, "What is the word-overlap score?")}
    for fn, htmlc in pages.items():
        open(os.path.join(OUT, fn), "w", encoding="utf-8").write(htmlc)
    open(os.path.join(OUT, "vercel.json"), "w", encoding="utf-8").write('{\n  "cleanUrls": true\n}\n')
    print("wrote", OUT, "->", sorted(pages))


if __name__ == "__main__":
    main()
