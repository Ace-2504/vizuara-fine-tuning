# VIDEO SCRIPT — Set 1 · Assignment 1 (Pretraining + QA and RAFT) · ~5.5 minutes

A complete read-aloud script. **[STAGE: …]** tells you what to show; **SAY:** is the exact words
to read. Pace ~150 words/min. Covers the assignment's required items: the deployed-site training
details (parameters · architecture · tokens · cost), one-click sample questions, and the
qualitative performance comparison.
Report page: **https://reportsite-jade.vercel.app/set1** · Model grid: **https://slm-arena-harman.vercel.app**

---

### 0:00–0:45 · Greeting & who I am
**[STAGE: On camera (or with the Set 1 page open) — https://reportsite-jade.vercel.app/set1.]**

**SAY:** "Hello, Dr. Sreedath Panat — sir, thank you for this bootcamp; it has genuinely changed
the way I think about these models. My name is Harman Sandhu, I'm a third-semester Computer
Science student based in the NCR region, and this is the recording of my Assignment 1 video. For
this assignment I trained and fine-tuned a set of small language models and then compared them
head-to-head, so I could see with real evidence which fine-tuning method works best — and at
which model size. The task is grounded question-answering: each model is handed a set of legal or
financial documents and must answer using only those documents, and say so when the answer isn't
there, without making things up. I put every model through three forms — base, supervised
fine-tuning, and RAFT — across two model sizes, and this experiment let me see clearly which
method helps, which one backfires, and just how much sheer model size matters."

---

### 0:45–1:05 · The models
**[STAGE: Point at the blue “The question” box on the Set 1 page.]**

**SAY:** "Quick setup: **base** is the model before any fine-tuning, **SFT** is where I taught it
to answer from the documents, and **RAFT** goes further — I also trained it to say ‘not in the
documents’ when the answer genuinely isn't there. I did this at two sizes — my 125-million model
and the 2-billion Gemma — so I could compare the recipes and the sizes together."

---

### 1:05–2:00 · The deployed model sites — parameters, architecture, tokens, cost
**[STAGE: Open the model grid — https://slm-arena-harman.vercel.app — and click a cell, e.g.
“125M · base”, to open that model's own deployed site. Scroll slowly through its stats.]**

**SAY:** "First, the deployed sites the assignment asks me to show. Every model in this experiment
has its own page. On this one you can see the four things that matter: the **trainable
parameters** — 125.8 million; the full **model architecture** — a 12-layer Llama-style network,
with its attention heads, RoPE positions and normalization laid out; the **total tokens across
the training epochs** — 2.48 billion over four epochs; and the **training cost** — the A100 hours
it took. And down here are **one-click sample questions**, so anyone can try the model without
typing. Every model — base, SFT and RAFT, at both sizes — has a page exactly like this, all
linked from this arena grid."

---

### 2:00–2:50 · The headline result — size dominates
**[STAGE: Go to the Set 1 page's first chart, “AI-judged correctness vs word-overlap.” Point at
the amber bars.]**

**SAY:** "Now the comparison. This chart is the heart of it. The amber bar is how often an
independent AI judge rated each model's answer correct, from zero to one. The two Gemma models sit
near the top, around 0.95 to 0.97; my 125-million models are far shorter. My first finding is
blunt — **size dominates**: the 2-billion model is in a different league, no matter which recipe I
use. The grey bars are an older word-counting score; notice how tiny they are for Gemma even
though its answers are correct — and that is exactly why I made the **AI judge, not word-overlap,
my headline metric**."

---

### 2:50–3:40 · The twist — RAFT hurts the small model
**[STAGE: Point at “SLM-125M SFT” then “SLM-125M RAFT.” Then scroll to “How the RAFT models behave
in four situations” and point at the 125M RAFT first row.]**

**SAY:** "Here's the finding that surprised me most. For my 125-million model, plain SFT lifts it
from basically zero to 0.26 — but RAFT, the version I taught to refuse, drops back down to 0.05.
Teaching the small model to be cautious made it worse. This table shows why: on questions the
document *does* answer, my 125M RAFT model still refuses about 84% of the time. It's too small to
tell ‘the answer is missing’ from ‘the answer is right here.’ The Gemma model handles the same
recipe fine — so RAFT is a method whose benefit depends on the size."

---

### 3:40–4:20 · Real examples
**[STAGE: Scroll to the “Real examples” section on the Set 1 page.]**

**SAY:** "I didn't want to only show charts, so here are real questions with every fine-tuned
model's actual answer and the judge's score, side by side. You can see it directly — the Gemma
models answer correctly and completely; my 125-million models often give short, wrong, or refused
answers on the very same question. Reading the real outputs makes the comparison concrete."

---

### 4:20–5:05 · See it live — the Arena
**[STAGE: Open the SLM Arena's Arena tab — https://slm-arena-harman.vercel.app — pick a held-out
question and click “Ask all 13 & judge.”]**

**SAY:** "And here it is live. In my SLM Arena I pick a held-out question and every model answers
it in real time, then the same blind judge scores each one against the gold answer. Watch my
125-million base produce almost nothing, its SFT version give a clean answer, and RAFT do the same
— the base-versus-fine-tuned gap from the chart, happening live, on a question none of them were
trained on."

---

### 5:05–5:35 · Close
**[STAGE: Go back to the Set 1 page — the “Ranking & which differences are real” section.]**

**SAY:** "So, sir, what this assignment let me establish: for grounded question-answering, plain
fine-tuning is what makes a tiny model usable at all; RAFT's refusal-training backfires at small
scale but is safe at 2 billion; and across the board, model size matters more than the recipe —
every size gap passed a statistical significance test. In short — SFT works everywhere, RAFT only
pays off once the model is big enough, and scale wins. Thank you for watching."

**[STAGE: Stop.]**

---
**Delivery notes:** speak the amber-vs-grey contrast slowly — it's the one visual people misread.
If asked why base-Gemma already scores 0.95, the honest answer: base-Gemma is Google's
*instruction-tuned* Gemma, a strong starting point rather than a blank slate; my 125M base is a
true blank slate, which is why it scores near zero. Confirm before recording that the model site
you open visibly shows all four items (parameters, architecture, tokens/epochs, cost).
