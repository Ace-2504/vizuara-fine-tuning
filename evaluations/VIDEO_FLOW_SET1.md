# VIDEO SCRIPT — Set 1 (base vs SFT vs RAFT) · ~5 minutes

A complete read-aloud script. **[STAGE: …]** tells you what to show; **SAY:** is the exact
words to read. Pace is ~150 words/min; the whole thing is ~5 minutes. Live page:
**https://reportsite-jade.vercel.app/set1**

---

### 0:00–0:40 · Intro
**[STAGE: Open the Set 1 page — https://reportsite-jade.vercel.app/set1 — full screen, at the top.]**

**SAY:** "Small language models are cheap to run and easy to deploy — a single GPU, even a
laptop, is enough. The real question is how *capable* a small model can be on a serious task.
The task here is grounded question-answering: the model is given a set of legal or financial
documents and must answer using only those documents — and say so when the answer isn't there,
never inventing facts. This experiment asks the most basic version of that question: does
fine-tuning make a small model good at this, and does teaching it to refuse when it's unsure
help — or hurt?"

---

### 0:35–1:10 · What the experiment is
**[STAGE: Point at the blue “The question” intro box near the top of the Set 1 page.]**

**SAY:** "Each model is shown one or more documents and a question it must answer using only
those documents. I test every model in three forms. **Base** is the model before any
fine-tuning. **SFT** is taught to answer from the documents. And **RAFT** goes one step
further — it's also taught to say ‘not in the documents’ when the answer genuinely isn't
there. So the question underneath is: does fine-tuning help, and does teaching the model to
*refuse* help or hurt?"

---

### 1:10–2:00 · The headline chart
**[STAGE: Scroll to the first chart, titled “AI-judged correctness vs word-overlap.”
Point at the amber bars.]**

**SAY:** "This chart is the heart of it. For every model, the amber bar shows how often an
independent AI judge — Google's gemini-3.1-flash-lite — rated its answer correct, from zero to one. Look at the two Gemma-2B
models near the bottom: their bars are almost full, around 0.95 to 0.97. Now look at the
125M models at the top — much shorter. The single biggest finding of Set 1 is simply that
**model size dominates**. The 2-billion model is in a different league from the 125-million
one, no matter how you train it."

**[STAGE: Point at the grey bars next to the amber ones, then at the legend.]**

**SAY:** "The grey bars are an older automatic score called *word-overlap*, which just counts
shared words with the reference answer. Notice how for Gemma the grey bar is tiny while the
amber bar is nearly full. That's not a mistake — it's the reason I use an AI judge, and
I'll come back to it."

---

### 2:00–2:55 · The twist — RAFT hurts the small model
**[STAGE: Still on the chart, point at “SLM-125M SFT” then “SLM-125M RAFT.”]**

**SAY:** "Here's the surprising result. For the 125-million model, plain fine-tuning — SFT —
lifts it from basically zero up to about 0.26. But RAFT, the version taught to refuse,
actually drops *back down* to 0.05. Teaching the small model to be cautious made it worse."

**[STAGE: Scroll down to the section “How the RAFT models behave in four situations.”
Point at the SLM-125M RAFT table, the first row.]**

**SAY:** "This table shows why. In the very first row — where the document *does* contain the
answer — the 125M RAFT model refuses to answer about 84% of the time. It over-learned the
lesson. It's too small to tell the difference between ‘the answer is missing’ and ‘the
answer is right here,’ so it just refuses everything. For the 2-billion Gemma model, the
same recipe works fine — it refuses only when it should."

---

### 3:00–3:45 · Why an AI judge, not word-overlap
**[STAGE: Scroll back to the first chart, point again at Gemma-2B RAFT — tiny grey bar,
near-full amber bar. Then click the “What is the word-overlap score?” link under the
Full-scores table.]**

**SAY:** "Remember the grey bars. Word-overlap rated Gemma's RAFT model at 0.14 — a failing
score — while the judge rated it 0.97. Who's right? On this page you can see a real example:
the model gave a completely correct answer, just worded differently from the reference, so
word-overlap punished it. This is exactly why the headline score here is an AI judge that
reads for *meaning*, not an automatic word-counter."

**[STAGE: Go back to /set1.]**

---

### 3:45–4:30 · Real examples
**[STAGE: Scroll to the “Real examples” section. Pick one table on screen where the
125M models score low and Gemma scores high.]**

**SAY:** "These are real questions with every fine-tuned model's actual answer and the
judge's rating side by side. You can see the pattern with your own eyes: the Gemma models
give correct, complete answers; the 125M models often give short, wrong, or refused answers
on the same question. It's one thing to see a bar chart — it's another to read the actual
outputs."

---

### 4:30–5:00 · Close
**[STAGE: Scroll to the “Ranking & which differences are real” section.]**

**SAY:** "So, three findings from Set 1. First, plain fine-tuning is what makes a tiny model
usable at all. Second, teaching it to refuse backfires at small scale — it becomes too timid.
And third, size dominates everything: the 2-billion model beats the 125-million one on every
recipe, and every one of those gaps passed a statistical significance test. The takeaway —
fine-tuning is essential, refusal-training is a trap at small scale, and scale still wins."

**[STAGE: Click through to /set2 if recording both back-to-back, otherwise stop.]**

---
**Delivery notes:** speak the amber-vs-grey contrast slowly — it's the one visual people
misread. If asked why base-Gemma already scores 0.95, the honest answer: base-Gemma is
Google's *instruction-tuned* Gemma, so it's a strong starting point, not a blank slate; the
125M base is a true blank slate, which is why it scores zero.
