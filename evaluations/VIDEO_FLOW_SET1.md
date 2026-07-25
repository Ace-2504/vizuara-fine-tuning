# VIDEO SCRIPT — Set 1 (base vs SFT vs RAFT) · ~5–6 minutes (incl. live-arena demo)

A complete read-aloud script. **[STAGE: …]** tells you what to show; **SAY:** is the exact
words to read. Pace is ~150 words/min; the whole thing is ~5–6 minutes with the live-arena demo.
Live page: **https://reportsite-jade.vercel.app/set1**

---

### 0:00–0:50 · Greeting & who I am
**[STAGE: On camera (or with the Set 1 page open) — https://reportsite-jade.vercel.app/set1.]**

**SAY:** "Good day, Dr. Sreedath Panat — sir, thank you for this bootcamp; it has genuinely
changed the way I think about these models. My name is Harman Sandhu, I'm a third-semester
Computer Science student based in the NCR region, and this is the recording of my Assignment 1
video, where I'll walk you through the performance comparison of the small language models I
trained and fine-tuned. Small language models are cheap to run and easy to deploy — a single
GPU, even a laptop, is enough — but what I really wanted to find out is how *capable* a small
model can be on a serious task. The task I chose is grounded question-answering: the model is
handed a set of legal or financial documents and must answer using only those documents, and
say so when the answer isn't there, never making things up. This experiment asks the most
fundamental version of that question: does fine-tuning actually make a small model good at this
— and does teaching it to refuse when it's unsure help, or backfire?"

---

### 0:50–1:20 · What the experiment is
**[STAGE: Point at the blue “The question” intro box near the top of the Set 1 page.]**

**SAY:** "Let me set it up. I take each model and put it through three forms. **Base** is the
model before any fine-tuning. **SFT** is where I taught it to answer from the documents. And
**RAFT** goes one step further — I also trained it to say ‘not in the documents’ when the
answer genuinely isn't there. So the real question underneath is: does fine-tuning help, and
does teaching the model to *refuse* pay off — or does it cost me?"

---

### 1:20–2:10 · The headline chart
**[STAGE: Scroll to the first chart, titled “AI-judged correctness vs word-overlap.”
Point at the amber bars.]**

**SAY:** "This first chart is where it all comes together — and honestly, I was thrilled when I
saw it. For every model, the amber bar is how often the judge rated its answer correct, from
zero to one. Look at the two Gemma-2B models near the bottom — their bars are almost full,
around 0.95 to 0.97. Now look at my 125-million models at the top — much shorter. The single
biggest finding of this experiment is that **model size dominates**: the 2-billion model is in
a different league from the 125-million one, no matter how I train it."

**[STAGE: Point at the grey bars next to the amber ones, then at the legend.]**

**SAY:** "The grey bars are an older automatic score called *word-overlap*, which just counts
shared words with the reference answer. Notice how for Gemma the grey bar is tiny while the
amber bar is nearly full — that gap is not a mistake, and it's exactly why I chose an AI judge
over word-counting. I'll come back to it."

---

### 2:10–3:00 · The twist — RAFT hurts the small model
**[STAGE: Still on the chart, point at “SLM-125M SFT” then “SLM-125M RAFT.”]**

**SAY:** "Now here's the result that genuinely surprised me. For my 125-million model, plain
fine-tuning — SFT — lifts it from basically zero up to about 0.26. But RAFT, the version I
taught to refuse, actually drops *back down* to 0.05. Teaching the small model to be cautious
made it worse — the opposite of what I expected."

**[STAGE: Scroll down to the section “How the RAFT models behave in four situations.”
Point at the SLM-125M RAFT table, the first row.]**

**SAY:** "And this table shows exactly why. In the very first row — where the document *does*
contain the answer — my 125M RAFT model refuses to answer about 84% of the time. It over-learned
the lesson. It's simply too small to tell ‘the answer is missing’ from ‘the answer is right
here,’ so it refuses almost everything. My 2-billion Gemma model handles the same recipe
perfectly — it only refuses when it should."

---

### 3:00–3:45 · Why an AI judge, not word-overlap
**[STAGE: Scroll back to the first chart, point again at Gemma-2B RAFT — tiny grey bar,
near-full amber bar. Then click the “What is the word-overlap score?” link under the
Full-scores table.]**

**SAY:** "Let me come back to those grey bars, because this was an important design decision for
me. Word-overlap rated Gemma's RAFT model at 0.14 — a failing score — while the judge rated it
0.97. So who's right? On this page I've put a real example: the model gave a completely correct
answer, just worded differently from the reference, and word-overlap punished it for that.
That's exactly why I made the AI judge my headline metric — it reads for *meaning*, not matching
words."

**[STAGE: Go back to the Set 1 page.]**

---

### 3:45–4:30 · Real examples
**[STAGE: Scroll to the “Real examples” section. Pick one table on screen where the
125M models score low and Gemma scores high.]**

**SAY:** "I didn't want to just show you bar charts, so here are real questions with every
fine-tuned model's actual answer and the judge's rating, side by side. You can see the pattern
with your own eyes: the Gemma models give correct, complete answers; my 125M models often give
short, wrong, or refused answers on the very same question. Reading the actual outputs makes it
real in a way a chart can't."

---

### 4:30–5:15 · See it live — the SLM Arena
**[STAGE: Open the SLM Arena — https://slm-arena-harman.vercel.app — on the Arena tab.
Pick a held-out question from the list and click “Ask all 13 & judge”.]**

**SAY:** "Everything so far has been the finished numbers — but I also wanted you to see it
happen live. This is my SLM Arena. I pick one of the held-out questions, and every model I
built answers it in real time, then a blind judge — the same gemini-3.1-flash-lite, handed the
gold answer — scores each one on the spot. Watch my 125-million base model produce almost
nothing, its SFT version give a clean, correct answer, and RAFT do the same. That
base-versus-fine-tuned gap from the chart is happening right in front of you, on a question none
of these models were trained on."

**[STAGE: (optional) Click the Leaderboard tab and point at the caption under the heading.]**

**SAY (optional):** "I've also built a combined leaderboard of all my models here. One quick
note if I show it: it scores on a richer zero-to-ten rubric — correctness, completeness,
groundedness and clarity — so those numbers are on a different scale than this experiment's
chart. The caption underneath explains it: same questions, same answers, same judge, just a
different scoring lens."

---

### 5:15–5:50 · Close
**[STAGE: Go back to the Set 1 page — the “Ranking & which differences are real” section.]**

**SAY:** "So, sir, three findings from this experiment. First, plain fine-tuning is what makes a
tiny model usable at all. Second, teaching it to refuse backfires at small scale — it becomes
far too timid. And third, size dominates everything: the 2-billion model beats the 125-million
one on every recipe, and every one of those gaps passed a statistical significance test. My
takeaway — fine-tuning is essential, refusal-training is a trap at small scale, and scale still
wins. Thank you for watching."

**[STAGE: Stop.]**

---
**Delivery notes:** speak the amber-vs-grey contrast slowly — it's the one visual people
misread. If asked why base-Gemma already scores 0.95, the honest answer: base-Gemma is Google's
*instruction-tuned* Gemma, so it's a strong starting point, not a blank slate; my 125M base is a
true blank slate, which is why it scores zero.
