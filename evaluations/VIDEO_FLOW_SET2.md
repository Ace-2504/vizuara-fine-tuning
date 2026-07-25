# VIDEO SCRIPT — Set 2 (alignment: DPO vs RLAIF) · ~5–6 minutes (incl. live-arena demo)

A complete read-aloud script. **[STAGE: …]** tells you what to show; **SAY:** is the exact
words to read. Pace ~150 words/min; ~5–6 minutes total with the live-arena demo.
Live page: **https://reportsite-jade.vercel.app/set2**

---

### 0:00–0:50 · Greeting & who I am
**[STAGE: On camera (or with the Set 2 page open) — https://reportsite-jade.vercel.app/set2.]**

**SAY:** "Hello, Dr. Sreedath Panat — sir, thank you again for this bootcamp; the depth of it
has pushed me to build things I didn't think I could. I'm Harman Sandhu, a third-semester
Computer Science student based in the NCR region, and this is the recording of my Assignment 2
video, where I'll take you through the performance comparison of the small language models I
trained, fine-tuned, and aligned. Small language models are cheap to run and easy to deploy,
which makes them genuinely appealing for real products — if you can make them good enough. The
task I focused on is grounded question-answering: the model is given legal or financial
documents and must answer using only those documents, without making things up. In this
experiment I start from a model that can already answer, and ask: can I make it *better* by
aligning it toward the answers people prefer — and if so, which alignment method should I use,
DPO or RLAIF?"

---

### 0:50–1:30 · What the experiment is
**[STAGE: Point at the blue “The question” intro box at the top.]**

**SAY:** "Here's the setup. Every model here I first taught to answer, and then aligned two
different ways. The first is **DPO** — it learns directly from pairs of answers, a preferred one
and a rejected one. The second is **RLAIF** — it learns through a separate reward model that
scores answers. I ran both methods across three model sizes — 125 million, 500 million, and the
2-billion Gemma — because I wanted to know not just whether alignment helps, but whether the
*best* method changes with scale."

---

### 1:30–2:10 · Chart 1 — the scores
**[STAGE: Scroll to the first chart, “AI-judged correctness vs word-overlap.” Point at the
amber bars, top to bottom.]**

**SAY:** "Same kind of chart I use throughout — the amber bar is how often the judge rated each
model correct. And the same headline jumps straight out: **size dominates**. My two Gemma
models score around 0.9; the 500-million models sit near 0.3 to 0.44; the 125-million models are
lowest. The grey bars are word-overlap — notice how badly it under-rates the Gemma models, which
is exactly why I don't rely on it."

---

### 2:10–3:10 · Chart 2 — the flip (the key finding)
**[STAGE: Scroll to the second chart, “Which alignment method wins, by model size.”]**

**SAY:** "Now this second chart is the one I'm most proud of, because it's the real finding. It
shows, for each model size, the gap between the two alignment methods. Bars pointing right mean
RLAIF won; pointing left means DPO won. And here's the result: **the winning method flips with
size.** For my small 125-million and 500-million models, RLAIF wins clearly. But for the
2-billion Gemma, DPO wins. There is no single best method — it depends on how big the model is.
My read: on thin preference data, the reward-model approach steadied the small models, while the
simpler direct method pulled ahead at the top end."

---

### 3:10–3:50 · The DPO collapse
**[STAGE: Scroll to the “Full scores” table. Point at the “SLM-125M SFT+DPO” row, the
AI-judged correctness cell and the “typical length” cell.]**

**SAY:** "There's also a cautionary tale I have to be honest about. Look at my 125-million model
aligned with DPO — its judged correctness is just 0.05, and its typical answer is about two
words long. DPO didn't just underperform on the smallest model — it *broke* it, collapsing it to
near-empty answers. RLAIF on that same tiny model was far safer. So I learned that DPO is fragile
when the model is small and the preference data is thin."

---

### 3:50–4:20 · The reward column and fairness
**[STAGE: Point at the “reward” column of the same table, and the note that begins
“About the reward column.” Optionally click “What these scores mean.”]**

**SAY:** "There's one fairness decision I want to be explicit about, because I think it matters.
You'll see a reward column, and the RLAIF rows say ‘omitted.’ That's deliberate on my part.
RLAIF is *trained* to maximise that reward score, so if I graded RLAIF with the same reward it
would look good by definition — that would be circular. So I made the headline metric a
completely independent judge that no model was trained on. I wanted the comparison to be honest,
even where it's less flattering to my own work."

---

### 4:20–4:55 · Real examples
**[STAGE: Scroll to the “Real examples” section. Land on a table that includes the
125M DPO two-word answer next to the RLAIF answer.]**

**SAY:** "And here are real answers from all six of my aligned models on the same questions. You
can literally see the 125-million DPO model giving two-word non-answers right next to the RLAIF
version actually attempting the question, and the Gemma models answering fully. The tables make
the numbers concrete."

---

### 4:55–5:40 · See it live — the SLM Arena
**[STAGE: Open the SLM Arena — https://slm-arena-harman.vercel.app — on the Arena tab.
Pick a held-out question and click “Ask all 13 & judge”.]**

**SAY:** "I also wanted to show this happening live. This is my SLM Arena — I ask a held-out
question and every model I built answers it in real time, then a blind judge,
gemini-3.1-flash-lite, scores each answer against the gold answer. Watch my two 125-million
alignment models side by side: the DPO version giving a two-word non-answer while the RLAIF
version actually attempts the question — that collapse from the chart, live — and the Gemma
models answering fully. It's the same story you just saw in the numbers, on a fresh question
none of them were trained on."

**[STAGE: (optional) Click the Leaderboard tab and point at the caption under the heading.]**

**SAY (optional):** "I've also built a combined leaderboard of all my models here. If I show it:
it uses a zero-to-ten four-dimension rubric — a different scale than this experiment's chart —
and the caption under it explains it's the same questions, answers and judge, just scored a
different way."

---

### 5:40–5:50 · Close
**SAY:** "So, sir, my experiment in one line: alignment helps, but the best method flips with
model size — RLAIF for the small models, DPO for the large one — and DPO can destabilise a model
that's too small. And because I judged every model with an independent AI, not its own reward,
the comparison stays fair. Thank you for watching."

**[STAGE: Stop.]**

---
**Delivery notes:** the flip chart at 2:10 is the moment — pause on it. If asked why the Gemma
numbers look high while their word-overlap is low, point to the “What is the word-overlap score?”
page: the Gemma models paraphrase, which fools word-counting but not the judge.
