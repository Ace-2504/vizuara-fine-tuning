# VIDEO SCRIPT — Set 2 (alignment: DPO vs RLAIF) · ~5–6 minutes (incl. live-arena demo)

A complete read-aloud script. **[STAGE: …]** tells you what to show; **SAY:** is the exact
words to read. Pace ~150 words/min; ~5–6 minutes total with the live-arena demo. Live page:
**https://reportsite-jade.vercel.app/set2**

---

### 0:00–0:40 · Intro
**[STAGE: Open the Set 2 page — https://reportsite-jade.vercel.app/set2 — full screen, at the top.]**

**SAY:** "Small language models are cheap to run and easy to deploy, which makes them appealing
for real products — if you can make them good enough. The task here is grounded
question-answering: a small model is given legal or financial documents and must answer using
only those documents, without making things up. Now suppose you've already trained a small
model to do that. Can you make it *better* by aligning it toward the answers people prefer —
and if so, which alignment method should you use? That is exactly what this experiment
measures."

---

### 0:35–1:20 · What the experiment is
**[STAGE: Point at the blue “The question” intro box at the top.]**

**SAY:** "Every model here was first taught to answer, and then aligned two different ways.
The first method is **DPO** — it learns directly from pairs of answers, a preferred one and a
rejected one. The second is **RLAIF** — it learns through a separate reward model that scores
answers. I ran both methods on three model sizes: 125 million, 500 million, and the 2-billion
Gemma. The question is: does alignment help, and is DPO or RLAIF better?"

---

### 1:20–2:05 · Chart 1 — the scores
**[STAGE: Scroll to the first chart, “AI-judged correctness vs word-overlap.” Point at the
amber bars, top to bottom.]**

**SAY:** "Same setup as before — the amber bar is how often the independent AI judge — Google's gemini-3.1-flash-lite — rated
each model correct. And the same headline jumps out: **size dominates**. The two Gemma models
score around 0.9; the 500-million models sit near 0.3 to 0.44; the 125-million models are
lowest. The grey bars are the older word-overlap score — notice how badly it under-rates the
Gemma models, which is why we don't rely on it here."

---

### 2:05–3:05 · Chart 2 — the flip (the key finding)
**[STAGE: Scroll to the second chart, “Which alignment method wins, by model size.”]**

**SAY:** "This chart is the most important result of Set 2. It shows, for each model size, the
gap between the two alignment methods. Bars pointing right mean RLAIF won; bars pointing left
mean DPO won. And here's the finding: **the winning method flips with size.** For the small
125-million and 500-million models, RLAIF wins clearly. But for the 2-billion Gemma model,
DPO wins. There is no universal best method — it depends on how big your model is. On thin
preference data, the reward-model approach steadied the small models, while the simpler
direct method pulled ahead at the top end."

---

### 3:05–3:45 · The DPO collapse
**[STAGE: Scroll to the “Full scores” table. Point at the “SLM-125M SFT+DPO” row, the
AI-judged correctness cell and the “typical length” cell.]**

**SAY:** "There's a cautionary tale in this table. Look at the 125-million model aligned with
DPO — its AI-judged correctness is just 0.05, and its typical answer length is about two
words. DPO didn't just underperform on the smallest model — it *broke* it, collapsing it to
near-empty answers. RLAIF, on the same tiny model, was far safer. So DPO is fragile when the
model is small and the preference data is thin."

---

### 3:45–4:15 · The reward column and fairness
**[STAGE: Point at the “reward” column of the same table, and the note that begins
“About the reward column.” Optionally click “What these scores mean.”]**

**SAY:** "One fairness point I want to be explicit about. You'll see a reward column, and the
RLAIF rows say ‘omitted.’ That's deliberate. RLAIF is *trained* to maximise that reward
score, so grading RLAIF with the same reward would be circular — it would look good by
definition. That's why the headline metric is a completely independent AI judge that no model
was trained on. It keeps the comparison honest."

---

### 4:15–4:50 · Real examples
**[STAGE: Scroll to the “Real examples” section. Land on a table that includes the
125M DPO two-word answer next to the RLAIF answer.]**

**SAY:** "And here are real answers from all six aligned models on the same questions. You can
literally see the 125-million DPO model giving two-word non-answers next to the RLAIF version
actually attempting the question, and the Gemma models answering fully. The tables make the
numbers concrete."

---

### 4:50–5:35 · See it live — the SLM Arena
**[STAGE: Open the SLM Arena — https://slm-arena-harman.vercel.app — on the Arena tab.
Pick a held-out question and click “Ask all 13 & judge”.]**

**SAY:** "Here's the same evaluation *live*. This is my SLM Arena — I ask a held-out question
and every model answers in real time, then a blind judge, gemini-3.1-flash-lite, scores each
answer against the gold answer. Look at the two 125-million alignment models side by side: the
DPO version giving a two-word non-answer while the RLAIF version actually attempts the
question — the collapse from the chart, live — and the Gemma models answering fully. It's the
same story you just saw in the numbers, on a fresh question none of them were trained on."

**[STAGE: (optional) Click the Leaderboard tab and point at the caption under the heading.]**

**SAY (optional):** "There's a combined leaderboard of all my models here too. If you show it:
it uses a zero-to-ten four-dimension rubric — a different scale than this experiment's chart —
and the caption under it explains that it's the same questions, answers and judge, just scored
a different way."

---

### 5:35–5:45 · Close
**SAY:** "So, Set 2 in one line: alignment helps, but the best method flips with model size —
RLAIF for the small models, DPO for the large one — and DPO can destabilise a model that's
too small. And because we judged every model with an independent AI, not its own reward, the
comparison is fair."

**[STAGE: Stop.]**

---
**Delivery notes:** the flip chart at 2:05 is the moment — pause on it. If asked why the
Gemma numbers look high while their word-overlap is low, point to the “What is the word-overlap
score?” page: the Gemma models paraphrase, which fools word-counting but not the judge.
