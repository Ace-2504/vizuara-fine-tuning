# VIDEO SCRIPT — Set 2 (alignment: DPO vs RLAIF) · ~5 minutes

A complete read-aloud script. **[STAGE: …]** tells you what to show; **SAY:** is the exact
words to read. Pace ~150 words/min; ~5 minutes total. Live page:
**https://reportsite-jade.vercel.app/set2**

---

### 0:00–0:35 · Intro
**[STAGE: Open the Set 2 page — https://reportsite-jade.vercel.app/set2 — at the top.]**

**SAY:** "This is the second experiment. In the first one I showed that fine-tuning lets a
small model answer questions from documents. Set 2 asks the follow-up: once a model can
answer, can we make it *better* by aligning it toward preferred answers — and if so, which
alignment method should we use?"

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
lowest. The grey bars are the older word-overlap score — and just like in Set 1, notice how
badly it under-rates the Gemma models, which is why we don't rely on it."

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

### 4:50–5:00 · Close
**SAY:** "So, Set 2 in one line: alignment helps, but the best method flips with model size —
RLAIF for the small models, DPO for the large one — and DPO can destabilise a model that's
too small. And because we judged every model with an independent AI, not its own reward, the
comparison is fair."

**[STAGE: Stop, or return to the Overview page to wrap both experiments.]**

---
**Delivery notes:** the flip chart at 2:05 is the moment — pause on it. If asked why the
Gemma numbers look high while their word-overlap is low, point to the “What is the word-overlap
score?” page: the Gemma models paraphrase, which fools word-counting but not the judge.
