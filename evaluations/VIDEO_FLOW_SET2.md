# VIDEO SCRIPT — Set 2 · Assignment 2 (RLAIF and DPO) · ~5.5 minutes

A complete read-aloud script. **[STAGE: …]** tells you what to show; **SAY:** is the exact words
to read. Pace ~150 words/min. Covers the assignment's required items: the deployed-site training
details (parameters · architecture · tokens · cost), one-click sample questions, the qualitative
performance comparison, **and the single unified site with an independent judge and a judgment
score** — which for this assignment is the SLM Arena.
Report page: **https://reportsite-jade.vercel.app/set2** · Unified judge site: **https://slm-arena-harman.vercel.app**

---

### 0:00–0:45 · Greeting & who I am
**[STAGE: On camera (or with the Set 2 page open) — https://reportsite-jade.vercel.app/set2.]**

**SAY:** "Hello, Dr. Sreedath Panat — sir, thank you again for this bootcamp; the depth of it has
pushed me to build things I didn't think I could. I'm Harman Sandhu, a third-semester Computer
Science student based in the NCR region, and this is the recording of my Assignment 2 video. For
this assignment I took models that could already answer, aligned them two different ways — DPO and
RLAIF — and compared them, so I could see which alignment method works best, and at which model
size. The task is grounded question-answering: the model is given legal or financial documents and
must answer using only those documents, without making things up. I ran both alignment methods
across three sizes — 125 million, 500 million, and the 2-billion Gemma — and this experiment let me
see clearly whether alignment helps, and which method wins as the model gets bigger."

---

### 0:45–1:05 · The models
**[STAGE: Point at the blue “The question” box on the Set 2 page.]**

**SAY:** "Quick setup: every model here I first taught to answer, then aligned two ways — **DPO**,
which learns directly from preferred-versus-rejected answer pairs, and **RLAIF**, which learns
through a separate reward model. I ran both across three sizes — 125 million, 500 million, and the
2-billion Gemma — to see which method wins, and whether that changes with size."

---

### 1:05–1:55 · The deployed model sites — parameters, architecture, tokens, cost
**[STAGE: Open the model grid — https://slm-arena-harman.vercel.app — and click a cell, e.g.
“500M · dpo”, to open that model's own deployed site. Scroll slowly through its stats.]**

**SAY:** "First, the deployed sites the assignment asks me to show. Each aligned model has its own
page. On this one you can see the **trainable parameters**, the full **model architecture**, the
**total tokens across the training epochs**, and the **training cost**. And there are **one-click
sample questions** so anyone can try it instantly. Every one of my seven aligned models has a page
like this, all linked from this arena grid."

---

### 1:55–2:35 · Chart 1 — the scores
**[STAGE: Go to the Set 2 page's first chart, “AI-judged correctness vs word-overlap.” Point at
the amber bars, top to bottom.]**

**SAY:** "Now the comparison. The amber bar is how often the judge rated each model correct. The
same headline that shows up in any size comparison is here too — **size dominates**: my Gemma
models are near 0.9, the 500-million around 0.3 to 0.44, and the 125-million lowest. The grey bars
are the older word-overlap score, which badly under-rates the Gemma models — which is why I don't
rely on it."

---

### 2:35–3:25 · Chart 2 — the flip (the key finding)
**[STAGE: Scroll to the second chart, “Which alignment method wins, by model size.”]**

**SAY:** "This second chart is the real result of the assignment. For each size it shows the gap
between the two alignment methods — a bar to the right means RLAIF won, to the left means DPO won.
And the finding is: **the winning method flips with size.** For my 125-million and 500-million
models, RLAIF wins clearly; for the 2-billion Gemma, DPO wins. There is no universal best method —
it depends on the size. My read: on thin preference data, RLAIF's reward model steadied the small
models, while the simpler DPO pulled ahead at the top."

---

### 3:25–4:00 · The DPO collapse and a fairness note
**[STAGE: Scroll to the “Full scores” table. Point at the “SLM-125M SFT+DPO” row — correctness and
typical length — then at the reward column and its “About the reward column” note.]**

**SAY:** "Two things I have to be honest about. First, DPO didn't just underperform on my smallest
model — it *broke* it: 0.05 correct, answers about two words long. RLAIF on that same model was far
safer. Second, on fairness: you'll see the reward column says ‘omitted’ for the RLAIF rows. That's
deliberate — RLAIF is *trained* to maximise that reward, so grading it with the same reward would
be circular. That's why my headline judge is completely independent of it."

---

### 4:00–5:05 · The unified judge site — the required deliverable
**[STAGE: Open the SLM Arena — https://slm-arena-harman.vercel.app. On the Arena tab pick a
held-out question and click “Ask all 13 & judge” — let the models answer and be scored live. Then
click the Leaderboard tab.]**

**SAY:** "This is the piece the assignment specifically asks for — a single unified site where I can
run inference on all of these models and have one independent judge score them fairly and produce a
judgment score. I pick a held-out question; every model answers live; and a blind judge, handed the
gold answer, scores each response and gives it a number. Watch my 125-million DPO model give a
two-word non-answer while the RLAIF version actually attempts it, and the Gemma models answer fully
— the collapse from the chart, live. And here on the leaderboard is the judge's verdict across all
five hundred held-out questions, every model ranked by its mean score. One note: this leaderboard
uses a richer zero-to-ten rubric — a different scale than the earlier chart — and the caption under
it explains it's the same questions, answers, and judge, just scored a different way."

---

### 5:05–5:35 · Close
**SAY:** "So, sir, what this assignment let me establish: alignment does help, but the best method
flips with model size — RLAIF for my small models, DPO for the large one — and DPO can destabilise
a model that's too small. And because every model is judged by one independent AI, not its own
reward, the comparison stays fair. Thank you for watching."

**[STAGE: Stop.]**

---
**Delivery notes:** the flip chart is the moment — pause on it. Give the unified arena its time;
it's the graded deliverable for this assignment. If asked why the Gemma numbers look high while
their word-overlap is low, point to the “What is the word-overlap score?” page: the Gemma models
paraphrase, which fools word-counting but not the judge. Confirm before recording that the model
site you open visibly shows all four items (parameters, architecture, tokens/epochs, cost).
