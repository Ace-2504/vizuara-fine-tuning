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
genuinely pushed me to build things I didn't think I could. My name is Harman Sandhu, I'm a
third-semester Computer Science student based in the NCR region, and this is the recording of my
Assignment 2 video. For this assignment I took models that could already answer, then aligned them
two different ways — **DPO** and **RLAIF** — and compared them head-to-head, so I could see with real
evidence which alignment method works best, and at which model size. The task is grounded
question-answering: each model is handed a set of legal or financial documents and must answer using
only those documents, without making things up."

---

### 0:45–1:05 · The models
**[STAGE: Point at the blue “The question” box on the Set 2 page.]**

**SAY:** "Every model here I first taught to answer using QA SFT, and then aligned two
ways. **DPO** learns directly from pairs of answers — a preferred one and a rejected one — and pulls
the model toward the preferred kind. **RLAIF** goes about it differently: it learns through a separate
reward model that scores the answers. I ran both of these across three sizes — my 125-million model,
your 500-million base model, and the 2-billion Gemma model — so that I could compare the methods and the
sizes together, and see whether the winner changes as the model gets bigger."

---

### 1:05–1:55 · The deployed model sites — parameters, architecture, tokens, cost
**[STAGE: Open the model grid — https://slm-arena-harman.vercel.app — and click a cell, e.g.
“500M · dpo”, to open that model's own deployed site. Scroll slowly through its stats.]**

**SAY:** "First, I would like to take you through the deployed websites. Every model in this experiment has its own page. On this site you can see the four things that are required: the trainable parameters; the full model architecture; the total tokens across the training epochs; and the training cost associated with fine tuning + pre-training it took. And the one-click sample questions, so that anyone can try the model without typing. Every model has a page exactly like this, all linked from this arena grid."

---

### 1:55–2:35 · Chart 1 — the scores
**[STAGE: Go to the Set 2 page's first chart, “AI-judged correctness vs word-overlap.” Point at
the amber bars, top to bottom.]**

**SAY:** "Now the comparison. The amber bar is how often an independent AI judge rated each model's
answer correct, from zero to one. And my first finding is the predictable one — size dominates. My
Gemma models sit near the top, around 0.9; my 500-million models land in the middle, roughly 0.3 to
0.44; and my 125-million models are the lowest, no matter which alignment method I used. The grey bars
are the word-overlap score. Word-overlap grades an answer by how many words it shares with the
reference answer — so a correct answer worded differently scores low, which unfairly graded the
Gemma models even when their answers are right. That is exactly why I switched to AI judging as my main
grading criteria."

---

### 2:35–3:25 · Chart 2 — the flip (the key finding)
**[STAGE: Scroll to the second chart, “Which alignment method wins, by model size.”]**

**SAY:** "This second chart is the real result of the assignment, and it surprised me. For each size it
shows the gap between the two alignment methods — a bar to the right means RLAIF won, and a bar to the
left means DPO won. And the finding is this: the winning method flips with size. For my 125-million and
my 500-million models, RLAIF wins clearly; but for the 2-billion Gemma, DPO wins. So there is no single
best alignment method — it depends on the size of the model. My reading of it is that on thin preference
data, RLAIF's separate reward model steadied my small models and kept them on track, while the simpler
DPO pulled ahead once the model was big enough to handle it directly."

---

### 3:25–4:05 · The DPO collapse and a fairness note
**[STAGE: Scroll to the “Full scores” table. Point at the “SLM-125M SFT+DPO” row — correctness and
typical length — then at the reward column and its “About the reward column” note. Point at an RLAIF
row (“omitted”).]**

**SAY:** "There are two things I want to highlight, sir. First, on my smallest model, DPO didn't
just underperform — it broke it: 0.05 correct, with answers only about two words long. RLAIF on that
very same model was far safer. Second, on fairness — you'll notice the reward column says ‘omitted’ for
every RLAIF row, and that is deliberate. RLAIF is *trained* to maximise that reward, so grading it with
the same reward would be circular and would flatter it unfairly. And one more honest note about this
column: my reward model was built on my 500-million model, so its scores are only truly comparable
inside that family — which is exactly why my headline judge is a completely independent AI judge, with
no connection to that reward at all."

---

### 4:05–5:05 · The unified judge site — the required deliverable
**[STAGE: Open the SLM Arena — https://slm-arena-harman.vercel.app. On the Arena tab pick a
held-out question and click “Ask all 13 & judge” — let the models answer and be scored live. Then
click the Leaderboard tab.]**

**SAY:** "And sir, this is the piece the assignment specifically asks for — a single unified site where
I can run every one of these models and have one independent judge score them fairly and produce a
judgment score. Let me set that up right now.. I pick a held-out question, every model answers it live,
and a blind judge — handed the real answer — scores each response and gives it a number. While they
generate, watch what happens: my 125-million DPO model gives a two-word non-answer, the RLAIF version
actually attempts it, and the Gemma models answer fully — the collapse from the chart, happening live.
And here on the leaderboard is the judge's verdict across all five hundred held-out questions, every
model ranked by its mean score. One honest note: this leaderboard grades on a richer zero-to-ten rubric
— correctness, completeness, groundedness and clarity — a different scale than the earlier chart, which
scored zero-to-one on correctness alone. The caption underneath explains it's the same questions, the
same answers, and the same judge, just scored a different way."

---

### 5:05–5:35 · Close
**[STAGE: Go back to the Set 2 page — the leaderboard / summary section.]**

**SAY:** "So, sir, to recap my key findings: alignment does help, but the best method flips with model
size — RLAIF for my small models, DPO for the large one — and DPO can destabilise a model that's simply
too small, as it did to my 125-million one. And because every model is judged by one independent AI, not
by its own reward, the comparison stays fair. But this result left me with a question I keep turning
over: is the method flip something fundamental about how these two techniques scale, or is it a sign that
my small models just didn't have enough clean preference data for DPO to work with? And in your
experience, sir, is there a rough size at which DPO reliably becomes the safer choice over RLAIF for
grounded question-answering? I would love to hear your views and opinions on these on discord if
possible. That said, getting to this point has taught me more than I expected, and I'm eager to take
your feedback and build upon it. Thank you for watching sir, I'm looking forward to continue this
conversation with you."

**[STAGE: Stop.]**

---
**Delivery notes:** the flip chart is the moment — pause on it and let it land, since it's the whole
result of the assignment. Give the unified arena its time too; it's the graded deliverable. Speak the
amber-vs-grey contrast on the first chart slowly — it's the one visual people misread. If asked why the
Gemma numbers look high while their word-overlap is low, the honest answer: the Gemma models paraphrase,
which fools word-counting but not the AI judge. If asked about the reward column: there is only one
reward model, built on the 500M model; every non-RLAIF row has a score (the Gemma DPO one was computed
post-hoc on the local GPU with the exact same scorer, validated against the cloud runs), and “omitted”
for RLAIF means it was trained on that same reward, so scoring it there proves nothing. Confirm before
recording that the model site you open visibly shows all four items (parameters, architecture,
tokens/epochs, cost).
