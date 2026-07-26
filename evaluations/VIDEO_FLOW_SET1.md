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
there, without making things up."

---

### 0:45–1:05 · The models
**[STAGE: Point at the blue “The question” box on the Set 1 page.]**

**SAY:** "I refer to **base** as the model before any fine-tuning, **SFT** is where I taught it
to answer from the documents using QA pairs, and **RAFT** goes further by teaching the model how to refuse when the answer genuinely isn't there. I did this at two sizes — my 125-million model
and the 2-billion Gemma 2 model — so that I could compare the techniques and the sizes together."

---

### 1:05–2:00 · The deployed model sites — parameters, architecture, tokens, cost
**[STAGE: Open the model grid — https://slm-arena-harman.vercel.app — and click a cell, e.g.
“125M · base”, to open that model's own deployed site. Scroll slowly through its stats.]**

**SAY:** "First, I would like to take you through the deployed websites. Every model in this experiment
has its own page. On this site you can see the four things that are required: the **trainable
parameters**; the full **model architecture**; the **total tokens across
the training epochs**; and the **training cost** associated with fine tuning + pre-training
it took. And the **one-click sample questions**, so that anyone can try the model without
typing. Every model — base, SFT and RAFT, at both sizes — has a page exactly like this, all
linked from this arena grid."

---

### 2:00–2:50 · The headline result — size dominates
**[STAGE: Go to the Set 1 page's first chart, “AI-judged correctness vs word-overlap.” Point at
the amber bars.]**

**SAY:** "Now the comparison. The amber bar is how often an
independent AI judge rated each model's answer correct, from zero to one. The Gemma models sit
near the top, around 0.95 to 0.97; whereas my 125-million models got much lower scores. My first finding is a predictable one, size of the gemma model dominates over my 125M models no matter which fine tuning technique i use. The grey bars are the word-overlap score. Word-overlap or F1 scoring grades an answer by how many words it shares with the reference answer — so a correct answer worded differently scores low, which led to unfair grading of the Gemma models as
even though its answers were correct, it wasn't getting a high score, so i switched to AI judging as my main grading criteria."

---

### 2:50–3:40 · The twist — RAFT hurts the small model
**[STAGE: Point at “SLM-125M SFT” then “SLM-125M RAFT.” Then scroll to “How the RAFT models behave
in four situations” and point at the 125M RAFT first row.]**

**SAY:** "However, there is a finding that surprised me alot. For my 125-million model, plain SFT lifts it
from basically zero to 0.26 — but RAFT, the version I taught to refuse, drops it back down to 0.05.
Teaching the small model to be cautious made it worse. This table shows why: on the questions that my model
*does* answer, it still refuses about 84% of the time. It's too small to
tell ‘the answer is missing’ from ‘the answer is right here.’ The Gemma model handles the same
recipe fine — so RAFT is a method whose benefit depends on the size."

---

### 3:40–4:20 · Real examples
**[STAGE: Scroll to the “Real examples” section on the Set 1 page.]**

**SAY:** "I didn't want to only show charts and numbers, so here are some real questions with every fine-tuned
model's actual answer and the judge's score. And after looking at them, the observations are very clear — the Gemma
models answer correctly and completely; while my 125-million SFT model often gives a short or wrong answer and the RAFT version refuses
answers on the very same question. Reading the real outputs makes the comparison clear."

---

### 4:20–5:05 · See it live — the Arena
**[STAGE: Open the SLM Arena's Arena tab — https://slm-arena-harman.vercel.app — pick a held-out
question and click “Ask all 13 & judge.”]**

**SAY:** "And sir, here is my live, deployed SLM arena where you pick a template question or ask a custom question and every model answers
it in real time, then the blind judge scores each one against the real answer. Let me set that up right now.. Ok, While my models are generating their output, I want to highlight that the judge here grades differently than the judge did in my analysis. This judge performs grading based on 4 parameters - correctness, completeness, groundedness and clarity and scores are on a grading scale of 10. Whereas the judge in my initial analysis graded the models from 0 to 1 only on correctness. By changing the grading style I noticed something interesting, my top performing model had changed. On the broader grading criteria, i see that gemma QA slightly beats over the RAFT version whereas in my analysis section it was the opposite. But, the observations regarding my 125 million models stay the same: base and RAFT still get poor scores while the QA model is relatively better, but nowhere near the gemma models. The generation has finished let's take a look at it's results... ."

---

### 5:05–5:35 · Close
**[STAGE: Go back to the Set 1 page — the “Ranking & which differences are real” section.]**

**SAY:** "So, sir, to recap my key findings: for grounded question-answering, QA SFT is what makes a tiny model usable at all; RAFT's refusal-training backfires at the small
size but it is safe at 2 billion; and across the board, model size matters more than the fine tuning technique —
the size gap played the biggest role in the difference of grades between my 125M models and the gemma models. In short — QA SFT works everywhere and RAFT only
pays off once the model is big enough. But sir, this statement left me thinking whether my results are actually correct. Was RAFT actually supposed to hurt my model's performance? or could it a data-balancing problem on my end? Another question I have is that in your experience, is there a rough parameter scale where the fine-tuning technique starts to matter more than the raw model size for grounded QA? Sir I would love to hear your views and opinions on these on discord if possible. That said, getting to this point has taught me more than I expected, and I'm eager to take your feedback and build upon it. Thank you for watching sir, I'm looking forward to continue this conversation with you."

**[STAGE: Stop.]**

---
**Delivery notes:** speak the amber-vs-grey contrast slowly — it's the one visual people misread.
If asked why base-Gemma already scores 0.95, the honest answer: base-Gemma is Google's
*instruction-tuned* Gemma, a strong starting point rather than a blank slate; my 125M base is a
true blank slate, which is why it scores near zero. Confirm before recording that the model site
you open visibly shows all four items (parameters, architecture, tokens/epochs, cost).
