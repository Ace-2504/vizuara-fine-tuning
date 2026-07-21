"""Teacher prompt builders + JSON schemas for every recipe."""
from __future__ import annotations
import config as C

# ---------------- schemas ----------------
QA_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "question": {"type": "string"}, "quote": {"type": "string"},
    "answer": {"type": "string"}, "difficulty": {"type": "string"}},
    "required": ["question", "quote", "answer", "difficulty"]}}

AUX_SCHEMA = {"type": "object", "properties": {
    "instruction": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["instruction", "answer"]}

UNANSWERABLE_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "question": {"type": "string"}}, "required": ["question"]}}

EVOLVE_SCHEMA = {"type": "object", "properties": {
    "question": {"type": "string"}, "quote": {"type": "string"},
    "answer": {"type": "string"}}, "required": ["question", "quote", "answer"]}

JUDGE_SCHEMA = {"type": "object", "properties": {
    "correct": {"type": "integer"}, "grounded": {"type": "boolean"},
    "reason": {"type": "string"}}, "required": ["correct", "grounded", "reason"]}


# ---------------- generation ----------------
def qa(passage: str, n: int = C.QA_PER_CHUNK) -> str:
    return (f"Read the PASSAGE and write {n} diverse, high-quality question-answer pairs.\n"
            "Each item has:\n"
            '- "question": self-contained; MUST NOT say "the passage/text/document/above"; '
            "name the company, case, or topic so it stands alone.\n"
            '- "quote": the EXACT verbatim span from the passage supporting the answer, '
            "character-for-character. No paraphrase, no ellipses.\n"
            '- "answer": short, direct, fully supported by the quote.\n'
            '- "difficulty": "easy" | "medium" | "hard".\n'
            "Rules: vary difficulty; do NOT invent facts; no near-duplicate questions.\n\n"
            f"PASSAGE:\n{passage}")


_AUX = {
    "summarize": ('Write ONE faithful, concise summary of the TEXT. Return {"instruction","answer"} '
                  'where instruction asks for a summary (self-contained) and answer is the summary. '
                  "Do NOT add facts not in the TEXT."),
    "extract": ('Extract the key named fields from the TEXT into a compact JSON object rendered as '
                'a string. Return {"instruction","answer"} where instruction asks to extract the '
                "fields and answer is the JSON string. Only fields present in the TEXT."),
    "rewrite": ('Rewrite the TEXT in plain, clear English without changing its meaning or adding '
                'facts. Return {"instruction","answer"} where instruction asks for the rewrite and '
                "answer is the rewritten text."),
}
def aux(task: str, passage: str) -> str:
    return f"{_AUX[task]}\n\nTEXT:\n{passage}"


def unanswerable(passage: str, n: int = 3) -> str:
    return (f"Read the PASSAGE and write {n} on-topic questions that the passage does NOT answer "
            "(the specific fact is genuinely absent). Plausible and on-topic, not trivially "
            'unrelated. Return a list of {"question"}.\n\n'
            f"PASSAGE:\n{passage}")


def evolve(passage: str, question: str) -> str:
    return ("Rewrite the QUESTION into ONE harder version — add a constraint, edge case, or "
            "multi-step reasoning — still answerable ONLY from the passage. Return "
            '{"question","quote","answer"} with an EXACT verbatim quote from the passage.\n\n'
            f"PASSAGE:\n{passage}\n\nQUESTION: {question}")


# ---------------- judge ----------------
def judge(passage: str, question: str, answer: str) -> str:
    return ("Grade a synthetic training pair using ONLY the PASSAGE.\n"
            "- correct (1-5): does the ANSWER correctly and completely answer the QUESTION?\n"
            "- grounded (bool): is every claim in the ANSWER supported by the PASSAGE?\n"
            "Be strict; a fluent but unsupported or incomplete answer scores low.\n\n"
            f"PASSAGE:\n{passage}\n\nQUESTION: {question}\nANSWER: {answer}")
