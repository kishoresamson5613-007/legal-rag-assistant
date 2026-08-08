# ============================================================
# LLM-AS-JUDGE — scores generated answers on two dimensions
# ============================================================
# Faithfulness:  is every claim in the answer supported by the
#                retrieved context? (catches hallucination)
# Correctness:   does the answer agree with the golden answer?
#
# A larger model (Llama 3.3 70B) judges the 8B generator. Judge
# prompts return strict JSON: {"score": 1-5, "reason": "..."}.
#
# IMPORTANT for the portfolio writeup: an LLM judge is only
# trustworthy once validated. Hand-score ~20 answers yourself and
# check agreement with the judge (see eval/README.md).
# ============================================================

from .llm import JUDGE_MODEL, chat, extract_json

FAITHFULNESS_PROMPT = """You are evaluating a legal question-answering system.

Below are document excerpts that were retrieved, and the answer the system generated.
Judge FAITHFULNESS only: is every factual claim in the answer directly supported by
the excerpts? Ignore whether the answer is complete or well-written.

Score 1-5:
5 = every claim is supported by the excerpts
4 = supported, with at most one minor unsupported detail
3 = mostly supported but contains a claim not found in the excerpts
2 = several claims not found in the excerpts
1 = the answer contradicts the excerpts or is largely fabricated

If the answer correctly says the information is not in the document, score 5.

EXCERPTS:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""

CORRECTNESS_PROMPT = """You are evaluating a legal question-answering system.

Compare the system's answer with the reference answer written by a human.
Judge CORRECTNESS only: does the system's answer convey the same facts and
conclusion as the reference? Extra correct detail is fine. Ignore style.

Score 1-5:
5 = fully agrees with the reference
4 = agrees on the main point, minor omission
3 = partially correct, misses or muddles an important part
2 = mostly incorrect or answers a different question
1 = contradicts the reference

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

SYSTEM ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def judge_faithfulness(client, answer: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(f"[Excerpt {i}]\n{c['text']}" for i, c in enumerate(chunks, 1))
    raw = chat(client, FAITHFULNESS_PROMPT.format(context=context, answer=answer),
               model=JUDGE_MODEL)
    return extract_json(raw)


def judge_correctness(client, question: str, reference: str, answer: str) -> dict:
    raw = chat(client, CORRECTNESS_PROMPT.format(
        question=question, reference=reference, answer=answer), model=JUDGE_MODEL)
    return extract_json(raw)
