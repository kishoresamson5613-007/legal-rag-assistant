# ============================================================
# SYNTHETIC TEST-SET GENERATOR (optional — needs GROQ_API_KEY)
# ============================================================
# The starter golden test set was hand-written. Use this script to
# SCALE it up: an LLM reads evenly-spaced excerpts of the document
# and drafts (question, answer, verbatim evidence quote) triples.
#
#   python -m eval.generate_testset --num 40
#
# Output: eval/testset/candidates.jsonl
#
# THEN REVIEW BY HAND. Delete bad questions, fix answers, and append
# the survivors to eval/testset/golden_testset.jsonl. Synthetic
# questions that skip human review make the eval meaningless —
# mention this curation step in interviews, it matters.
# ============================================================

import argparse
import json
from pathlib import Path

from .llm import JUDGE_MODEL, chat, extract_json, get_groq_client
from .metrics import build_norm_map, find_evidence_span
from .rag_core import load_pdf_text, split_into_chunks

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_PDF = EVAL_DIR / "corpus" / "adani_hindenburg_sc_2024.pdf"
OUT = EVAL_DIR / "testset" / "candidates.jsonl"

PROMPT = """You are building an evaluation set for a legal document Q&A system.

Below is an excerpt from a Supreme Court of India judgment. Write ONE question
that a lawyer or journalist might genuinely ask, which this excerpt answers.

Rules:
- The question must be answerable from THIS excerpt alone.
- Do not mention "the excerpt" or "the passage" in the question.
- "evidence" must be an EXACT verbatim quote from the excerpt (max 250
  characters) that contains the answer. Copy it character-for-character.
- "answer" is a concise 1-2 sentence answer in your own words.

EXCERPT:
{chunk}

Respond with ONLY a JSON object:
{{"question": "...", "answer": "...", "evidence": "..."}}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=30,
                        help="number of candidate questions to draft")
    args = parser.parse_args()

    text = load_pdf_text(str(CORPUS_PDF))
    norm_text, positions = build_norm_map(text)

    # Wide, non-overlapping chunks spread evenly across the document,
    # so questions cover the whole judgment rather than one section.
    chunks = split_into_chunks(text, chunk_size=1500, chunk_overlap=0)
    stride = max(len(chunks) // args.num, 1)
    picked = chunks[::stride][: args.num]

    client = get_groq_client()
    kept, dropped = 0, 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(picked, 1):
            try:
                raw = chat(client, PROMPT.format(chunk=chunk["text"]),
                           model=JUDGE_MODEL, temperature=0.3)
                item = extract_json(raw)
            except Exception as e:
                print(f"  [{i}/{len(picked)}] generation failed: {e}")
                dropped += 1
                continue

            # Reject if the "verbatim" quote can't actually be found.
            if find_evidence_span(norm_text, positions, item.get("evidence", "")) is None:
                print(f"  [{i}/{len(picked)}] dropped — evidence not verbatim")
                dropped += 1
                continue

            item["type"] = "answerable"
            item["origin"] = "synthetic"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            print(f"  [{i}/{len(picked)}] {item['question'][:70]}")

    print(f"\nKept {kept}, dropped {dropped}. Review {OUT} by hand, then append "
          "good ones to golden_testset.jsonl")


if __name__ == "__main__":
    main()
