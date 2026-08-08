# ============================================================
# RUN EVALUATION
# ============================================================
# Usage (from the project root):
#
#   python -m eval.run_eval                      # retrieval metrics, all configs (offline)
#   python -m eval.run_eval --configs baseline rerank
#   python -m eval.run_eval --judge              # + generation & LLM-as-judge (needs GROQ_API_KEY)
#   python -m eval.run_eval --judge --limit 5    # smoke test on 5 questions
#
# Outputs:
#   eval/results/<config>.json    full per-question detail
#   eval/results/summary.md       comparison table across all configs run so far
# ============================================================

import argparse
import json
import time
from pathlib import Path

from .configs import CONFIGS, CONFIGS_BY_NAME
from .metrics import (aggregate, build_norm_map, chunk_is_relevant,
                      find_evidence_span, score_retrieval)
from .rag_core import (LOW_CONFIDENCE_THRESHOLD, build_index, generate_answer,
                       load_pdf_text, retrieve)

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_PDF = EVAL_DIR / "corpus" / "adani_hindenburg_sc_2024.pdf"
TESTSET = EVAL_DIR / "testset" / "golden_testset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

METRIC_COLUMNS = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "precision@5"]


def load_testset() -> list[dict]:
    if not TESTSET.exists():
        raise SystemExit(f"Golden test set not found: {TESTSET}\n"
                         "Create it first (see eval/README.md).")
    items = []
    for line in TESTSET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def evaluate_config(config, text, norm_text, positions, testset,
                    judge=False, limit=None) -> dict:
    from .rag_core import get_embedder  # warm the model before timing
    get_embedder(config.embed_model)

    t0 = time.perf_counter()
    collection, chunks = build_index(text, config)
    index_seconds = round(time.perf_counter() - t0, 1)
    print(f"  indexed {len(chunks)} chunks in {index_seconds}s")

    answerable = [q for q in testset if q.get("type", "answerable") == "answerable"]
    unanswerable = [q for q in testset if q.get("type") == "unanswerable"]
    if limit:
        answerable = answerable[:limit]
        unanswerable = unanswerable[: max(1, limit // 3)]

    groq_client = None
    if judge:
        from .llm import get_groq_client
        groq_client = get_groq_client()

    per_question = []
    retrieval_scores = []

    for item in answerable:
        ranked = retrieve(collection, item["question"], config, n_results=10)
        ev_span = find_evidence_span(norm_text, positions, item["evidence"])
        scores = score_retrieval(ranked, ev_span, item["evidence"])
        retrieval_scores.append(scores)

        row = {
            "question": item["question"],
            "evidence_located": ev_span is not None,
            **scores,
        }

        # Hallucination-guard check: answerable questions should NOT be flagged.
        best_distance = min(r["distance"] for r in ranked)
        row["guard_false_alarm"] = best_distance > LOW_CONFIDENCE_THRESHOLD

        if judge:
            from .judge import judge_correctness, judge_faithfulness
            context_chunks = ranked[: config.top_k]
            answer = generate_answer(groq_client, item["question"], context_chunks)
            faith = judge_faithfulness(groq_client, answer, context_chunks)
            correct = judge_correctness(groq_client, item["question"],
                                        item["answer"], answer)
            row.update({
                "generated_answer": answer,
                "faithfulness": faith["score"],
                "faithfulness_reason": faith["reason"],
                "correctness": correct["score"],
                "correctness_reason": correct["reason"],
            })

        per_question.append(row)

    # Unanswerable questions: the guard SHOULD flag these (low confidence).
    guard_rows = []
    for item in unanswerable:
        ranked = retrieve(collection, item["question"], config, n_results=10)
        best_distance = min(r["distance"] for r in ranked)
        guard_rows.append({
            "question": item["question"],
            "flagged": best_distance > LOW_CONFIDENCE_THRESHOLD,
            "best_distance": round(best_distance, 3),
        })

    summary = {
        "config": config.to_dict(),
        "num_chunks": len(chunks),
        "index_seconds": index_seconds,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval": aggregate(retrieval_scores),
        "guard_false_alarm_rate": round(
            sum(r["guard_false_alarm"] for r in per_question) / len(per_question), 4)
            if per_question else None,
        "guard_catch_rate": round(
            sum(r["flagged"] for r in guard_rows) / len(guard_rows), 4)
            if guard_rows else None,
        "evidence_not_located": sum(not r["evidence_located"] for r in per_question),
    }
    if judge and per_question:
        summary["faithfulness_mean"] = round(
            sum(r["faithfulness"] for r in per_question) / len(per_question), 2)
        summary["correctness_mean"] = round(
            sum(r["correctness"] for r in per_question) / len(per_question), 2)

    return {"summary": summary, "per_question": per_question,
            "unanswerable": guard_rows}


def write_summary_table():
    """Rebuild summary.md from every per-config JSON present."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data["summary"]
        rows.append(s)
    if not rows:
        return

    lines = [
        "# Evaluation Results",
        "",
        f"Corpus: `{CORPUS_PDF.name}` · "
        f"{rows[0]['n_answerable']} answerable + {rows[0]['n_unanswerable']} unanswerable questions",
        "",
        "| config | chunks | " + " | ".join(METRIC_COLUMNS) +
        " | guard false-alarm | guard catch | faith | correct |",
        "|" + "---|" * (len(METRIC_COLUMNS) + 6),
    ]
    for s in rows:
        r = s["retrieval"]
        lines.append(
            f"| {s['config']['name']} | {s['num_chunks']} | "
            + " | ".join(f"{r.get(m, '')}" for m in METRIC_COLUMNS)
            + f" | {s.get('guard_false_alarm_rate', '')} "
            + f"| {s.get('guard_catch_rate', '')} "
            + f"| {s.get('faithfulness_mean', '—')} "
            + f"| {s.get('correctness_mean', '—')} |"
        )
    lines += [
        "",
        "**Metric notes** — hit@k: fraction of questions whose golden evidence appears in the "
        "top-k retrieved chunks. MRR: mean reciprocal rank of the first relevant chunk. "
        "Guard false-alarm: answerable questions wrongly flagged low-confidence (lower is better). "
        "Guard catch: unanswerable questions correctly flagged (higher is better). "
        "Faith/correct: 1-5 LLM-as-judge means (only when run with --judge).",
    ]
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", help="config names (default: all)")
    parser.add_argument("--judge", action="store_true",
                        help="also generate answers and run LLM-as-judge (needs GROQ_API_KEY)")
    parser.add_argument("--limit", type=int, help="only evaluate first N questions")
    args = parser.parse_args()

    configs = ([CONFIGS_BY_NAME[n] for n in args.configs]
               if args.configs else CONFIGS)

    text = load_pdf_text(str(CORPUS_PDF))
    norm_text, positions = build_norm_map(text)
    testset = load_testset()
    print(f"Corpus: {len(text)} chars · test set: {len(testset)} questions")

    RESULTS_DIR.mkdir(exist_ok=True)
    for config in configs:
        print(f"\n=== {config.name} ===")
        result = evaluate_config(config, text, norm_text, positions, testset,
                                 judge=args.judge, limit=args.limit)
        out = RESULTS_DIR / f"{config.name}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"  retrieval: {result['summary']['retrieval']}")

    write_summary_table()
    print(f"\nSummary table written to {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
