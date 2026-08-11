# Evaluation Results

Corpus: `adani_hindenburg_sc_2024.pdf` · 15 answerable + 5 unanswerable questions

| config | chunks | hit@1 | hit@3 | hit@5 | hit@10 | mrr | precision@5 | guard false-alarm | guard catch | faith | correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 105 | 0.4 | 0.5333 | 0.6667 | 0.8 | 0.514 | 0.1467 | 0.0 | 1.0 | — | — |

**Metric notes** — hit@k: fraction of questions whose golden evidence appears in the top-k retrieved chunks. MRR: mean reciprocal rank of the first relevant chunk. Guard false-alarm: answerable questions wrongly flagged low-confidence (lower is better). Guard catch: unanswerable questions correctly flagged (higher is better). Faith/correct: 1-5 LLM-as-judge means (only when run with --judge).