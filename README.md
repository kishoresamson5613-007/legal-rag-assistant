# Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) system for legal documents. Upload any legal PDF and ask questions in plain English — answers come directly from the document, not from the AI's general knowledge.

## Demo

Upload a legal PDF → Ask a question → Get a grounded answer with source citations.

## How It Works

```
Upload PDF → Extract text → Split into chunks → Embed with sentence-transformers
→ Store in ChromaDB → Ask question → Retrieve relevant chunks → Send to LLM → Answer
```

1. **Chunk** — Document split into 800-character overlapping pieces
2. **Embed** — Each chunk converted to a vector using `all-MiniLM-L6-v2` (runs locally)
3. **Store** — Vectors saved in ChromaDB for fast similarity search
4. **Query** — Question matched against chunks; top 5 sent to `openai/gpt-oss-20b` via Groq
5. **Guard** — If no chunk is relevant enough, warns instead of hallucinating

**Data flow:** PDF parsing, chunking, embedding and retrieval all run on your machine. Generation does not: your question and the top 5 retrieved chunks are sent to Groq's cloud API. Don't upload documents you can't share with a third-party API.

## Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (in-memory) |
| LLM | `openai/gpt-oss-20b` via Groq API |
| UI | Streamlit |
| PDF parsing | pypdf |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires a free [Groq API key](https://console.groq.com) — enter it in the sidebar.

## Features

- Hallucination guard — warns when the document lacks relevant information
- Source citations — shows which chunk each answer came from, with similarity score
- Multi-PDF support — index multiple documents in one session
- No GPU required — embeddings run on CPU

## Evaluation

Retrieval and guard behaviour are measured by `eval/` against a golden set built from `eval/corpus/adani_hindenburg_sc_2024.pdf` (15 answerable + 5 unanswerable questions). Run it with `python -m eval.run_eval --configs baseline rerank`.

| Metric | Baseline | With reranking |
|--------|----------|----------------|
| hit@1 | 0.40 | 0.60 |
| hit@5 | 0.667 | 0.80 |
| MRR | 0.514 | 0.667 |
| precision@5 | 0.147 | 0.187 |

The reranking config retrieves 20 candidates and reorders them with the `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder. It is an eval config only; `app.py` does not rerank yet.

| Hallucination guard | Threshold 0.8 | Threshold 0.41 (current) |
|--------|----------|----------------|
| Catch rate (unanswerable questions flagged) | 0% | 100% |
| False-alarm rate (answerable questions wrongly flagged) | 0% | 0% |

Answer faithfulness and correctness (LLM-as-judge) need `--judge` mode, which hasn't been run yet, so only retrieval metrics are reported.

**Limitation:** the golden set is 20 questions on a single PDF, and the guard threshold was tuned on this same set, so these numbers are optimistic and may not generalise.

## LangChain version

`langchain_version.py` is a parallel reimplementation of the same pipeline using LangChain (`RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, `Chroma`, `ChatGroq`), written to show the design works in LangChain as well as with the plain libraries in `app.py`.
