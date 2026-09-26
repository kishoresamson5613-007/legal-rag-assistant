# Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) system for legal documents. Upload any legal PDF and ask questions in plain English — answers come directly from the document, not from the AI's general knowledge.

## Live Demo

**[Try it on Streamlit Cloud →](https://legal-rag-assistant-usp7kh8qjt5bruawxfygz4.streamlit.app)**

Upload a legal PDF → Ask a question → Get a grounded answer with source citations.

You'll need a free [Groq API key](https://console.groq.com) to ask questions — enter it in the sidebar.

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

If you'll be committing, enable the repo's git hooks once per clone (git doesn't turn them on automatically):

```bash
git config core.hooksPath .githooks
```

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

### Why LangChain

The hand-built chunker in `eval/rag_core.py` (`split_into_chunks`) cuts the text at fixed character positions. Every chunk is exactly 800 characters, and each new one starts 700 characters after the last one. That's simple and predictable, but it ignores the content. Chunks often end mid-word or mid-sentence, so a fact can be split across two chunks and neither one holds all of it. LangChain's `RecursiveCharacterTextSplitter` tries a list of break points in order: paragraph breaks (`\n\n`), then line breaks (`\n`), then spaces, and only as a last resort single characters. It then combines those pieces into chunks of up to 800 characters, so chunks usually end at a paragraph, line or word boundary. That's also why the same settings (`chunk_size=800`, `chunk_overlap=100`) gave 123 chunks instead of 105 on the same PDF. For LangChain, 800 is a maximum, not a fixed length. When adding the next paragraph or line would go over the limit, the chunk ends early at the last clean break. PDF text has many line breaks, so this happens often. Most chunks come out shorter than 800 characters, and more of them are needed to cover the same document. The extra chunks are the cost of keeping each chunk intact enough to be retrieved and read on its own.
