# ============================================================
# RAG CORE — headless, config-driven version of the pipeline
# ============================================================
# app.py is the Streamlit UI with one hard-coded configuration.
# This module exposes the same pipeline (load → chunk → embed →
# retrieve → generate) as plain functions driven by a RagConfig,
# so experiments can run without the UI and compare settings.
#
# Key difference vs app.py: chunks carry their character offsets
# (start, end) into the full document text. The golden test set
# references evidence by text span, so offsets let us score
# retrieval no matter how the chunking config slices the document.
# ============================================================

import io
from dataclasses import dataclass, asdict

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

GEN_MODEL = "llama-3.1-8b-instant"

# BGE models want a prefix on the QUERY side only (not on documents).
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

LOW_CONFIDENCE_THRESHOLD = 0.41  # same hallucination-guard threshold as app.py


@dataclass(frozen=True)
class RagConfig:
    name: str
    chunk_size: int = 800
    chunk_overlap: int = 100
    embed_model: str = "all-MiniLM-L6-v2"
    use_reranker: bool = False
    rerank_candidates: int = 20   # retrieve this many, rerank, keep the top ones
    top_k: int = 5                # chunks passed to the LLM at generation time

    def to_dict(self):
        return asdict(self)


# ── Model caches (embedders and the cross-encoder are slow to load) ──
_embedders: dict[str, SentenceTransformer] = {}
_reranker = None


def get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _embedders:
        _embedders[model_name] = SentenceTransformer(model_name)
    return _embedders[model_name]


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# ── Load PDF (same extraction as app.py) ─────────────────────
def load_pdf_text(path: str) -> str:
    with open(path, "rb") as f:
        reader = PdfReader(io.BytesIO(f.read()))
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n\n"
    return full_text.strip()


# ── Chunking with character offsets ──────────────────────────
def split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunks = []
    start = 0
    index = 0
    step = max(chunk_size - chunk_overlap, 1)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": f"chunk_{index}",
                "text": chunk_text,
                "start": start,
                "end": end,
            })
            index += 1
        start += step

    return chunks


# ── Index build ──────────────────────────────────────────────
def build_index(text: str, config: RagConfig):
    """Chunk the document and load it into a fresh in-memory collection.

    Embeddings are computed here (not by Chroma) so we control the
    model and can apply query-side prefixes at retrieval time.
    Returns (collection, chunks).
    """
    chunks = split_into_chunks(text, config.chunk_size, config.chunk_overlap)
    embedder = get_embedder(config.embed_model)
    embeddings = embedder.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="eval_docs",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[{"start": c["start"], "end": c["end"]} for c in chunks],
    )
    return collection, chunks


# ── Retrieval (optionally reranked) ──────────────────────────
def retrieve(collection, question: str, config: RagConfig, n_results: int = 10) -> list[dict]:
    """Return the ranked chunk list for a question.

    Always returns up to `n_results` chunks so metrics can be computed
    at several cutoffs (hit@1/3/5/10) from a single retrieval call.
    """
    embedder = get_embedder(config.embed_model)
    prefix = QUERY_PREFIXES.get(config.embed_model, "")
    query_emb = embedder.encode([prefix + question], normalize_embeddings=True)

    fetch_n = config.rerank_candidates if config.use_reranker else n_results
    fetch_n = min(fetch_n, collection.count())

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    ranked = [
        {
            "text": doc,
            "start": meta["start"],
            "end": meta["end"],
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

    if config.use_reranker and ranked:
        reranker = get_reranker()
        scores = reranker.predict([(question, r["text"]) for r in ranked])
        for r, s in zip(ranked, scores):
            r["rerank_score"] = float(s)
        ranked.sort(key=lambda r: r["rerank_score"], reverse=True)

    return ranked[:n_results]


# ── Generation (same prompt as app.py) ───────────────────────
def build_prompt(question: str, chunks: list[dict]) -> str:
    context_block = ""
    for i, chunk in enumerate(chunks, 1):
        context_block += f"[Excerpt {i}]\n{chunk['text']}\n\n"

    return f"""You are a legal document assistant.
Answer ONLY using the document excerpts below. Do not use outside knowledge.
If the answer is not in the excerpts, clearly say so.
Always end your response with: '⚠️ This is not legal advice.'

{context_block}
Question: {question}

Answer:"""


def generate_answer(groq_client, question: str, chunks: list[dict]) -> str:
    response = groq_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        temperature=0.1,
    )
    return response.choices[0].message.content
