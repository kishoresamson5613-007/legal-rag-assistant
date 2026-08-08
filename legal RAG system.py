# ============================================================
# LEGAL RAG ASSISTANT
# ============================================================
#
# What is RAG?
# RAG = Retrieval Augmented Generation
#
# Normal AI: Question → AI answers from its training memory
# RAG:       Question → Find relevant parts of YOUR document
#                    → Give those parts to AI → AI answers
#
# Why RAG for legal documents?
# Legal answers must come from the ACTUAL document, not from
# what the AI "thinks" it knows. RAG makes answers trustworthy.
#
# Flow:
#  Upload PDF → Split into chunks → Convert to vectors (numbers)
#  → Store in ChromaDB → User asks question → Find matching chunks
#  → Send to Groq (Llama) → Get grounded answer
#
# ============================================================

import io
import os
import uuid

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq
from pypdf import PdfReader

# ── Configuration ────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"          # Local embedding model — no API needed
GEN_MODEL   = "llama-3.1-8b-instant"      # Groq-hosted Llama — fast & free
COLLECTION  = "legal_docs"                 # ChromaDB collection name

CHUNK_SIZE    = 800   # How many characters per chunk
CHUNK_OVERLAP = 100   # Characters shared between neighbouring chunks

TOP_K = 5             # How many chunks to retrieve per question

# Hallucination detection:
# ChromaDB distance: 0.0 = identical, 1.0 = completely different
# If best chunk distance > 0.6 → document likely doesn't have the answer
LOW_CONFIDENCE_THRESHOLD = 0.8


# ============================================================
# STEP 1 — INITIALISE CHROMADB
# ============================================================
# @st.cache_resource means this runs ONCE and reuses the connection
# every time — avoids reconnecting on every button click.
# ============================================================
@st.cache_resource
def get_embedding_fn():
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def initialise_db():
    # EphemeralClient = in-memory, no disk writes — safe for cloud deployment.
    # One client per session so users don't share each other's documents.
    embedding_fn = get_embedding_fn()
    db_client = chromadb.EphemeralClient()
    doc_collection = db_client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return db_client, doc_collection


# ============================================================
# STEP 2 — LOAD PDF
# ============================================================
def load_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n\n"
    return full_text.strip()


# ============================================================
# STEP 3 — SPLIT TEXT INTO CHUNKS
# ============================================================
# Why chunks? AI models can only handle a limited amount of text.
# We split the document into small overlapping pieces so we can
# find and send only the RELEVANT piece to the AI.
#
# Example: 10-page document → ~40 chunks of 800 characters each
# ============================================================
def split_into_chunks(text: str, filename: str) -> list[dict]:
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "id":          str(uuid.uuid4()),
                "text":        chunk_text,
                "source":      filename,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# STEP 4 — ADD DOCUMENT TO CHROMADB
# ============================================================
EMBED_BATCH_SIZE = 50

def add_document(collection, file_bytes: bytes, filename: str) -> int:
    text   = load_pdf(file_bytes)
    chunks = split_into_chunks(text, filename)

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        collection.add(
            ids       = [c["id"]   for c in batch],
            documents = [c["text"] for c in batch],
            metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]}
                         for c in batch],
        )

    return len(chunks)


# ============================================================
# STEP 5 — RAG QUERY (The Core of the System)
# ============================================================
# 1. Convert question into a vector (local sentence-transformers)
# 2. Search ChromaDB for the most similar document chunks
# 3. Check confidence (hallucination guard)
# 4. Build a prompt = chunks + question
# 5. Send to Groq (Llama) → grounded answer
# ============================================================
def rag_query(collection, groq_client, question: str):
    if collection.count() == 0:
        return "Please upload a PDF document first.", [], False

    results = collection.query(
        query_texts=[question],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # ── Hallucination Guard ──────────────────────────────────
    best_distance  = min(distances)
    low_confidence = best_distance > LOW_CONFIDENCE_THRESHOLD

    # ── Build context block ───────────────────────────────────
    context_block = ""
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
        context_block += f"[Excerpt {i} from: {meta['source']}]\n{chunk}\n\n"

    # ── Build prompt ─────────────────────────────────────────
    prompt = f"""You are a legal document assistant.
Answer ONLY using the document excerpts below. Do not use outside knowledge.
If the answer is not in the excerpts, clearly say so.
Always end your response with: '⚠️ This is not legal advice.'

{context_block}
Question: {question}

Answer:"""

    # Send to Groq — fast cloud inference, free tier
    response = groq_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    answer = response.choices[0].message.content

    sources = [
        {
            "source":      meta["source"],
            "chunk_index": meta["chunk_index"],
            "text":        chunk,
            "similarity":  round(1 - dist, 2),
        }
        for chunk, meta, dist in zip(chunks, metadatas, distances)
    ]

    return answer, sources, low_confidence


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Legal RAG Assistant")
st.caption(
    "Upload a legal PDF (e.g. Constitution, contract, case law) and ask questions. "
    "All answers come directly from your document — no hallucination."
)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:

    # ── API Key ───────────────────────────────────────────────
    st.header("🔑 Groq API Key")
    # Check Streamlit Cloud secrets first, then local env var
    try:
        default_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    except Exception:
        default_key = os.environ.get("GROQ_API_KEY", "")

    api_key = st.text_input(
        "Enter your free Groq API key",
        type="password",
        value=default_key,
        help="Get a free key at console.groq.com",
    )

    if not api_key:
        st.warning("Enter a Groq API key to enable answers.")
        groq_client = None
    else:
        groq_client = Groq(api_key=api_key)
        st.success("API key set.")

    st.divider()

    # ── DB Initialisation ─────────────────────────────────────
    # Store in session_state so each browser tab gets its own isolated DB.
    if "db_client" not in st.session_state:
        try:
            st.session_state.db_client, st.session_state.collection = initialise_db()
        except Exception as e:
            st.error(f"Could not start database: {e}")
            st.stop()
    db_client  = st.session_state.db_client
    collection = st.session_state.collection

    st.header("📄 Upload PDF")

    uploaded = st.file_uploader("Choose a legal PDF file", type=["pdf"])

    if uploaded:
        already_indexed = set()
        if collection.count() > 0:
            all_meta = collection.get(include=["metadatas"])["metadatas"]
            already_indexed = {m["source"] for m in all_meta}

        if uploaded.name not in already_indexed:
            with st.spinner(f"Reading and indexing '{uploaded.name}'..."):
                try:
                    num_chunks = add_document(collection, uploaded.read(), uploaded.name)
                    st.success(f"Done! Indexed {num_chunks} chunks.")
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.info(f"'{uploaded.name}' is already indexed.")

    st.divider()
    st.header("📚 Loaded Documents")

    if collection.count() > 0:
        all_meta  = collection.get(include=["metadatas"])["metadatas"]
        doc_names = sorted({m["source"] for m in all_meta})
        for name in doc_names:
            st.markdown(f"📄 {name}")
        st.caption(f"Total chunks in database: {collection.count()}")

        if st.button("🗑️ Clear All Documents", use_container_width=True):
            db_client.delete_collection(COLLECTION)
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.caption("No documents loaded yet. Upload a PDF above.")

    st.divider()

    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
**5-step RAG pipeline:**

1. **Upload** — PDF text is extracted page by page
2. **Chunk** — Text is split into 800-character pieces
3. **Embed** — Each chunk → vectors via local sentence-transformers
4. **Store** — Vectors saved in ChromaDB (on disk)
5. **Query** — Question matched against chunks → top matches sent to Llama via Groq → answer

**Hallucination Guard:**
If no chunk scores above the similarity threshold, a warning
is shown instead of letting the AI make up an answer.
        """)


# ── Chat Interface ────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    avatar = "👤" if msg["role"] == "user" else "⚖️"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("low_confidence"):
            st.warning(
                "⚠️ Low confidence — the document may not contain enough "
                "relevant information for this question."
            )
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Source Excerpts ({len(msg['sources'])} chunks used)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Excerpt {i}** · `{src['source']}` · "
                        f"chunk #{src['chunk_index']} · "
                        f"similarity **{src['similarity']}**"
                    )
                    preview = src["text"][:300]
                    if len(src["text"]) > 300:
                        preview += "..."
                    st.text(preview)
                    st.divider()

user_question = st.chat_input("Ask a question about your legal document...")

if user_question:
    if not groq_client:
        st.error("Please enter your Groq API key in the sidebar first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.spinner("Searching document and generating answer..."):
            try:
                answer, sources, low_confidence = rag_query(
                    collection, groq_client, user_question
                )
                st.session_state.chat_history.append({
                    "role":           "assistant",
                    "content":        answer,
                    "sources":        sources,
                    "low_confidence": low_confidence,
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role":           "assistant",
                    "content":        f"❌ Error: {str(e)}",
                    "sources":        [],
                    "low_confidence": False,
                })

        st.rerun()
