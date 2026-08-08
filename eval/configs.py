# ============================================================
# EXPERIMENT MATRIX
# ============================================================
# One-factor-at-a-time sweeps around the production baseline
# (app.py: 800 chars / 100 overlap / MiniLM / no reranker).
# Comparing each config against `baseline` isolates the effect
# of a single change — cleaner conclusions than a full grid.
# ============================================================

from .rag_core import RagConfig

CONFIGS = [
    # Production settings as shipped in app.py
    RagConfig(name="baseline",
              chunk_size=800, chunk_overlap=100),

    # ── Chunking sweeps ──────────────────────────────────────
    RagConfig(name="chunk_400",
              chunk_size=400, chunk_overlap=50),
    RagConfig(name="chunk_1600",
              chunk_size=1600, chunk_overlap=200),
    RagConfig(name="no_overlap",
              chunk_size=800, chunk_overlap=0),

    # ── Embedding model sweep ────────────────────────────────
    # bge-small is a stronger retrieval model at a similar size.
    # First run downloads ~130 MB from HuggingFace.
    RagConfig(name="embed_bge",
              chunk_size=800, chunk_overlap=100,
              embed_model="BAAI/bge-small-en-v1.5"),

    # ── Reranker sweep ───────────────────────────────────────
    # Retrieve 20 candidates with MiniLM, rerank with a cross-encoder.
    RagConfig(name="rerank",
              chunk_size=800, chunk_overlap=100,
              use_reranker=True, rerank_candidates=20),

    # ── Best-of combination (edit after seeing sweep results) ─
    RagConfig(name="bge_plus_rerank",
              chunk_size=800, chunk_overlap=100,
              embed_model="BAAI/bge-small-en-v1.5",
              use_reranker=True, rerank_candidates=20),
]

CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}
