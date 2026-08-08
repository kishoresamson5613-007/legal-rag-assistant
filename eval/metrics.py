# ============================================================
# RETRIEVAL METRICS — deterministic, no LLM needed
# ============================================================
# The golden test set stores EVIDENCE as a verbatim quote from the
# document, not a chunk ID. Chunk IDs change whenever the chunking
# config changes, so we instead locate the quote's character span in
# the full document once, and a retrieved chunk counts as relevant
# if its span covers enough of the evidence span.
#
# Fallback: if the quote can't be located verbatim (PDF extraction
# quirks, LLM paraphrase), we fall back to token overlap.
# ============================================================

import re

SPAN_COVERAGE_THRESHOLD = 0.5    # chunk must cover ≥50% of the evidence span
TOKEN_OVERLAP_THRESHOLD = 0.7    # fallback: ≥70% of evidence tokens in chunk


# ── Whitespace-insensitive evidence location ─────────────────
def build_norm_map(text: str):
    """Lowercase, collapse whitespace; keep a map back to original offsets."""
    norm_chars = []
    positions = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            norm_chars.append(" ")
            positions.append(i)
            prev_space = True
        else:
            norm_chars.append(ch.lower())
            positions.append(i)
            prev_space = False
    return "".join(norm_chars), positions


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def find_evidence_span(norm_text: str, positions: list[int], evidence: str):
    """Locate the evidence quote in the document. Returns (start, end) or None."""
    needle = normalize(evidence)
    if not needle:
        return None
    idx = norm_text.find(needle)
    if idx == -1:
        return None
    return positions[idx], positions[idx + len(needle) - 1] + 1


def token_overlap(evidence: str, chunk_text: str) -> float:
    ev_tokens = {t for t in re.findall(r"[a-z0-9]+", evidence.lower()) if len(t) > 3}
    if not ev_tokens:
        return 0.0
    chunk_tokens = set(re.findall(r"[a-z0-9]+", chunk_text.lower()))
    return len(ev_tokens & chunk_tokens) / len(ev_tokens)


def chunk_is_relevant(chunk: dict, ev_span, evidence: str) -> bool:
    if ev_span is not None:
        ev_start, ev_end = ev_span
        overlap = min(chunk["end"], ev_end) - max(chunk["start"], ev_start)
        return overlap >= SPAN_COVERAGE_THRESHOLD * (ev_end - ev_start)
    return token_overlap(evidence, chunk["text"]) >= TOKEN_OVERLAP_THRESHOLD


# ── Per-question metrics ─────────────────────────────────────
def score_retrieval(ranked_chunks: list[dict], ev_span, evidence: str,
                    cutoffs=(1, 3, 5, 10)) -> dict:
    """Given the ranked retrieval list for one question, return hit@k, MRR,
    precision@5 based on evidence relevance."""
    relevant = [chunk_is_relevant(c, ev_span, evidence) for c in ranked_chunks]

    first_hit_rank = None
    for rank, is_rel in enumerate(relevant, 1):
        if is_rel:
            first_hit_rank = rank
            break

    scores = {f"hit@{k}": (1.0 if first_hit_rank and first_hit_rank <= k else 0.0)
              for k in cutoffs}
    scores["mrr"] = 1.0 / first_hit_rank if first_hit_rank else 0.0
    top5 = relevant[:5]
    scores["precision@5"] = sum(top5) / len(top5) if top5 else 0.0
    return scores


def aggregate(per_question: list[dict]) -> dict:
    """Mean of each metric over all questions."""
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {k: round(sum(q[k] for q in per_question) / len(per_question), 4)
            for k in keys}
