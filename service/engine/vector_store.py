from typing import Dict, List, Optional, Any
from rank_bm25 import BM25Okapi

# Dictionary mapping session_id -> {"docs": list[str], "bm25": BM25Okapi | None}
_stores: Dict[str, Dict[str, Any]] = {}
DEFAULT_SESSION_ID = "default"


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def embed_chunks(chunks):
    return [[] for _ in chunks]


def clear_store(session_id: str = DEFAULT_SESSION_ID) -> None:
    """Clear document chunks and index for a specific session."""
    if session_id in _stores:
        del _stores[session_id]


def add_to_store(chunks: List[str], _=None, session_id: str = DEFAULT_SESSION_ID) -> None:
    """Append chunks to a session's vector store and re-index."""
    if session_id not in _stores:
        _stores[session_id] = {"docs": [], "bm25": None}
    
    docs = _stores[session_id]["docs"]
    docs.extend(chunks)
    _stores[session_id]["bm25"] = BM25Okapi([_tokenize(d) for d in docs]) if docs else None


def store_chunks(chunks: List[str], session_id: str = DEFAULT_SESSION_ID, clear_existing: bool = True) -> int:
    """
    Store chunks for a specific session.
    By default, clear_existing=True clears any previous document chunks for this session
    to prevent document chunk leakage between uploads.
    """
    if clear_existing:
        clear_store(session_id)
    add_to_store(chunks, session_id=session_id)
    return len(chunks)


def search_chunks(query: str, n: int = 3, session_id: str = DEFAULT_SESSION_ID) -> List[str]:
    """Search relevant document chunks within a specific session's vector store."""
    store = _stores.get(session_id)
    if not store or not store.get("docs") or store.get("bm25") is None:
        # Fallback to default session if specified session is not found
        if session_id != DEFAULT_SESSION_ID and DEFAULT_SESSION_ID in _stores:
            store = _stores[DEFAULT_SESSION_ID]
        else:
            return []

    docs = store["docs"]
    bm25 = store["bm25"]
    if not docs or bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:min(n, len(docs))]
    return [docs[i] for i in top]


def get_store_stats(session_id: str = DEFAULT_SESSION_ID) -> Dict[str, Any]:
    """Return stats for a session's store."""
    store = _stores.get(session_id)
    return {
        "session_id": session_id,
        "doc_count": len(store["docs"]) if store else 0,
        "has_index": store is not None and store.get("bm25") is not None
    }

