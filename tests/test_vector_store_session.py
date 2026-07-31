"""
Unit tests for vector_store session and document isolation.
"""

import pytest
from service.engine.vector_store import (
    store_chunks,
    search_chunks,
    clear_store,
    get_store_stats,
    _stores,
)


def setup_function():
    """Clear all vector store sessions before each test."""
    _stores.clear()


def test_session_isolation_separate_documents():
    """Verify that document chunks in session_a do not leak into session_b."""
    session_a_chunks = [
        "Tesla revenue reached 81 billion dollars with strong automotive margins.",
        "Tesla EV delivery target surpassed expected figures in Q4."
    ]
    session_b_chunks = [
        "Microsoft Azure cloud revenue increased by 30 percent year over year.",
        "Microsoft surface hardware sales experienced modest seasonal growth."
    ]

    store_chunks(session_a_chunks, session_id="session_tesla")
    store_chunks(session_b_chunks, session_id="session_msft")

    # Search session_tesla
    res_a = search_chunks("automotive delivery", session_id="session_tesla")
    assert len(res_a) > 0
    assert any("Tesla" in chunk for chunk in res_a)
    assert not any("Microsoft" in chunk for chunk in res_a)

    # Search session_msft
    res_b = search_chunks("cloud revenue", session_id="session_msft")
    assert len(res_b) > 0
    assert any("Microsoft" in chunk for chunk in res_b)
    assert not any("Tesla" in chunk for chunk in res_b)


def test_reupload_clears_previous_session_chunks():
    """Verify that uploading a new document to the same session clears previous chunks."""
    old_chunks = ["Old Document: Company A reported 10 million in debt."]
    new_chunks = ["New Document: Company A cleared all liabilities and has zero debt."]

    store_chunks(old_chunks, session_id="user_session_1", clear_existing=True)
    stats_1 = get_store_stats(session_id="user_session_1")
    assert stats_1["doc_count"] == 1

    # Overwrite session with new document
    store_chunks(new_chunks, session_id="user_session_1", clear_existing=True)
    stats_2 = get_store_stats(session_id="user_session_1")
    assert stats_2["doc_count"] == 1

    results = search_chunks("debt", session_id="user_session_1")
    assert len(results) == 1
    assert "cleared all liabilities" in results[0]
    assert "10 million in debt" not in results[0]


def test_clear_store_isolated_by_session():
    """Verify clear_store removes only target session."""
    store_chunks(["Doc A"], session_id="sess_a")
    store_chunks(["Doc B"], session_id="sess_b")

    clear_store("sess_a")

    assert get_store_stats("sess_a")["doc_count"] == 0
    assert get_store_stats("sess_b")["doc_count"] == 1
    assert len(search_chunks("Doc B", session_id="sess_b")) == 1
