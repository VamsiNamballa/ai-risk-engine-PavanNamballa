"""
Unit tests for SQLite database assessment history engine.
"""

import pytest
from service.engine.db_store import (
    init_db,
    save_assessment_record,
    get_assessment_history,
    clear_assessment_history,
)


def setup_function():
    """Clear database records before each test."""
    init_db()
    clear_assessment_history()


def test_save_and_retrieve_assessment_history():
    """Verify saving risk assessment records and retrieving history."""
    payload_1 = {
        "company_name": "Acme Corp",
        "industry_sector": "Manufacturing",
        "reporting_year": 2025,
        "risk_score": 75,
        "risk_level": "Moderate",
        "data_confidence": 90.0,
    }
    payload_2 = {
        "company_name": "Beta Tech",
        "industry_sector": "Technology",
        "reporting_year": 2026,
        "risk_score": 88,
        "risk_level": "Low",
        "data_confidence": 95.0,
    }

    record_id_1 = save_assessment_record("session_101", payload_1)
    record_id_2 = save_assessment_record("session_102", payload_2)

    assert record_id_1 > 0
    assert record_id_2 > record_id_1

    history = get_assessment_history(limit=10)
    assert len(history) == 2

    # Verify order is most recent first
    assert history[0]["company_name"] == "Beta Tech"
    assert history[1]["company_name"] == "Acme Corp"


def test_clear_assessment_history():
    """Verify clearing all history records."""
    save_assessment_record("sess", {"company_name": "Test Co", "risk_score": 50})
    assert len(get_assessment_history()) == 1

    clear_assessment_history()
    assert len(get_assessment_history()) == 0
