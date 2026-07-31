"""
SQLite Database Assessment Cache & History Storage engine.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_store")
DB_PATH = os.path.join(DB_DIR, "assessment_cache.db")


def _get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize SQLite tables for assessment history."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                company_name TEXT,
                industry_sector TEXT,
                reporting_year INTEGER,
                risk_score INTEGER,
                risk_level TEXT,
                data_confidence REAL,
                payload_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_assessment_record(session_id: str, result_payload: Dict[str, Any]) -> int:
    """Save completed risk assessment payload to SQLite database."""
    init_db()
    company = result_payload.get("company_name", "Unknown Company")
    sector = result_payload.get("industry_sector", "General")
    year = result_payload.get("reporting_year")
    score = result_payload.get("risk_score")
    level = result_payload.get("risk_level", "Unknown")
    confidence = result_payload.get("data_confidence", 0.0)

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO assessment_history 
            (session_id, company_name, industry_sector, reporting_year, risk_score, risk_level, data_confidence, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, company, sector, year, score, level, confidence, json.dumps(result_payload)))
        conn.commit()
        return cursor.lastrowid


def get_assessment_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve recent assessment history records."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, session_id, company_name, industry_sector, reporting_year, risk_score, risk_level, data_confidence, created_at
            FROM assessment_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def clear_assessment_history():
    """Clear all assessment history records."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM assessment_history")
        conn.commit()
