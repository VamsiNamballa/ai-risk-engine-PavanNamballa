"""
Unit tests for sector-specific ratio benchmarking and metric extraction fallbacks.
"""

import pytest
from service.engine.schemas import FinancialMetricsInput
from service.engine.scoring_engine import _score_ratio_continuous, analyze_financial_risk
from service.engine.risk_calculator import _extract_via_regex, extract_financial_metrics


def test_sector_ratio_benchmarking():
    """Verify that technology sector benchmark applies different target bounds than general."""
    # Current Ratio = 2.5
    # General bounds: min=0.8, max=2.0 -> >= 2.0 gets 100 pts
    score_gen = _score_ratio_continuous("current_ratio", 2.5, sector="General")
    assert score_gen == 100.0

    # Tech bounds: min=1.2, max=3.0 -> 2.5 is interpolated within (1.2, 3.0) range
    score_tech = _score_ratio_continuous("current_ratio", 2.5, sector="Technology")
    assert 50.0 < score_tech < 100.0


def test_regex_metric_extraction_fallback():
    """Verify regex fallback extracts revenue and net income from raw document text."""
    sample_text = """
    Financial Overview:
    Total Revenue: $12,500,000
    Net Income: $1,250,000
    Total Debt: $3,000,000
    Total Assets: $15,000,000
    """
    extracted = _extract_via_regex(sample_text)
    assert extracted["revenue"] == 12500000.0
    assert extracted["net_income"] == 1250000.0
    assert extracted["total_debt"] == 3000000.0
    assert extracted["total_assets"] == 15000000.0
