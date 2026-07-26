"""
Unit tests for Risk Analysis Agent schemas.
"""

import pytest
from pydantic import ValidationError

from service.engine.schemas import (
    FinancialMetricsInput,
    ComputedRatios,
    CategoryScores,
    AltmanZScoreResult,
    RiskAnalysisResult,
)


def test_financial_metrics_input_defaults():
    """Verify default values for FinancialMetricsInput."""
    inp = FinancialMetricsInput()
    assert inp.company_name == "Unknown Company"
    assert inp.industry_sector == "General"
    assert inp.reporting_year is None
    assert inp.revenue is None
    assert inp.net_income is None


def test_financial_metrics_input_custom_values():
    """Verify parsing valid metrics dictionary."""
    data = {
        "company_name": "TestCorp",
        "industry_sector": "Technology",
        "reporting_year": 2025,
        "revenue": 5000000.0,
        "net_income": 450000.0,
        "total_debt": 1200000.0,
        "total_equity": 2500000.0,
        "total_assets": 4000000.0,
        "current_assets": 1800000.0,
        "current_liabilities": 900000.0,
        "inventory": 200000.0,
        "ebit": 600000.0,
        "interest_expense": 80000.0,
        "extra_unknown_field": "should_be_ignored"
    }
    inp = FinancialMetricsInput(**data)
    assert inp.company_name == "TestCorp"
    assert inp.industry_sector == "Technology"
    assert inp.revenue == 5000000.0
    assert inp.ebit == 600000.0


def test_risk_analysis_result_validation():
    """Verify validation constraints on RiskAnalysisResult."""
    # Valid instance
    res = RiskAnalysisResult(
        company_name="TestCorp",
        industry_sector="Technology",
        risk_score=85,
        risk_level="Low",
        data_confidence=92.5,
        circuit_breakers_triggered=[],
        category_scores=CategoryScores(solvency=90.0, liquidity=85.0),
        computed_ratios=ComputedRatios(current_ratio=2.0),
        risk_flags=["All ratios healthy"],
        actionable_recommendations=["Maintain current capital structure"]
    )
    assert res.risk_score == 85
    assert res.data_confidence == 92.5
    assert res.computed_ratios.current_ratio == 2.0

    # Invalid risk score (>100)
    with pytest.raises(ValidationError):
        RiskAnalysisResult(
            company_name="TestCorp",
            industry_sector="Technology",
            risk_score=150,  # Must be <= 100
        )

    # Invalid data_confidence (<0)
    with pytest.raises(ValidationError):
        RiskAnalysisResult(
            company_name="TestCorp",
            industry_sector="Technology",
            data_confidence=-5.0,  # Must be >= 0
        )


def test_schema_json_serialization():
    """Verify JSON serialization / deserialization roundtrip."""
    inp = FinancialMetricsInput(
        company_name="Acme Inc",
        revenue=1000.0,
        net_income=150.0
    )
    json_str = inp.model_dump_json()
    reconstructed = FinancialMetricsInput.model_validate_json(json_str)
    assert reconstructed.company_name == "Acme Inc"
    assert reconstructed.revenue == 1000.0
    assert reconstructed.net_income == 150.0
