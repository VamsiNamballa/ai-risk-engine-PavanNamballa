"""
Unit tests for scoring_engine and circuit_breakers.
"""

import pytest
from service.engine.schemas import FinancialMetricsInput
from service.engine.scoring_engine import (
    analyze_financial_risk,
    compute_category_scores,
    _score_ratio_continuous,
)
from service.engine.circuit_breakers import evaluate_circuit_breakers
from service.engine.ratio_engine import calculate_ratios
from service.engine.distress_models import calculate_altman_z_score


def test_continuous_ratio_scoring():
    """Verify linear interpolation ratio scoring."""
    # Current Ratio: min=0.8 (0 pts), max=2.0 (100 pts)
    assert _score_ratio_continuous("current_ratio", 0.5) == 0.0
    assert _score_ratio_continuous("current_ratio", 2.5) == 100.0
    assert _score_ratio_continuous("current_ratio", 1.4) == 50.0

    # Debt-to-Equity (lower is better): min=0.5 (100 pts), max=3.0 (0 pts)
    assert _score_ratio_continuous("debt_to_equity", 0.3) == 100.0
    assert _score_ratio_continuous("debt_to_equity", 3.5) == 0.0
    assert _score_ratio_continuous("debt_to_equity", 1.75) == 50.0


def test_healthy_company_analysis():
    """Verify end-to-end analysis for a healthy company."""
    healthy_input = FinancialMetricsInput(
        company_name="TechCorp",
        industry_sector="Technology",
        revenue=10000000.0,
        net_income=1500000.0,
        total_debt=2000000.0,
        total_equity=6000000.0,
        total_assets=8000000.0,
        current_assets=4000000.0,
        current_liabilities=1500000.0,
        inventory=500000.0,
        ebit=2000000.0,
        interest_expense=100000.0,
        ebitda=2400000.0,
        operating_cash_flow=1800000.0,
        retained_earnings=3000000.0,
        market_val_equity=9000000.0,
    )

    result = analyze_financial_risk(healthy_input)

    assert result.company_name == "TechCorp"
    assert result.risk_score is not None
    assert result.risk_score >= 80
    assert result.risk_level == "Low"
    assert result.circuit_breakers_triggered == []
    assert result.category_scores.solvency is not None
    assert result.category_scores.solvency > 80.0
    assert result.altman_z_score.zone == "Safe"
    assert result.data_confidence == 100.0


def test_circuit_breaker_insolvency():
    """Verify that Interest Coverage < 1.0 caps the risk score to Critical."""
    insolvent_input = FinancialMetricsInput(
        company_name="Insolvent Co",
        revenue=5000000.0,
        net_income=100000.0,
        total_debt=2000000.0,
        total_equity=1000000.0,
        total_assets=3000000.0,
        current_assets=1500000.0,
        current_liabilities=1000000.0,
        ebit=50000.0,             # EBIT 50K
        interest_expense=100000.0, # Interest 100K -> Coverage = 0.5x (< 1.0)
        operating_cash_flow=200000.0,
    )

    result = analyze_financial_risk(insolvent_input)

    assert "CB001_INSOLVENCY_ALERT" in result.circuit_breakers_triggered
    assert result.risk_score <= 39
    assert result.risk_level == "Critical"
    assert any("CRITICAL INSOLVENCY ALERT" in flag for flag in result.risk_flags)


def test_circuit_breaker_liquidity_distress():
    """Verify liquidity distress trigger caps score at Critical."""
    distress_input = FinancialMetricsInput(
        company_name="Distressed Store",
        revenue=2000000.0,
        net_income=-300000.0,           # Negative Net Income
        operating_cash_flow=-200000.0,  # Negative OCF
        current_assets=600000.0,
        current_liabilities=1000000.0,  # Current Ratio = 0.6 (< 1.0)
        total_debt=1500000.0,
        total_equity=200000.0,
        total_assets=1700000.0,
        ebit=-250000.0,
        interest_expense=80000.0,
    )

    result = analyze_financial_risk(distress_input)

    assert "CB002_LIQUIDITY_DISTRESS" in result.circuit_breakers_triggered
    assert result.risk_score <= 39
    assert result.risk_level == "Critical"
