"""
Unit tests for ratio_engine and distress_models.
"""

import pytest
from service.engine.schemas import FinancialMetricsInput
from service.engine.ratio_engine import calculate_ratios, _safe_div
from service.engine.distress_models import calculate_altman_z_score, calculate_beneish_m_score


def test_beneish_m_score():
    """Verify Beneish M-Score calculation."""
    data = FinancialMetricsInput(
        company_name="Sample Corp",
        revenue=1000000.0,
        net_income=100000.0,
        total_assets=2000000.0,
        total_debt=500000.0,
        operating_cash_flow=90000.0
    )

    m_res = calculate_beneish_m_score(data)
    assert m_res.score is not None
    assert m_res.manipulation_risk in ["Low", "High"]
    assert "Beneish M-Score" in m_res.interpretation


def test_safe_div():
    """Verify safe division function."""
    assert _safe_div(10, 2) == 5.0
    assert _safe_div(10, 0) is None
    assert _safe_div(None, 2) is None
    assert _safe_div(10, None) is None


def test_healthy_company_ratios_and_distress():
    """Verify ratio calculation & Altman Z-Score for a healthy firm."""
    healthy_data = FinancialMetricsInput(
        company_name="Healthy Corp",
        revenue=5000000.0,
        net_income=500000.0,
        total_debt=1000000.0,
        total_equity=3000000.0,
        total_assets=4000000.0,
        current_assets=2000000.0,
        current_liabilities=800000.0,
        inventory=400000.0,
        ebit=700000.0,
        interest_expense=70000.0,
        ebitda=850000.0,
        operating_cash_flow=650000.0,
        retained_earnings=1500000.0,
        market_val_equity=4000000.0,
    )

    ratios, confidence = calculate_ratios(healthy_data)

    # Ratio checks
    assert ratios.current_ratio == 2.5           # 2M / 0.8M
    assert ratios.quick_ratio == 2.0             # (2M - 0.4M) / 0.8M
    assert ratios.debt_to_equity == pytest.approx(0.333, rel=1e-2)  # 1M / 3M
    assert ratios.interest_coverage_ratio == 10.0 # 700K / 70K
    assert ratios.net_margin == 0.10             # 500K / 5M
    assert ratios.roa == 0.125                   # 500K / 4M
    assert ratios.roe == pytest.approx(0.166, rel=1e-2) # 500K / 3M

    assert confidence == 100.0

    # Altman Z-Score check
    z_res = calculate_altman_z_score(healthy_data)
    assert z_res.score is not None
    assert z_res.score > 2.99
    assert z_res.zone == "Safe"


def test_distressed_company_altman_z():
    """Verify Altman Z-Score for a distressed firm."""
    distressed_data = FinancialMetricsInput(
        company_name="Distressed LLC",
        revenue=2000000.0,
        net_income=-400000.0,
        total_debt=4000000.0,
        total_equity=500000.0,
        total_assets=4500000.0,
        current_assets=800000.0,
        current_liabilities=1800000.0,
        inventory=400000.0,
        ebit=-200000.0,
        interest_expense=300000.0,
        ebitda=-100000.0,
        operating_cash_flow=-300000.0,
        retained_earnings=-800000.0,
    )

    ratios, confidence = calculate_ratios(distressed_data)

    assert ratios.current_ratio == pytest.approx(0.444, rel=1e-2)
    assert ratios.interest_coverage_ratio == pytest.approx(-0.666, rel=1e-2)

    z_res = calculate_altman_z_score(distressed_data)
    assert z_res.score is not None
    assert z_res.score < 1.81
    assert z_res.zone == "Distress"


def test_missing_data_graceful_handling():
    """Verify behavior when critical inputs are missing."""
    sparse_data = FinancialMetricsInput(
        company_name="Sparse Inc",
        revenue=1000000.0,
        net_income=100000.0,
        # missing debt, assets, EBIT, etc.
    )

    ratios, confidence = calculate_ratios(sparse_data)
    assert ratios.current_ratio is None
    assert ratios.debt_to_equity is None
    assert ratios.net_margin == 0.10
    assert confidence < 50.0

    z_res = calculate_altman_z_score(sparse_data)
    assert z_res.score is None
    assert z_res.zone == "Unknown"
