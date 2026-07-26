"""
Unit tests for risk_agent primary orchestrator entry point.
"""

from service.engine.risk_agent import analyze_risk_agent, analyze_financial_risk
from service.engine.schemas import FinancialMetricsInput, RiskAnalysisResult


def test_analyze_risk_agent_dict_input():
    """Verify analyze_risk_agent accepts dict payload."""
    raw_payload = {
        "company_name": "Acme Widgets",
        "industry_sector": "Manufacturing",
        "metrics": {
            "revenue": 5000000.0,
            "net_income": 400000.0,
            "total_debt": 1000000.0,
            "total_equity": 2000000.0,
            "total_assets": 3000000.0,
            "current_assets": 1200000.0,
            "current_liabilities": 600000.0,
            "ebit": 500000.0,
            "interest_expense": 50000.0,
            "operating_cash_flow": 450000.0
        }
    }

    result = analyze_risk_agent(raw_payload)
    assert isinstance(result, RiskAnalysisResult)
    assert result.company_name == "Acme Widgets"
    assert result.risk_score is not None
    assert result.risk_score >= 70


def test_legacy_analyze_financial_risk():
    """Verify legacy analyze_financial_risk compatibility wrapper."""
    raw_data = {
        "company_name": "Legacy Corp",
        "revenue": 1000000.0,
        "net_income": 100000.0
    }
    dict_res = analyze_financial_risk(raw_data)
    assert isinstance(dict_res, dict)
    assert dict_res["company_name"] == "Legacy Corp"
    assert "risk_score" in dict_res
    assert "computed_ratios" in dict_res
