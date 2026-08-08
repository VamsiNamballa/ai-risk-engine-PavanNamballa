import pytest

from service.agents import (
    create_risk_analysis_plan,
    run_risk_analysis_agent,
)
from service.main import app
from service.models import (
    FinancialMetric,
    RiskAnalysisInput,
    RiskLevel,
    ValidationSeverity,
)
from service.tools.financial_ratio_tools import (
    calculate_financial_ratios,
)
from service.tools.financial_risk_models import (
    calculate_altman_z_score,
)
from service.tools.financial_trend_tools import (
    analyze_financial_trends,
)
from service.tools.financial_validation import (
    validate_financial_metrics,
)


def metric(name, value, period="2025"):
    return FinancialMetric(
        name=name,
        value=value,
        reporting_period=period,
    )


def complete_ratio_metrics():
    values = {
        "current_assets": 600,
        "current_liabilities": 300,
        "inventory": 100,
        "total_debt": 800,
        "total_equity": 200,
        "total_assets": 1200,
        "ebit": 100,
        "interest_expense": 50,
        "net_income": 60,
        "revenue": 1000,
        "ebitda": 150,
        "operating_cash_flow": 80,
    }
    return [metric(name, value) for name, value in values.items()]


def test_validation_rejects_impossible_asset_values():
    result = validate_financial_metrics(
        [
            metric("current_assets", 200),
            metric("total_assets", 100),
        ]
    )

    assert result.is_valid is False
    assert any(
        issue.code == "CURRENT_ASSETS_EXCEED_TOTAL_ASSETS"
        for issue in result.issues
    )


def test_validation_allows_negative_equity_with_warning():
    result = validate_financial_metrics(
        [
            metric("total_equity", -100),
            metric("total_assets", 1000),
        ]
    )

    assert result.is_valid is True
    assert any(
        issue.code == "NEGATIVE_EQUITY"
        and issue.severity == ValidationSeverity.WARNING
        for issue in result.issues
    )


def test_validation_rejects_conflicting_duplicate_values():
    result = validate_financial_metrics(
        [
            metric("revenue", 1000),
            metric("revenue", 1200),
        ]
    )

    assert result.is_valid is False
    assert any(
        issue.code == "CONFLICTING_DUPLICATE_VALUES"
        for issue in result.issues
    )


def test_ratio_tool_calculates_expected_values():
    result = calculate_financial_ratios(
        complete_ratio_metrics(),
        reporting_period="2025",
        selected_ratios=[
            "current_ratio",
            "quick_ratio",
            "debt_to_equity",
        ],
    )

    calculations = result.outputs["calculations"]

    assert result.status == "completed"
    assert calculations["current_ratio"]["value"] == pytest.approx(2.0)
    assert calculations["quick_ratio"]["value"] == pytest.approx(
        1.666667
    )
    assert calculations["debt_to_equity"]["value"] == pytest.approx(4.0)


def test_ratio_tool_handles_zero_denominator():
    result = calculate_financial_ratios(
        [
            metric("current_assets", 100),
            metric("current_liabilities", 0),
        ],
        reporting_period="2025",
        selected_ratios=["current_ratio"],
    )

    calculation = result.outputs["calculations"]["current_ratio"]

    assert calculation["status"] == "invalid_denominator"
    assert calculation["value"] is None


def test_altman_safe_zone_calculation():
    values = {
        "current_assets": 600,
        "current_liabilities": 300,
        "total_assets": 1000,
        "retained_earnings": 200,
        "ebit": 150,
        "market_value_equity": 800,
        "total_liabilities": 400,
        "revenue": 1200,
    }

    result = calculate_altman_z_score(
        [metric(name, value) for name, value in values.items()],
        reporting_period="2025",
    )

    assert result.status == "completed"
    assert result.outputs["score"] == pytest.approx(3.535)
    assert result.outputs["zone"] == "safe"


def test_trend_tool_detects_cash_flow_divergence():
    values = {
        "2024": {
            "revenue": 1000,
            "net_income": 100,
            "operating_cash_flow": 120,
        },
        "2025": {
            "revenue": 1100,
            "net_income": 140,
            "operating_cash_flow": 90,
        },
    }

    metrics = [
        metric(name, value, period)
        for period, period_values in values.items()
        for name, value in period_values.items()
    ]

    result = analyze_financial_trends(metrics)
    signal_codes = {
        signal["code"]
        for signal in result.outputs["risk_signals"]
    }

    assert result.status == "completed"
    assert "INCOME_CASH_FLOW_DIVERGENCE" in signal_codes


def test_planner_selects_only_relevant_analysis():
    payload = RiskAnalysisInput(
        query="Assess liquidity and leverage risk",
        metrics=complete_ratio_metrics(),
    )

    plan = create_risk_analysis_plan(payload)
    selected_tools = {
        selection.tool_name
        for selection in plan.selected_analyses
    }

    assert selected_tools == {"calculate_financial_ratios"}


def test_agent_returns_explainable_score():
    result = run_risk_analysis_agent(
        RiskAnalysisInput(
            query="Assess liquidity and leverage risk",
            company_name="Example Corp",
            metrics=complete_ratio_metrics(),
        )
    )

    assert result.overall_risk_score is not None
    assert 0 <= result.overall_risk_score <= 100
    assert result.risk_level != RiskLevel.UNKNOWN
    assert result.score_breakdown
    assert sum(
        component.weight
        for component in result.score_breakdown
    ) == pytest.approx(1.0, abs=0.00001)


def test_agent_reports_missing_data_without_hallucinating():
    result = run_risk_analysis_agent(
        RiskAnalysisInput(
            query="Check accounting manipulation risk",
            company_name="Incomplete Corp",
            metrics=[metric("revenue", 1000)],
        )
    )

    assert result.overall_risk_score is None
    assert result.risk_level == RiskLevel.UNKNOWN
    assert result.missing_data
    assert result.tool_executions[0].status == "insufficient_data"


def test_agent_routes_are_registered():
    routes = {
        (
            method,
            route.path,
        )
        for route in app.routes
        for method in route.methods or []
    }

    assert (
        "POST",
        "/agents/risk-analysis",
    ) in routes
    assert (
        "GET",
        "/agents/risk-analysis/capabilities",
    ) in routes