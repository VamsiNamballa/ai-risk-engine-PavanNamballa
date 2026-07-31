"""
Ratio Computation Engine
Calculates 10 core financial ratios from FinancialMetricsInput
and computes data completeness confidence.
"""

from typing import Optional, Tuple
from service.engine.schemas import FinancialMetricsInput, ComputedRatios


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Perform division safely. Returns None if either input is None or denominator is zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def calculate_ratios(data: FinancialMetricsInput) -> Tuple[ComputedRatios, float]:
    """
    Computes 10 core financial ratios and calculates data confidence percentage.
    
    Returns:
        (ComputedRatios, data_confidence_percentage)
    """
    # 1. Liquidity Ratios
    current_ratio = _safe_div(data.current_assets, data.current_liabilities)
    
    quick_assets = None
    if data.current_assets is not None and data.inventory is not None:
        quick_assets = data.current_assets - data.inventory
    elif data.current_assets is not None:
        quick_assets = data.current_assets  # fallback if inventory not specified
        
    quick_ratio = _safe_div(quick_assets, data.current_liabilities)
    cash_flow_ratio = _safe_div(data.operating_cash_flow, data.current_liabilities)

    # 2. Solvency & Leverage Ratios
    debt_to_equity = _safe_div(data.total_debt, data.total_equity)
    debt_ratio = _safe_div(data.total_debt, data.total_assets)
    interest_coverage = _safe_div(data.ebit, data.interest_expense)

    # 3. Profitability & Margins
    net_margin = _safe_div(data.net_income, data.revenue)
    ebitda_margin = _safe_div(data.ebitda, data.revenue)

    # 4. Asset & Equity Efficiency
    roa = _safe_div(data.net_income, data.total_assets)
    roe = _safe_div(data.net_income, data.total_equity)

    # 5. YoY Growth Trajectory
    revenue_growth_yoy = None
    if data.revenue is not None and data.prior_revenue is not None and data.prior_revenue > 0:
        revenue_growth_yoy = (data.revenue - data.prior_revenue) / data.prior_revenue

    net_income_growth_yoy = None
    if data.net_income is not None and data.prior_net_income is not None and data.prior_net_income != 0:
        net_income_growth_yoy = (data.net_income - data.prior_net_income) / abs(data.prior_net_income)

    ratios = ComputedRatios(
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        debt_to_equity=debt_to_equity,
        debt_ratio=debt_ratio,
        interest_coverage_ratio=interest_coverage,
        net_margin=net_margin,
        roa=roa,
        roe=roe,
        ebitda_margin=ebitda_margin,
        cash_flow_ratio=cash_flow_ratio,
        revenue_growth_yoy=revenue_growth_yoy,
        net_income_growth_yoy=net_income_growth_yoy,
    )

    # Compute Data Completeness Confidence %
    required_fields = [
        data.revenue, data.net_income, data.total_debt, data.total_equity,
        data.total_assets, data.current_assets, data.current_liabilities,
        data.ebit, data.interest_expense, data.operating_cash_flow
    ]
    present_count = sum(1 for field in required_fields if field is not None)
    confidence = round((present_count / len(required_fields)) * 100, 1)

    return ratios, confidence
