"""
Circuit Breakers & Red-Flag Override Engine
Detects critical solvency and liquidity failure triggers that override
standard category scores.
"""

from typing import List, Tuple, Optional
from service.engine.schemas import FinancialMetricsInput, ComputedRatios, AltmanZScoreResult


def evaluate_circuit_breakers(
    data: FinancialMetricsInput,
    ratios: ComputedRatios,
    z_score_res: AltmanZScoreResult
) -> Tuple[List[str], List[str], Optional[int]]:
    """
    Evaluates financial metrics against red-flag circuit breaker rules.
    
    Returns:
        (triggered_codes, warning_flags, max_score_cap)
        - triggered_codes: List of rule IDs (e.g. ['CB001_INSOLVENCY_ALERT'])
        - warning_flags: Detailed human-readable warning strings
        - max_score_cap: Upper bound cap for overall risk score (e.g. 39 or 49)
    """
    triggered_codes: List[str] = []
    warning_flags: List[str] = []
    caps: List[int] = []

    # CB001: Interest Coverage Insolvency (< 1.0)
    if ratios.interest_coverage_ratio is not None and ratios.interest_coverage_ratio < 1.0:
        triggered_codes.append("CB001_INSOLVENCY_ALERT")
        warning_flags.append(
            f"CRITICAL INSOLVENCY ALERT: Interest coverage ratio of {ratios.interest_coverage_ratio:.2f}x "
            "indicates earnings are insufficient to cover interest expense obligations."
        )
        caps.append(39)  # Cap overall score at Critical (<= 39)

    # CB002: Liquidity Distress (Negative Net Income & Cash Flow + Current Ratio < 1.0)
    if (
        data.net_income is not None and data.net_income < 0 and
        data.operating_cash_flow is not None and data.operating_cash_flow < 0 and
        ratios.current_ratio is not None and ratios.current_ratio < 1.0
    ):
        triggered_codes.append("CB002_LIQUIDITY_DISTRESS")
        warning_flags.append(
            "CRITICAL LIQUIDITY DISTRESS: Company is suffering net losses, negative cash flows, "
            f"and an inadequate current ratio of {ratios.current_ratio:.2f}."
        )
        caps.append(39)  # Cap overall score at Critical (<= 39)

    # CB003: Extreme Financial Leverage (D/E > 4.0 or Debt Ratio > 0.85)
    if (
        (ratios.debt_to_equity is not None and ratios.debt_to_equity > 4.0) or
        (ratios.debt_ratio is not None and ratios.debt_ratio > 0.85)
    ):
        triggered_codes.append("CB003_EXTREME_LEVERAGE")
        warning_flags.append(
            "EXTREME FINANCIAL LEVERAGE: Excessive debt relative to equity/assets places firm at high risk."
        )
        caps.append(49)  # Cap overall score at High (<= 49)

    # CB004: Altman Z-Score Statistical Bankruptcy Distress Zone (< 1.81)
    if z_score_res.zone == "Distress":
        triggered_codes.append("CB004_ALTMAN_DISTRESS")
        warning_flags.append(
            f"STATISTICAL BANKRUPT RISK: Altman Z-Score of {z_score_res.score} falls in the Distress Zone."
        )
        caps.append(49)  # Cap overall score at High (<= 49)

    max_score_cap = min(caps) if caps else None
    return triggered_codes, warning_flags, max_score_cap
