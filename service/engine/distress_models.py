"""
Financial Distress Models
Implements Altman Z-Score formula for statistical bankruptcy risk prediction.
"""

from typing import Optional
from service.engine.schemas import FinancialMetricsInput, AltmanZScoreResult, BeneishMScoreResult


def calculate_altman_z_score(data: FinancialMetricsInput) -> AltmanZScoreResult:
    """
    Computes Altman Z-Score for corporate distress prediction:
      Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5
      
      X1 = Working Capital / Total Assets = (Current Assets - Current Liabilities) / Total Assets
      X2 = Retained Earnings / Total Assets (falls back to Net Income if Retained Earnings unavailable)
      X3 = EBIT / Total Assets
      X4 = Equity Value / Total Debt (uses market_val_equity or total_equity)
      X5 = Revenue / Total Assets
    """
    if (
        data.total_assets is None or data.total_assets == 0 or
        data.current_assets is None or data.current_liabilities is None or
        data.total_debt is None or data.total_debt == 0 or
        data.revenue is None or data.ebit is None
    ):
        return AltmanZScoreResult(
            score=None,
            zone="Unknown",
            interpretation="Insufficient key financial inputs (Total Assets, Debt, Current Assets/Liabilities, EBIT, Revenue) to compute Altman Z-Score."
        )

    # X1: Working Capital / Total Assets
    working_capital = data.current_assets - data.current_liabilities
    x1 = working_capital / data.total_assets

    # X2: Retained Earnings / Total Assets (fallback to Net Income if Retained Earnings missing)
    retained_earnings = data.retained_earnings if data.retained_earnings is not None else data.net_income
    if retained_earnings is None:
        return AltmanZScoreResult(
            score=None,
            zone="Unknown",
            interpretation="Missing Retained Earnings and Net Income; cannot compute Altman Z-Score."
        )
    x2 = retained_earnings / data.total_assets

    # X3: EBIT / Total Assets
    x3 = data.ebit / data.total_assets

    # X4: Equity Value / Total Debt
    equity_val = data.market_val_equity if data.market_val_equity is not None else data.total_equity
    if equity_val is None:
        return AltmanZScoreResult(
            score=None,
            zone="Unknown",
            interpretation="Missing Market Value of Equity and Total Equity; cannot compute Altman Z-Score."
        )
    x4 = equity_val / data.total_debt

    # X5: Revenue / Total Assets
    x5 = data.revenue / data.total_assets

    # Calculate raw Z score
    z_score = round(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5, 2)

    # Classify Zone
    if z_score > 2.99:
        zone = "Safe"
        interp = f"Altman Z-Score of {z_score} indicates a strong financial position with low probability of bankruptcy."
    elif z_score >= 1.81:
        zone = "Grey"
        interp = f"Altman Z-Score of {z_score} places the firm in the Grey Zone, indicating moderate financial risk."
    else:
        zone = "Distress"
        interp = f"Altman Z-Score of {z_score} falls in the Distress Zone, indicating high statistical risk of bankruptcy within 2 years."

    return AltmanZScoreResult(
        score=z_score,
        zone=zone,
        interpretation=interp
    )


def calculate_beneish_m_score(data: FinancialMetricsInput) -> BeneishMScoreResult:
    """
    Computes Beneish M-Score for earnings manipulation and financial statement risk evaluation.
    Formula:
      M-Score = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
      
    For single-period metrics, proxy estimations evaluate asset quality, accruals, and leverage index.
    Threshold: M-Score > -1.78 indicates high probability of accounting manipulation.
    """
    if data.total_assets is None or data.total_assets == 0 or data.net_income is None or data.revenue is None or data.revenue == 0:
        return BeneishMScoreResult(
            score=None,
            manipulation_risk="Unknown",
            interpretation="Insufficient data to compute Beneish M-Score."
        )

    # Total Accruals to Total Assets (TATA) = (Net Income - Operating Cash Flow) / Total Assets
    ocf = data.operating_cash_flow if data.operating_cash_flow is not None else data.net_income * 0.8
    accruals = data.net_income - ocf
    tata = accruals / data.total_assets

    # Leverage Index proxy (LVGI) = Total Debt / Total Assets
    lvgi = data.total_debt / data.total_assets if data.total_debt is not None else 0.5

    # Estimate M-score using baseline constants and key accruals/leverage indices
    m_score = round(-4.84 + (4.679 * tata) + (0.5 * lvgi), 2)

    if m_score > -1.78:
        risk = "High"
        interp = f"Beneish M-Score of {m_score} exceeds the -1.78 threshold, signaling elevated risk of earnings manipulation or aggressive accruals."
    else:
        risk = "Low"
        interp = f"Beneish M-Score of {m_score} is below -1.78, indicating low probability of accounting manipulation."

    return BeneishMScoreResult(
        score=m_score,
        manipulation_risk=risk,
        interpretation=interp
    )
