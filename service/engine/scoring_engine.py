"""
Category-Weighted Scoring Engine
Applies continuous linear ratio interpolation, computes category scores
(Solvency 35%, Liquidity 30%, Profitability 20%, Efficiency 15%), applies
circuit breaker overrides, and formats recommendations.
"""

from typing import Optional, List, Dict
from service.engine.schemas import (
    FinancialMetricsInput,
    ComputedRatios,
    CategoryScores,
    AltmanZScoreResult,
    RiskAnalysisResult,
)
from service.engine.ratio_engine import calculate_ratios
from service.engine.distress_models import calculate_altman_z_score
from service.engine.circuit_breakers import evaluate_circuit_breakers


# Ratio Interpolation Target Bounds: (min_bound, max_bound, lower_is_better)
RATIO_BOUNDS: Dict[str, tuple[float, float, bool]] = {
    "interest_coverage_ratio": (1.0, 5.0, False),
    "debt_to_equity":           (0.5, 3.0, True),
    "debt_ratio":               (0.2, 0.8, True),
    "current_ratio":            (0.8, 2.0, False),
    "quick_ratio":              (0.5, 1.5, False),
    "cash_flow_ratio":          (0.2, 1.0, False),
    "net_margin":               (0.0, 0.15, False),
    "ebitda_margin":            (0.0, 0.25, False),
    "roa":                      (0.0, 0.10, False),
    "roe":                      (0.0, 0.18, False),
}

# Category Weights
CATEGORY_WEIGHTS = {
    "solvency": 0.35,
    "liquidity": 0.30,
    "profitability": 0.20,
    "efficiency": 0.15,
}

CATEGORY_RATIOS = {
    "solvency": ["interest_coverage_ratio", "debt_to_equity", "debt_ratio"],
    "liquidity": ["current_ratio", "quick_ratio", "cash_flow_ratio"],
    "profitability": ["net_margin", "ebitda_margin"],
    "efficiency": ["roa", "roe"],
}


def _score_ratio_continuous(name: str, value: Optional[float]) -> Optional[float]:
    """Linearly interpolates ratio value to a continuous 0-100 score."""
    if value is None or name not in RATIO_BOUNDS:
        return None

    min_b, max_b, lower_is_better = RATIO_BOUNDS[name]

    if lower_is_better:
        if value <= min_b:
            return 100.0
        if value >= max_b:
            return 0.0
        return round(100.0 * (1.0 - (value - min_b) / (max_b - min_b)), 1)
    else:
        if value <= min_b:
            return 0.0
        if value >= max_b:
            return 100.0
        return round(100.0 * ((value - min_b) / (max_b - min_b)), 1)


def compute_category_scores(ratios: ComputedRatios) -> CategoryScores:
    """Computes averaged scores (0-100) for each of the 4 financial categories."""
    ratio_dict = ratios.model_dump()
    cat_scores = {}

    for cat_name, ratio_keys in CATEGORY_RATIOS.items():
        scores = []
        for key in ratio_keys:
            val = ratio_dict.get(key)
            s = _score_ratio_continuous(key, val)
            if s is not None:
                scores.append(s)
        
        if scores:
            cat_scores[cat_name] = round(sum(scores) / len(scores), 1)
        else:
            cat_scores[cat_name] = None

    return CategoryScores(**cat_scores)


def generate_recommendations(
    ratios: ComputedRatios,
    cat_scores: CategoryScores,
    circuit_codes: List[str]
) -> List[str]:
    """Generates actionable recommendations based on weak areas."""
    recs = []

    if "CB001_INSOLVENCY_ALERT" in circuit_codes:
        recs.append("PRIORITY ACTION: Restructure or pay down high-cost debt immediately to improve interest coverage above 1.5x.")
    if "CB002_LIQUIDITY_DISTRESS" in circuit_codes:
        recs.append("CRITICAL: Secure emergency credit facilities or equity funding to cover short-term working capital deficits.")

    if cat_scores.liquidity is not None and cat_scores.liquidity < 60:
        recs.append("Liquidity position is constrained — reduce inventory holding times and negotiate extended supplier payment terms.")
    if cat_scores.solvency is not None and cat_scores.solvency < 60:
        recs.append("Leverage is elevated — avoid taking on additional long-term liabilities until equity base strengthens.")
    if cat_scores.profitability is not None and cat_scores.profitability < 60:
        recs.append("Profitability is weak — conduct an operational cost audit to improve EBITDA and net margins.")
    if cat_scores.efficiency is not None and cat_scores.efficiency < 60:
        recs.append("Asset utilization is low — review underperforming assets and optimize capital allocation.")

    if not recs:
        recs.append("Financial profile is healthy across all categories — maintain current capital management practices.")

    return recs


def analyze_financial_risk(data: FinancialMetricsInput) -> RiskAnalysisResult:
    """
    Main Orchestrator Entry Point for the Risk Analysis Agent.
    Input: FinancialMetricsInput
    Output: RiskAnalysisResult
    """
    # 1. Compute Ratios & Data Confidence
    ratios, confidence = calculate_ratios(data)

    # 2. Compute Altman Z-Score
    z_res = calculate_altman_z_score(data)

    # 3. Evaluate Circuit Breakers
    cb_codes, cb_flags, max_score_cap = evaluate_circuit_breakers(data, ratios, z_res)

    # 4. Compute Category Sub-Scores
    cat_scores = compute_category_scores(ratios)

    # 5. Compute Weighted Overall Risk Score
    weighted_sum = 0.0
    weight_used = 0.0

    cat_dict = cat_scores.model_dump()
    for cat_name, weight in CATEGORY_WEIGHTS.items():
        score = cat_dict.get(cat_name)
        if score is not None:
            weighted_sum += score * weight
            weight_used += weight

    if weight_used > 0:
        raw_score = round(weighted_sum / weight_used)
    else:
        raw_score = None

    # Apply Circuit Breaker Score Cap
    if raw_score is not None and max_score_cap is not None:
        final_score = min(raw_score, max_score_cap)
    else:
        final_score = raw_score

    # Map Risk Level
    if final_score is None:
        risk_level = "Unknown"
    elif final_score >= 80:
        risk_level = "Low"
    elif final_score >= 60:
        risk_level = "Moderate"
    elif final_score >= 40:
        risk_level = "High"
    else:
        risk_level = "Critical"

    # Assemble Flags & Recommendations
    all_flags = list(cb_flags)
    if not all_flags and final_score is not None:
        if final_score >= 80:
            all_flags.append("Strong financial solvency and liquidity profile.")
        elif final_score >= 60:
            all_flags.append("Moderate financial profile with minor ratio vulnerabilities.")

    recommendations = generate_recommendations(ratios, cat_scores, cb_codes)

    return RiskAnalysisResult(
        company_name=data.company_name,
        industry_sector=data.industry_sector,
        reporting_year=data.reporting_year,
        risk_score=final_score,
        risk_level=risk_level,
        data_confidence=confidence,
        circuit_breakers_triggered=cb_codes,
        category_scores=cat_scores,
        altman_z_score=z_res,
        computed_ratios=ratios,
        risk_flags=all_flags,
        actionable_recommendations=recommendations,
    )
