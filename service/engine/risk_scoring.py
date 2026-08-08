from statistics import mean
from typing import Dict, List, Optional, Tuple

from service.models import (
    ConfidenceLevel,
    RiskLevel,
    ScoreComponent,
    ToolExecution,
)


CATEGORY_WEIGHTS = {
    "financial_distress": 0.30,
    "liquidity_and_leverage": 0.25,
    "profitability": 0.15,
    "cash_flow_quality": 0.15,
    "manipulation_indicators": 0.15,
}


def _execution(
    executions: List[ToolExecution],
    tool_name: str,
) -> Optional[ToolExecution]:
    return next(
        (
            execution
            for execution in executions
            if execution.tool_name == tool_name
            and execution.status in {"completed", "partial"}
        ),
        None,
    )


def _ratio_value(
    calculations: Dict,
    ratio_name: str,
) -> Optional[float]:
    result = calculations.get(ratio_name, {})
    if result.get("status") != "calculated":
        return None
    return result.get("value")


def _low_is_risky(
    value: Optional[float],
    bad: float,
    warning: float,
) -> Optional[float]:
    if value is None:
        return None
    if value <= bad:
        return 90
    if value <= warning:
        return 60
    return 20


def _high_is_risky(
    value: Optional[float],
    warning: float,
    bad: float,
) -> Optional[float]:
    if value is None:
        return None
    if value >= bad:
        return 90
    if value >= warning:
        return 60
    return 20


def _negative_is_risky(
    value: Optional[float],
    warning: float = 0.05,
) -> Optional[float]:
    if value is None:
        return None
    if value < 0:
        return 90
    if value < warning:
        return 60
    return 20


def _average_available(values: List[Optional[float]]) -> Optional[float]:
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def _ratio_category_scores(
    executions: List[ToolExecution],
) -> Dict[str, float]:
    execution = _execution(
        executions,
        "calculate_financial_ratios",
    )
    if execution is None:
        return {}

    calculations = execution.outputs.get("calculations", {})

    liquidity_score = _average_available(
        [
            _low_is_risky(
                _ratio_value(calculations, "current_ratio"),
                bad=1.0,
                warning=1.5,
            ),
            _low_is_risky(
                _ratio_value(calculations, "quick_ratio"),
                bad=0.5,
                warning=1.0,
            ),
            _high_is_risky(
                _ratio_value(calculations, "debt_to_equity"),
                warning=1.0,
                bad=2.0,
            ),
            _high_is_risky(
                _ratio_value(calculations, "debt_ratio"),
                warning=0.4,
                bad=0.6,
            ),
            _low_is_risky(
                _ratio_value(
                    calculations,
                    "interest_coverage_ratio",
                ),
                bad=1.0,
                warning=2.0,
            ),
        ]
    )

    profitability_score = _average_available(
        [
            _negative_is_risky(
                _ratio_value(calculations, "net_margin")
            ),
            _negative_is_risky(
                _ratio_value(calculations, "return_on_assets")
            ),
            _negative_is_risky(
                _ratio_value(calculations, "return_on_equity")
            ),
            _negative_is_risky(
                _ratio_value(calculations, "ebitda_margin")
            ),
        ]
    )

    cash_flow_ratio = _ratio_value(
        calculations,
        "operating_cash_flow_ratio",
    )
    cash_flow_score = _low_is_risky(
        cash_flow_ratio,
        bad=0.0,
        warning=1.0,
    )

    scores = {}
    if liquidity_score is not None:
        scores["liquidity_and_leverage"] = liquidity_score
    if profitability_score is not None:
        scores["profitability"] = profitability_score
    if cash_flow_score is not None:
        scores["cash_flow_quality"] = cash_flow_score

    return scores


def _model_category_scores(
    executions: List[ToolExecution],
) -> Dict[str, float]:
    scores = {}

    altman = _execution(
        executions,
        "calculate_altman_z_score",
    )
    if altman is not None:
        zone = altman.outputs.get("zone")
        scores["financial_distress"] = {
            "distress": 95,
            "grey": 60,
            "safe": 15,
        }.get(zone, 50)

    beneish = _execution(
        executions,
        "calculate_beneish_m_score",
    )
    if beneish is not None:
        scores["manipulation_indicators"] = (
            90
            if beneish.outputs.get("possible_manipulator")
            else 20
        )

    return scores


def _apply_trend_signals(
    scores: Dict[str, float],
    executions: List[ToolExecution],
) -> Dict[str, float]:
    trend = _execution(executions, "analyze_financial_trends")
    if trend is None:
        return scores

    signal_categories = {
        "RAPID_DEBT_GROWTH": (
            "liquidity_and_leverage",
            85,
        ),
        "LIQUIDITY_DETERIORATION": (
            "liquidity_and_leverage",
            70,
        ),
        "NET_MARGIN_DETERIORATION": (
            "profitability",
            80,
        ),
        "REVENUE_DECLINE": (
            "profitability",
            75,
        ),
        "INCOME_CASH_FLOW_DIVERGENCE": (
            "cash_flow_quality",
            90,
        ),
        "RECEIVABLES_OUTPACE_REVENUE": (
            "manipulation_indicators",
            80,
        ),
    }

    for signal in trend.outputs.get("risk_signals", []):
        mapping = signal_categories.get(signal.get("code"))
        if mapping is None:
            continue

        category, signal_score = mapping
        scores[category] = max(
            scores.get(category, 0),
            signal_score,
        )

    return scores


def _risk_level(score: float) -> RiskLevel:
    if score >= 75:
        return RiskLevel.CRITICAL
    if score >= 50:
        return RiskLevel.HIGH
    if score >= 25:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def _confidence(
    executions: List[ToolExecution],
    available_weight: float,
    missing_data_count: int,
    conflict_count: int,
) -> ConfidenceLevel:
    if not executions:
        return ConfidenceLevel.LOW

    completed = sum(
        execution.status in {"completed", "partial"}
        for execution in executions
    )
    completion_rate = completed / len(executions)

    quality_score = (
        (available_weight * 0.60)
        + (completion_rate * 0.40)
    )

    quality_score -= min(missing_data_count * 0.03, 0.30)
    quality_score -= min(conflict_count * 0.10, 0.30)
    quality_score = max(0.0, min(1.0, quality_score))

    if quality_score >= 0.80:
        return ConfidenceLevel.HIGH
    if quality_score >= 0.50:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def calculate_explainable_risk_score(
    executions: List[ToolExecution],
    missing_data_count: int = 0,
    conflict_count: int = 0,
) -> Tuple[
    Optional[float],
    RiskLevel,
    ConfidenceLevel,
    List[ScoreComponent],
]:
    category_scores = _ratio_category_scores(executions)
    category_scores.update(_model_category_scores(executions))
    category_scores = _apply_trend_signals(
        category_scores,
        executions,
    )

    if not category_scores:
        return (
            None,
            RiskLevel.UNKNOWN,
            ConfidenceLevel.LOW,
            [],
        )

    available_weight = sum(
        CATEGORY_WEIGHTS[category]
        for category in category_scores
    )

    breakdown = []
    final_score = 0.0

    for category, score in category_scores.items():
        normalized_weight = (
            CATEGORY_WEIGHTS[category] / available_weight
        )
        weighted_score = score * normalized_weight
        final_score += weighted_score

        breakdown.append(
            ScoreComponent(
                category=category,
                score=round(score, 2),
                weight=round(normalized_weight, 6),
                weighted_score=round(weighted_score, 2),
                explanation=(
                    f"{category.replace('_', ' ').title()} contributed "
                    f"{weighted_score:.2f} points using an effective "
                    f"weight of {normalized_weight:.1%}."
                ),
            )
        )

    final_score = round(max(0.0, min(100.0, final_score)), 2)

    confidence = _confidence(
        executions=executions,
        available_weight=available_weight,
        missing_data_count=missing_data_count,
        conflict_count=conflict_count,
    )

    return (
        final_score,
        _risk_level(final_score),
        confidence,
        breakdown,
    )