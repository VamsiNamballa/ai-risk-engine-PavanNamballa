import re
from typing import Dict, List, Set

from service.models import (
    AnalysisSelection,
    RiskAnalysisInput,
    RiskAnalysisPlan,
)


RATIO_METRICS = {
    "current_assets",
    "current_liabilities",
    "inventory",
    "total_debt",
    "total_equity",
    "total_assets",
    "ebit",
    "interest_expense",
    "net_income",
    "revenue",
    "ebitda",
    "operating_cash_flow",
}

BENEISH_METRICS = {
    "revenue",
    "accounts_receivable",
    "cost_of_goods_sold",
    "current_assets",
    "property_plant_equipment",
    "total_assets",
    "depreciation",
    "selling_general_admin_expense",
    "total_debt",
    "net_income",
    "operating_cash_flow",
}

ALTMAN_METRICS = {
    "current_assets",
    "current_liabilities",
    "total_assets",
    "retained_earnings",
    "ebit",
    "market_value_equity",
    "total_liabilities",
    "revenue",
}

TREND_METRICS = {
    "revenue",
    "net_income",
    "total_debt",
    "total_equity",
    "total_assets",
    "operating_cash_flow",
    "accounts_receivable",
    "current_assets",
    "current_liabilities",
}

INTENT_KEYWORDS: Dict[str, Set[str]] = {
    "liquidity": {
        "liquidity",
        "current ratio",
        "quick ratio",
        "short-term debt",
        "working capital",
    },
    "leverage": {
        "leverage",
        "debt",
        "solvency",
        "interest coverage",
        "debt-to-equity",
    },
    "profitability": {
        "profit",
        "profitability",
        "margin",
        "return on assets",
        "return on equity",
        "earnings",
    },
    "distress": {
        "bankruptcy",
        "distress",
        "insolvency",
        "failure risk",
        "altman",
        "going concern",
    },
    "manipulation": {
        "manipulation",
        "fraud",
        "beneish",
        "earnings quality",
        "accounting irregularity",
        "misstatement",
    },
    "trend": {
        "trend",
        "growth",
        "decline",
        "year over year",
        "year-over-year",
        "deterioration",
        "improvement",
        "compare periods",
    },
    "general_risk": {
        "risk",
        "financial health",
        "financial condition",
        "overall assessment",
        "analyze company",
    },
}


def _period_sort_key(period: str):
    years = re.findall(r"\d{4}", period)
    year = int(years[-1]) if years else -1
    return year, period


def _detect_intents(query: str) -> List[str]:
    normalized_query = " ".join(query.lower().split())

    intents = [
        intent
        for intent, keywords in INTENT_KEYWORDS.items()
        if any(keyword in normalized_query for keyword in keywords)
    ]

    return sorted(set(intents or ["general_risk"]))


def _metrics_by_period(
    payload: RiskAnalysisInput,
) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}

    for metric in payload.metrics:
        if metric.value is None or not metric.reporting_period:
            continue

        result.setdefault(metric.reporting_period, set()).add(metric.name)

    return result


def _missing_for_latest_period(
    required: Set[str],
    metrics_by_period: Dict[str, Set[str]],
    periods: List[str],
) -> List[str]:
    if not periods:
        return sorted(required)

    return sorted(required - metrics_by_period[periods[-1]])


def _missing_for_two_periods(
    required: Set[str],
    metrics_by_period: Dict[str, Set[str]],
    periods: List[str],
) -> List[str]:
    if len(periods) < 2:
        return ["at least two reporting periods"]

    selected_periods = periods[-2:]
    missing = []

    for period in selected_periods:
        for metric_name in sorted(required - metrics_by_period[period]):
            missing.append(f"{period}:{metric_name}")

    return missing


def create_risk_analysis_plan(
    payload: RiskAnalysisInput,
) -> RiskAnalysisPlan:
    intents = _detect_intents(payload.query)
    metrics_by_period = _metrics_by_period(payload)
    periods = sorted(metrics_by_period, key=_period_sort_key)

    selected: List[AnalysisSelection] = []
    skipped: List[AnalysisSelection] = []

    ratio_relevant = bool(
        set(intents)
        & {
            "liquidity",
            "leverage",
            "profitability",
            "general_risk",
        }
    )

    if ratio_relevant:
        missing = _missing_for_latest_period(
            RATIO_METRICS,
            metrics_by_period,
            periods,
        )
        selected.append(
            AnalysisSelection(
                analysis_name="Financial ratio analysis",
                tool_name="calculate_financial_ratios",
                reason=(
                    "The question requires liquidity, leverage, "
                    "profitability or general financial-risk analysis."
                ),
                required_metrics=sorted(RATIO_METRICS),
                missing_metrics=missing,
                reporting_periods=periods[-1:] if periods else [],
            )
        )
    else:
        skipped.append(
            AnalysisSelection(
                analysis_name="Financial ratio analysis",
                tool_name="calculate_financial_ratios",
                reason="The detected intent does not require ratio analysis.",
                required_metrics=sorted(RATIO_METRICS),
            )
        )

    if "manipulation" in intents or "general_risk" in intents:
        missing = _missing_for_two_periods(
            BENEISH_METRICS,
            metrics_by_period,
            periods,
        )
        selected.append(
            AnalysisSelection(
                analysis_name="Beneish manipulation analysis",
                tool_name="calculate_beneish_m_score",
                reason=(
                    "The question concerns manipulation, fraud, "
                    "earnings quality or overall risk."
                ),
                required_metrics=sorted(BENEISH_METRICS),
                missing_metrics=missing,
                reporting_periods=periods[-2:],
            )
        )
    else:
        skipped.append(
            AnalysisSelection(
                analysis_name="Beneish manipulation analysis",
                tool_name="calculate_beneish_m_score",
                reason="No manipulation or general-risk intent was detected.",
                required_metrics=sorted(BENEISH_METRICS),
            )
        )

    if "distress" in intents or "general_risk" in intents:
        missing = _missing_for_latest_period(
            ALTMAN_METRICS,
            metrics_by_period,
            periods,
        )
        selected.append(
            AnalysisSelection(
                analysis_name="Altman financial-distress analysis",
                tool_name="calculate_altman_z_score",
                reason=(
                    "The question concerns bankruptcy, distress, "
                    "solvency or overall financial risk."
                ),
                required_metrics=sorted(ALTMAN_METRICS),
                missing_metrics=missing,
                reporting_periods=periods[-1:] if periods else [],
            )
        )
    else:
        skipped.append(
            AnalysisSelection(
                analysis_name="Altman financial-distress analysis",
                tool_name="calculate_altman_z_score",
                reason="No financial-distress intent was detected.",
                required_metrics=sorted(ALTMAN_METRICS),
            )
        )

    if "trend" in intents or "general_risk" in intents:
        missing = _missing_for_two_periods(
            TREND_METRICS,
            metrics_by_period,
            periods,
        )
        selected.append(
            AnalysisSelection(
                analysis_name="Multi-period trend analysis",
                tool_name="analyze_financial_trends",
                reason=(
                    "The question requires historical comparison "
                    "or an overall risk assessment."
                ),
                required_metrics=sorted(TREND_METRICS),
                missing_metrics=missing,
                reporting_periods=periods,
            )
        )
    else:
        skipped.append(
            AnalysisSelection(
                analysis_name="Multi-period trend analysis",
                tool_name="analyze_financial_trends",
                reason="No trend or general-risk intent was detected.",
                required_metrics=sorted(TREND_METRICS),
            )
        )

    return RiskAnalysisPlan(
        detected_intents=intents,
        selected_analyses=selected,
        skipped_analyses=skipped,
        available_reporting_periods=periods,
    )