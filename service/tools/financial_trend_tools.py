import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from service.models import FinancialMetric, ToolExecution
from service.tools.financial_validation import validate_financial_metrics


def _period_sort_key(period: str) -> Tuple[int, str]:
    years = re.findall(r"\d{4}", period)
    year = int(years[-1]) if years else -1
    return year, period


def _percentage_change(
    previous: float,
    current: float,
) -> Optional[float]:
    if previous == 0:
        return None
    return (current - previous) / abs(previous)


def _ratio(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _collect_evidence(
    metric_maps: Dict[str, Dict[str, FinancialMetric]],
    periods: List[str],
    metric_names: List[str],
) -> List[str]:
    return sorted(
        {
            evidence_id
            for period in periods
            for name in metric_names
            if name in metric_maps.get(period, {})
            for evidence_id in metric_maps[period][name].evidence_ids
        }
    )


def analyze_financial_trends(
    metrics: List[FinancialMetric],
) -> ToolExecution:
    validation = validate_financial_metrics(metrics)

    if not validation.is_valid:
        return ToolExecution(
            tool_name="analyze_financial_trends",
            status="validation_failed",
            outputs={
                "validation_issues": [
                    issue.model_dump(mode="json")
                    for issue in validation.issues
                ]
            },
            error="Financial metrics failed validation.",
        )

    metric_maps: Dict[str, Dict[str, FinancialMetric]] = defaultdict(dict)

    for metric in validation.valid_metrics:
        if metric.reporting_period and metric.value is not None:
            metric_maps[metric.reporting_period][metric.name] = metric

    periods = sorted(metric_maps, key=_period_sort_key)

    if len(periods) < 2:
        return ToolExecution(
            tool_name="analyze_financial_trends",
            status="insufficient_data",
            missing_inputs=["at least two reporting periods"],
            error="Trend analysis requires at least two reporting periods.",
        )

    comparisons = []
    risk_signals = []
    protective_signals = []
    unavailable_changes = []

    for previous_period, current_period in zip(periods, periods[1:]):
        previous = {
            name: metric.value
            for name, metric in metric_maps[previous_period].items()
        }
        current = {
            name: metric.value
            for name, metric in metric_maps[current_period].items()
        }

        period_result = {
            "previous_period": previous_period,
            "current_period": current_period,
            "changes": {},
        }

        for metric_name in [
            "revenue",
            "net_income",
            "total_debt",
            "total_equity",
            "total_assets",
            "operating_cash_flow",
            "accounts_receivable",
            "current_assets",
            "current_liabilities",
        ]:
            if metric_name not in previous or metric_name not in current:
                unavailable_changes.append(
                    f"{previous_period}->{current_period}:{metric_name}"
                )
                continue

            change = _percentage_change(
                previous[metric_name],
                current[metric_name],
            )
            period_result["changes"][metric_name] = (
                round(change, 6) if change is not None else None
            )

        previous_margin = _ratio(
            previous.get("net_income"),
            previous.get("revenue"),
        )
        current_margin = _ratio(
            current.get("net_income"),
            current.get("revenue"),
        )

        if previous_margin is not None and current_margin is not None:
            margin_change = current_margin - previous_margin
            period_result["changes"]["net_margin_change"] = round(
                margin_change,
                6,
            )

            if margin_change <= -0.05:
                risk_signals.append(
                    {
                        "code": "NET_MARGIN_DETERIORATION",
                        "severity": "high",
                        "period": current_period,
                        "message": (
                            "Net margin decreased by at least "
                            "five percentage points."
                        ),
                        "value": round(margin_change, 6),
                    }
                )
            elif margin_change > 0:
                protective_signals.append(
                    {
                        "code": "NET_MARGIN_IMPROVEMENT",
                        "period": current_period,
                        "message": "Net margin improved.",
                        "value": round(margin_change, 6),
                    }
                )

        revenue_change = period_result["changes"].get("revenue")
        debt_change = period_result["changes"].get("total_debt")
        receivables_change = period_result["changes"].get(
            "accounts_receivable"
        )
        cash_flow_change = period_result["changes"].get(
            "operating_cash_flow"
        )
        income_change = period_result["changes"].get("net_income")

        if debt_change is not None and debt_change >= 0.25:
            risk_signals.append(
                {
                    "code": "RAPID_DEBT_GROWTH",
                    "severity": "high",
                    "period": current_period,
                    "message": "Total debt increased by at least 25%.",
                    "value": debt_change,
                }
            )

        if revenue_change is not None and revenue_change <= -0.10:
            risk_signals.append(
                {
                    "code": "REVENUE_DECLINE",
                    "severity": "high",
                    "period": current_period,
                    "message": "Revenue declined by at least 10%.",
                    "value": revenue_change,
                }
            )
        elif revenue_change is not None and revenue_change > 0:
            protective_signals.append(
                {
                    "code": "REVENUE_GROWTH",
                    "period": current_period,
                    "message": "Revenue increased.",
                    "value": revenue_change,
                }
            )

        if (
            receivables_change is not None
            and revenue_change is not None
            and receivables_change - revenue_change >= 0.15
        ):
            risk_signals.append(
                {
                    "code": "RECEIVABLES_OUTPACE_REVENUE",
                    "severity": "high",
                    "period": current_period,
                    "message": (
                        "Accounts receivable growth exceeded revenue "
                        "growth by at least 15 percentage points."
                    ),
                    "value": round(
                        receivables_change - revenue_change,
                        6,
                    ),
                }
            )

        if (
            income_change is not None
            and cash_flow_change is not None
            and income_change > 0
            and cash_flow_change < 0
        ):
            risk_signals.append(
                {
                    "code": "INCOME_CASH_FLOW_DIVERGENCE",
                    "severity": "high",
                    "period": current_period,
                    "message": (
                        "Net income increased while operating cash "
                        "flow declined."
                    ),
                    "value": round(
                        income_change - cash_flow_change,
                        6,
                    ),
                }
            )

        current_ratio_previous = _ratio(
            previous.get("current_assets"),
            previous.get("current_liabilities"),
        )
        current_ratio_current = _ratio(
            current.get("current_assets"),
            current.get("current_liabilities"),
        )

        if (
            current_ratio_previous is not None
            and current_ratio_current is not None
        ):
            ratio_change = (
                current_ratio_current - current_ratio_previous
            )
            period_result["changes"]["current_ratio_change"] = round(
                ratio_change,
                6,
            )

            if ratio_change <= -0.25:
                risk_signals.append(
                    {
                        "code": "LIQUIDITY_DETERIORATION",
                        "severity": "medium",
                        "period": current_period,
                        "message": (
                            "Current ratio deteriorated by at least 0.25."
                        ),
                        "value": round(ratio_change, 6),
                    }
                )

        comparisons.append(period_result)

    evidence_ids = _collect_evidence(
        metric_maps,
        periods,
        [
            "revenue",
            "net_income",
            "total_debt",
            "operating_cash_flow",
            "accounts_receivable",
            "current_assets",
            "current_liabilities",
        ],
    )

    status = "completed" if comparisons else "insufficient_data"

    return ToolExecution(
        tool_name="analyze_financial_trends",
        status=status,
        inputs={"reporting_periods": periods},
        outputs={
            "comparisons": comparisons,
            "risk_signals": risk_signals,
            "protective_signals": protective_signals,
            "unavailable_changes": sorted(set(unavailable_changes)),
        },
        evidence_ids=evidence_ids,
    )