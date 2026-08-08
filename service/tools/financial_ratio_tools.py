from typing import Callable, Dict, List, Optional, Tuple

from service.models import FinancialMetric, ToolExecution
from service.tools.financial_validation import validate_financial_metrics


RatioDefinition = Tuple[
    Tuple[str, ...],
    Callable[[Dict[str, float]], float],
    str,
]


RATIO_DEFINITIONS: Dict[str, RatioDefinition] = {
    "current_ratio": (
        ("current_assets", "current_liabilities"),
        lambda values: values["current_assets"] / values["current_liabilities"],
        "current_assets / current_liabilities",
    ),
    "quick_ratio": (
        ("current_assets", "inventory", "current_liabilities"),
        lambda values: (
            values["current_assets"] - values["inventory"]
        ) / values["current_liabilities"],
        "(current_assets - inventory) / current_liabilities",
    ),
    "debt_to_equity": (
        ("total_debt", "total_equity"),
        lambda values: values["total_debt"] / values["total_equity"],
        "total_debt / total_equity",
    ),
    "debt_ratio": (
        ("total_debt", "total_assets"),
        lambda values: values["total_debt"] / values["total_assets"],
        "total_debt / total_assets",
    ),
    "interest_coverage_ratio": (
        ("ebit", "interest_expense"),
        lambda values: values["ebit"] / values["interest_expense"],
        "ebit / interest_expense",
    ),
    "net_margin": (
        ("net_income", "revenue"),
        lambda values: values["net_income"] / values["revenue"],
        "net_income / revenue",
    ),
    "return_on_assets": (
        ("net_income", "total_assets"),
        lambda values: values["net_income"] / values["total_assets"],
        "net_income / total_assets",
    ),
    "return_on_equity": (
        ("net_income", "total_equity"),
        lambda values: values["net_income"] / values["total_equity"],
        "net_income / total_equity",
    ),
    "ebitda_margin": (
        ("ebitda", "revenue"),
        lambda values: values["ebitda"] / values["revenue"],
        "ebitda / revenue",
    ),
    "operating_cash_flow_ratio": (
        ("operating_cash_flow", "current_liabilities"),
        lambda values: (
            values["operating_cash_flow"]
            / values["current_liabilities"]
        ),
        "operating_cash_flow / current_liabilities",
    ),
}


def _select_period(
    metrics: List[FinancialMetric],
    reporting_period: Optional[str],
) -> Tuple[Optional[str], List[FinancialMetric], Optional[str]]:
    populated = [metric for metric in metrics if metric.value is not None]
    available_periods = sorted(
        {
            metric.reporting_period
            for metric in populated
            if metric.reporting_period is not None
        }
    )

    if reporting_period is not None:
        selected = [
            metric
            for metric in populated
            if metric.reporting_period == reporting_period
        ]
        if not selected:
            return (
                reporting_period,
                [],
                f"No metrics found for reporting period {reporting_period}.",
            )
        return reporting_period, selected, None

    if len(available_periods) > 1:
        return (
            None,
            [],
            "Multiple reporting periods are available; select one explicitly.",
        )

    selected_period = available_periods[0] if available_periods else None
    selected = [
        metric
        for metric in populated
        if metric.reporting_period in {selected_period, None}
    ]
    return selected_period, selected, None


def calculate_financial_ratios(
    metrics: List[FinancialMetric],
    reporting_period: Optional[str] = None,
    selected_ratios: Optional[List[str]] = None,
) -> ToolExecution:
    validation = validate_financial_metrics(metrics)

    if not validation.is_valid:
        return ToolExecution(
            tool_name="calculate_financial_ratios",
            status="validation_failed",
            missing_inputs=[],
            outputs={
                "validation_issues": [
                    issue.model_dump(mode="json")
                    for issue in validation.issues
                ]
            },
            error="Financial metrics failed validation.",
        )

    period, period_metrics, period_error = _select_period(
        validation.valid_metrics,
        reporting_period,
    )

    if period_error:
        return ToolExecution(
            tool_name="calculate_financial_ratios",
            status="failed",
            outputs={"available_period": period},
            error=period_error,
        )

    metric_map = {
        metric.name: metric
        for metric in period_metrics
        if metric.value is not None
    }
    value_map = {
        name: metric.value
        for name, metric in metric_map.items()
    }

    requested = selected_ratios or list(RATIO_DEFINITIONS)
    unknown = sorted(set(requested) - set(RATIO_DEFINITIONS))

    if unknown:
        return ToolExecution(
            tool_name="calculate_financial_ratios",
            status="failed",
            error=f"Unknown ratios requested: {', '.join(unknown)}",
        )

    calculations = {}
    all_missing = set()
    all_evidence_ids = set()

    for ratio_name in requested:
        required, calculator, formula = RATIO_DEFINITIONS[ratio_name]
        missing = [
            metric_name
            for metric_name in required
            if metric_name not in value_map
        ]

        if missing:
            all_missing.update(missing)
            calculations[ratio_name] = {
                "status": "insufficient_data",
                "formula": formula,
                "required_metrics": list(required),
                "missing_metrics": missing,
                "value": None,
            }
            continue

        denominator_name = required[-1]
        if value_map[denominator_name] == 0:
            calculations[ratio_name] = {
                "status": "invalid_denominator",
                "formula": formula,
                "required_metrics": list(required),
                "missing_metrics": [],
                "value": None,
                "error": f"{denominator_name} cannot be zero.",
            }
            continue

        evidence_ids = sorted(
            {
                evidence_id
                for metric_name in required
                for evidence_id in metric_map[metric_name].evidence_ids
            }
        )
        all_evidence_ids.update(evidence_ids)

        calculations[ratio_name] = {
            "status": "calculated",
            "formula": formula,
            "inputs": {
                metric_name: value_map[metric_name]
                for metric_name in required
            },
            "value": round(calculator(value_map), 6),
            "evidence_ids": evidence_ids,
        }

    successful = sum(
        calculation["status"] == "calculated"
        for calculation in calculations.values()
    )

    if successful == len(requested):
        status = "completed"
    elif successful > 0:
        status = "partial"
    else:
        status = "insufficient_data"

    return ToolExecution(
        tool_name="calculate_financial_ratios",
        status=status,
        inputs={
            "reporting_period": period,
            "selected_ratios": requested,
        },
        outputs={"calculations": calculations},
        missing_inputs=sorted(all_missing),
        evidence_ids=sorted(all_evidence_ids),
    )