import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from service.models import (
    FinancialMetric,
    MetricValidationIssue,
    MetricValidationResult,
    ValidationSeverity,
)


SUPPORTED_METRICS = {
    "revenue",
    "net_income",
    "total_debt",
    "total_equity",
    "current_assets",
    "current_liabilities",
    "inventory",
    "total_assets",
    "ebit",
    "interest_expense",
    "ebitda",
    "operating_cash_flow",
}

NON_NEGATIVE_METRICS = {
    "revenue",
    "total_debt",
    "current_assets",
    "current_liabilities",
    "inventory",
    "total_assets",
    "interest_expense",
}

UNIT_ALIASES = {
    "usd": "USD",
    "$": "USD",
    "dollars": "USD",
    "thousand usd": "USD_THOUSANDS",
    "usd thousands": "USD_THOUSANDS",
    "millions": "USD_MILLIONS",
    "million usd": "USD_MILLIONS",
    "usd millions": "USD_MILLIONS",
}


def _normalized_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None

    cleaned = " ".join(unit.strip().lower().split())
    return UNIT_ALIASES.get(cleaned, cleaned.upper())


def _issue(
    code: str,
    message: str,
    severity: ValidationSeverity,
    metrics: List[FinancialMetric],
) -> MetricValidationIssue:
    periods = {
        metric.reporting_period
        for metric in metrics
        if metric.reporting_period
    }
    evidence_ids = sorted(
        {
            evidence_id
            for metric in metrics
            for evidence_id in metric.evidence_ids
        }
    )

    return MetricValidationIssue(
        code=code,
        message=message,
        severity=severity,
        metric_names=sorted({metric.name for metric in metrics}),
        reporting_period=next(iter(periods)) if len(periods) == 1 else None,
        evidence_ids=evidence_ids,
    )


def validate_financial_metrics(
    metrics: List[FinancialMetric],
) -> MetricValidationResult:
    issues: List[MetricValidationIssue] = []
    valid_metrics: List[FinancialMetric] = []

    populated_metrics = [
        metric for metric in metrics if metric.value is not None
    ]

    if not populated_metrics:
        issues.append(
            MetricValidationIssue(
                code="NO_FINANCIAL_DATA",
                message="No populated financial metrics were supplied.",
                severity=ValidationSeverity.ERROR,
            )
        )
        return MetricValidationResult(
            is_valid=False,
            valid_metrics=[],
            issues=issues,
        )

    for metric in populated_metrics:
        if metric.name not in SUPPORTED_METRICS:
            issues.append(
                _issue(
                    code="UNSUPPORTED_METRIC",
                    message=f"Unsupported financial metric: {metric.name}.",
                    severity=ValidationSeverity.WARNING,
                    metrics=[metric],
                )
            )
            continue

        if not math.isfinite(metric.value):
            issues.append(
                _issue(
                    code="NON_FINITE_VALUE",
                    message=f"{metric.name} must contain a finite number.",
                    severity=ValidationSeverity.ERROR,
                    metrics=[metric],
                )
            )
            continue

        if metric.name in NON_NEGATIVE_METRICS and metric.value < 0:
            issues.append(
                _issue(
                    code="UNEXPECTED_NEGATIVE_VALUE",
                    message=f"{metric.name} cannot normally be negative.",
                    severity=ValidationSeverity.ERROR,
                    metrics=[metric],
                )
            )
            continue

        if metric.name == "total_equity" and metric.value < 0:
            issues.append(
                _issue(
                    code="NEGATIVE_EQUITY",
                    message=(
                        "Negative total equity is allowed but represents "
                        "a significant financial-risk indicator."
                    ),
                    severity=ValidationSeverity.WARNING,
                    metrics=[metric],
                )
            )

        valid_metrics.append(metric)

    grouped: Dict[Tuple[str, Optional[str]], List[FinancialMetric]] = (
        defaultdict(list)
    )
    for metric in valid_metrics:
        grouped[(metric.name, metric.reporting_period)].append(metric)

    for (name, period), duplicates in grouped.items():
        if len(duplicates) < 2:
            continue

        values = {metric.value for metric in duplicates}
        if len(values) > 1:
            issues.append(
                _issue(
                    code="CONFLICTING_DUPLICATE_VALUES",
                    message=(
                        f"Conflicting values were supplied for {name} "
                        f"in reporting period {period or 'unspecified'}."
                    ),
                    severity=ValidationSeverity.ERROR,
                    metrics=duplicates,
                )
            )
        else:
            issues.append(
                _issue(
                    code="DUPLICATE_METRIC",
                    message=(
                        f"Duplicate values were supplied for {name} "
                        f"in reporting period {period or 'unspecified'}."
                    ),
                    severity=ValidationSeverity.WARNING,
                    metrics=duplicates,
                )
            )

    units_by_metric: Dict[str, set] = defaultdict(set)
    metrics_by_name: Dict[str, List[FinancialMetric]] = defaultdict(list)

    for metric in valid_metrics:
        metrics_by_name[metric.name].append(metric)
        normalized_unit = _normalized_unit(metric.unit)
        if normalized_unit:
            units_by_metric[metric.name].add(normalized_unit)

    for name, units in units_by_metric.items():
        if len(units) > 1:
            issues.append(
                _issue(
                    code="INCONSISTENT_UNITS",
                    message=(
                        f"{name} uses inconsistent units: "
                        f"{', '.join(sorted(units))}."
                    ),
                    severity=ValidationSeverity.ERROR,
                    metrics=metrics_by_name[name],
                )
            )

    by_period: Dict[Optional[str], Dict[str, FinancialMetric]] = (
        defaultdict(dict)
    )
    for metric in valid_metrics:
        by_period[metric.reporting_period][metric.name] = metric

    for period, period_metrics in by_period.items():
        current_assets = period_metrics.get("current_assets")
        total_assets = period_metrics.get("total_assets")
        inventory = period_metrics.get("inventory")

        if (
            current_assets
            and total_assets
            and current_assets.value > total_assets.value
        ):
            issues.append(
                _issue(
                    code="CURRENT_ASSETS_EXCEED_TOTAL_ASSETS",
                    message=(
                        "Current assets cannot exceed total assets for "
                        "the same reporting period."
                    ),
                    severity=ValidationSeverity.ERROR,
                    metrics=[current_assets, total_assets],
                )
            )

        if (
            inventory
            and current_assets
            and inventory.value > current_assets.value
        ):
            issues.append(
                _issue(
                    code="INVENTORY_EXCEEDS_CURRENT_ASSETS",
                    message=(
                        "Inventory cannot exceed current assets for "
                        "the same reporting period."
                    ),
                    severity=ValidationSeverity.ERROR,
                    metrics=[inventory, current_assets],
                )
            )

    error_exists = any(
        issue.severity == ValidationSeverity.ERROR
        for issue in issues
    )

    return MetricValidationResult(
        is_valid=not error_exists,
        valid_metrics=valid_metrics,
        issues=issues,
    )