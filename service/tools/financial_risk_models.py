from typing import Dict, List, Optional

from service.models import FinancialMetric, ToolExecution
from service.tools.financial_validation import validate_financial_metrics


BENEISH_THRESHOLD = -1.78
ALTMAN_DISTRESS_THRESHOLD = 1.81
ALTMAN_SAFE_THRESHOLD = 2.99


def _period_map(
    metrics: List[FinancialMetric],
    reporting_period: str,
) -> Dict[str, FinancialMetric]:
    return {
        metric.name: metric
        for metric in metrics
        if metric.reporting_period == reporting_period
        and metric.value is not None
    }


def _values(
    metric_map: Dict[str, FinancialMetric],
) -> Dict[str, float]:
    return {
        name: metric.value
        for name, metric in metric_map.items()
        if metric.value is not None
    }


def _evidence_ids(
    metric_map: Dict[str, FinancialMetric],
    required_metrics: List[str],
) -> List[str]:
    return sorted(
        {
            evidence_id
            for name in required_metrics
            if name in metric_map
            for evidence_id in metric_map[name].evidence_ids
        }
    )


def _missing(
    values: Dict[str, float],
    required: List[str],
) -> List[str]:
    return sorted(name for name in required if name not in values)


def _divide(
    numerator: float,
    denominator: float,
    label: str,
) -> float:
    if denominator == 0:
        raise ValueError(f"Cannot calculate {label}: denominator is zero.")
    return numerator / denominator


def calculate_beneish_m_score(
    metrics: List[FinancialMetric],
    current_period: str,
    previous_period: str,
) -> ToolExecution:
    validation = validate_financial_metrics(metrics)

    if not validation.is_valid:
        return ToolExecution(
            tool_name="calculate_beneish_m_score",
            status="validation_failed",
            outputs={
                "validation_issues": [
                    issue.model_dump(mode="json")
                    for issue in validation.issues
                ]
            },
            error="Financial metrics failed validation.",
        )

    required = [
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
    ]

    current_map = _period_map(validation.valid_metrics, current_period)
    previous_map = _period_map(validation.valid_metrics, previous_period)
    current = _values(current_map)
    previous = _values(previous_map)

    missing_current = _missing(current, required)
    missing_previous = _missing(previous, required)
    missing_inputs = [
        *[f"{current_period}:{name}" for name in missing_current],
        *[f"{previous_period}:{name}" for name in missing_previous],
    ]

    if missing_inputs:
        return ToolExecution(
            tool_name="calculate_beneish_m_score",
            status="insufficient_data",
            inputs={
                "current_period": current_period,
                "previous_period": previous_period,
            },
            missing_inputs=missing_inputs,
            error="Beneish M-Score requires two complete reporting periods.",
        )

    try:
        dsri = _divide(
            _divide(
                current["accounts_receivable"],
                current["revenue"],
                "current receivables-to-sales ratio",
            ),
            _divide(
                previous["accounts_receivable"],
                previous["revenue"],
                "previous receivables-to-sales ratio",
            ),
            "DSRI",
        )

        current_gross_margin = _divide(
            current["revenue"] - current["cost_of_goods_sold"],
            current["revenue"],
            "current gross margin",
        )
        previous_gross_margin = _divide(
            previous["revenue"] - previous["cost_of_goods_sold"],
            previous["revenue"],
            "previous gross margin",
        )
        gmi = _divide(previous_gross_margin, current_gross_margin, "GMI")

        current_asset_quality = 1 - _divide(
            current["current_assets"]
            + current["property_plant_equipment"],
            current["total_assets"],
            "current asset quality",
        )
        previous_asset_quality = 1 - _divide(
            previous["current_assets"]
            + previous["property_plant_equipment"],
            previous["total_assets"],
            "previous asset quality",
        )
        aqi = _divide(
            current_asset_quality,
            previous_asset_quality,
            "AQI",
        )

        sgi = _divide(
            current["revenue"],
            previous["revenue"],
            "SGI",
        )

        current_depreciation_rate = _divide(
            current["depreciation"],
            current["depreciation"]
            + current["property_plant_equipment"],
            "current depreciation rate",
        )
        previous_depreciation_rate = _divide(
            previous["depreciation"],
            previous["depreciation"]
            + previous["property_plant_equipment"],
            "previous depreciation rate",
        )
        depi = _divide(
            previous_depreciation_rate,
            current_depreciation_rate,
            "DEPI",
        )

        sgai = _divide(
            _divide(
                current["selling_general_admin_expense"],
                current["revenue"],
                "current SGA-to-sales ratio",
            ),
            _divide(
                previous["selling_general_admin_expense"],
                previous["revenue"],
                "previous SGA-to-sales ratio",
            ),
            "SGAI",
        )

        lvgi = _divide(
            _divide(
                current["total_debt"],
                current["total_assets"],
                "current leverage",
            ),
            _divide(
                previous["total_debt"],
                previous["total_assets"],
                "previous leverage",
            ),
            "LVGI",
        )

        tata = _divide(
            current["net_income"] - current["operating_cash_flow"],
            current["total_assets"],
            "TATA",
        )
    except ValueError as exc:
        return ToolExecution(
            tool_name="calculate_beneish_m_score",
            status="failed",
            error=str(exc),
        )

    indices = {
        "dsri": dsri,
        "gmi": gmi,
        "aqi": aqi,
        "sgi": sgi,
        "depi": depi,
        "sgai": sgai,
        "lvgi": lvgi,
        "tata": tata,
    }

    score = (
        -4.84
        + (0.920 * dsri)
        + (0.528 * gmi)
        + (0.404 * aqi)
        + (0.892 * sgi)
        + (0.115 * depi)
        - (0.172 * sgai)
        + (4.679 * tata)
        - (0.327 * lvgi)
    )

    possible_manipulator = score > BENEISH_THRESHOLD
    evidence = sorted(
        set(
            _evidence_ids(current_map, required)
            + _evidence_ids(previous_map, required)
        )
    )

    return ToolExecution(
        tool_name="calculate_beneish_m_score",
        status="completed",
        inputs={
            "current_period": current_period,
            "previous_period": previous_period,
        },
        outputs={
            "model": "Beneish M-Score eight-variable model",
            "indices": {
                name: round(value, 6)
                for name, value in indices.items()
            },
            "score": round(score, 6),
            "threshold": BENEISH_THRESHOLD,
            "possible_manipulator": possible_manipulator,
            "interpretation": (
                "Possible earnings manipulation"
                if possible_manipulator
                else "No manipulation signal from the Beneish threshold"
            ),
        },
        evidence_ids=evidence,
    )


def calculate_altman_z_score(
    metrics: List[FinancialMetric],
    reporting_period: str,
) -> ToolExecution:
    validation = validate_financial_metrics(metrics)

    if not validation.is_valid:
        return ToolExecution(
            tool_name="calculate_altman_z_score",
            status="validation_failed",
            outputs={
                "validation_issues": [
                    issue.model_dump(mode="json")
                    for issue in validation.issues
                ]
            },
            error="Financial metrics failed validation.",
        )

    required = [
        "current_assets",
        "current_liabilities",
        "total_assets",
        "retained_earnings",
        "ebit",
        "market_value_equity",
        "total_liabilities",
        "revenue",
    ]

    metric_map = _period_map(
        validation.valid_metrics,
        reporting_period,
    )
    values = _values(metric_map)
    missing_inputs = _missing(values, required)

    if missing_inputs:
        return ToolExecution(
            tool_name="calculate_altman_z_score",
            status="insufficient_data",
            inputs={"reporting_period": reporting_period},
            missing_inputs=missing_inputs,
            error="Altman Z-Score inputs are incomplete.",
        )

    try:
        working_capital = (
            values["current_assets"]
            - values["current_liabilities"]
        )
        x1 = _divide(
            working_capital,
            values["total_assets"],
            "working capital to total assets",
        )
        x2 = _divide(
            values["retained_earnings"],
            values["total_assets"],
            "retained earnings to total assets",
        )
        x3 = _divide(
            values["ebit"],
            values["total_assets"],
            "EBIT to total assets",
        )
        x4 = _divide(
            values["market_value_equity"],
            values["total_liabilities"],
            "market value of equity to total liabilities",
        )
        x5 = _divide(
            values["revenue"],
            values["total_assets"],
            "sales to total assets",
        )
    except ValueError as exc:
        return ToolExecution(
            tool_name="calculate_altman_z_score",
            status="failed",
            error=str(exc),
        )

    score = (
        (1.2 * x1)
        + (1.4 * x2)
        + (3.3 * x3)
        + (0.6 * x4)
        + x5
    )

    if score < ALTMAN_DISTRESS_THRESHOLD:
        zone = "distress"
    elif score <= ALTMAN_SAFE_THRESHOLD:
        zone = "grey"
    else:
        zone = "safe"

    return ToolExecution(
        tool_name="calculate_altman_z_score",
        status="completed",
        inputs={"reporting_period": reporting_period},
        outputs={
            "model": "Altman Z-Score for public manufacturing firms",
            "components": {
                "working_capital_to_total_assets": round(x1, 6),
                "retained_earnings_to_total_assets": round(x2, 6),
                "ebit_to_total_assets": round(x3, 6),
                "market_equity_to_total_liabilities": round(x4, 6),
                "sales_to_total_assets": round(x5, 6),
            },
            "score": round(score, 6),
            "zone": zone,
            "thresholds": {
                "distress_below": ALTMAN_DISTRESS_THRESHOLD,
                "safe_above": ALTMAN_SAFE_THRESHOLD,
            },
            "applicability": (
                "This model is intended for publicly traded "
                "manufacturing companies."
            ),
        },
        evidence_ids=_evidence_ids(metric_map, required),
    )