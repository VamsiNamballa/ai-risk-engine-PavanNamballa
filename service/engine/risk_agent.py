"""
Financial Risk Analysis Agent — Primary Orchestrator Entry Point

Exposes clean interface for analyzing financial risk from structured input metrics.
Integrated with Ratio Engine, Altman Z-Score Distress Model, Circuit Breakers,
and Category-Weighted Scoring Framework.
"""

from typing import Union, Dict, Any
from service.engine.schemas import FinancialMetricsInput, RiskAnalysisResult
from service.engine.scoring_engine import analyze_financial_risk as _analyze_risk


def analyze_risk_agent(input_payload: Union[FinancialMetricsInput, Dict[str, Any]]) -> RiskAnalysisResult:
    """
    Main Risk Analysis Agent Entry Point.
    
    Accepts either a FinancialMetricsInput object or a raw dictionary.
    Returns a validated RiskAnalysisResult payload.
    """
    if isinstance(input_payload, dict):
        # Allow nested dict under "metrics" or flat dict
        if "metrics" in input_payload and isinstance(input_payload["metrics"], dict):
            metrics_dict = input_payload["metrics"]
            metrics_dict["company_name"] = input_payload.get("company_name", metrics_dict.get("company_name", "Unknown Company"))
            metrics_dict["industry_sector"] = input_payload.get("industry_sector", metrics_dict.get("industry_sector", "General"))
            metrics_dict["reporting_year"] = input_payload.get("reporting_year", metrics_dict.get("reporting_year"))
            data = FinancialMetricsInput(**metrics_dict)
        else:
            data = FinancialMetricsInput(**input_payload)
    else:
        data = input_payload

    return _analyze_risk(data)


# Backwards compatibility helper for existing legacy imports
def analyze_financial_risk(financial_data: dict) -> dict:
    """Legacy compatibility function returning raw dict format."""
    res = analyze_risk_agent(financial_data)
    return res.model_dump()