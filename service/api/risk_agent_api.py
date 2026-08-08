from fastapi import APIRouter, HTTPException

from service.agents import run_risk_analysis_agent
from service.models import (
    RiskAnalysisInput,
    RiskAnalysisResult,
)


router = APIRouter(
    prefix="/agents/risk-analysis",
    tags=["Risk Analysis Agent"],
)


@router.post(
    "",
    response_model=RiskAnalysisResult,
    summary="Run the Risk Analysis Agent",
)
def analyze_financial_risk(
    payload: RiskAnalysisInput,
) -> RiskAnalysisResult:
    try:
        return run_risk_analysis_agent(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Risk Analysis Agent failed unexpectedly: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


@router.get(
    "/capabilities",
    summary="List Risk Analysis Agent capabilities",
)
def risk_analysis_capabilities():
    return {
        "agent": "Risk Analysis Agent",
        "score_semantics": {
            "minimum": 0,
            "maximum": 100,
            "meaning": "Higher values indicate higher financial risk.",
        },
        "capabilities": [
            {
                "name": "Financial ratio analysis",
                "tool": "calculate_financial_ratios",
            },
            {
                "name": "Beneish manipulation analysis",
                "tool": "calculate_beneish_m_score",
            },
            {
                "name": "Altman financial-distress analysis",
                "tool": "calculate_altman_z_score",
            },
            {
                "name": "Multi-period trend analysis",
                "tool": "analyze_financial_trends",
            },
        ],
        "limitations": [
            (
                "Beneish analysis requires two sufficiently complete "
                "reporting periods."
            ),
            (
                "The classic Altman model is intended for publicly "
                "traded manufacturing firms."
            ),
            (
                "The agent does not retrieve documents; it analyzes "
                "structured financial metrics and supplied evidence."
            ),
        ],
    }
