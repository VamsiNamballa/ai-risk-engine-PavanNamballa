from typing import List

from service.engine.risk_scoring import (
    calculate_explainable_risk_score,
)

from service.agents.risk_analysis_planner import (
    create_risk_analysis_plan,
)
from service.models import (
    ConfidenceLevel,
    RiskAnalysisInput,
    RiskAnalysisResult,
    RiskFactor,
    RiskLevel,
    ToolExecution,
)
from service.tools.financial_ratio_tools import (
    calculate_financial_ratios,
)
from service.tools.financial_risk_models import (
    calculate_altman_z_score,
    calculate_beneish_m_score,
)
from service.tools.financial_trend_tools import (
    analyze_financial_trends,
)


MAX_AGENT_ITERATIONS = 1


def _execute_tool(
    tool_name: str,
    payload: RiskAnalysisInput,
    reporting_periods: List[str],
) -> ToolExecution:
    try:
        if tool_name == "calculate_financial_ratios":
            latest_period = (
                reporting_periods[-1]
                if reporting_periods
                else None
            )
            return calculate_financial_ratios(
                metrics=payload.metrics,
                reporting_period=latest_period,
            )

        if tool_name == "calculate_beneish_m_score":
            if len(reporting_periods) < 2:
                return ToolExecution(
                    tool_name=tool_name,
                    status="insufficient_data",
                    missing_inputs=["at least two reporting periods"],
                    error=(
                        "Beneish analysis requires current and "
                        "previous reporting periods."
                    ),
                )

            return calculate_beneish_m_score(
                metrics=payload.metrics,
                previous_period=reporting_periods[-2],
                current_period=reporting_periods[-1],
            )

        if tool_name == "calculate_altman_z_score":
            if not reporting_periods:
                return ToolExecution(
                    tool_name=tool_name,
                    status="insufficient_data",
                    missing_inputs=["reporting_period"],
                    error="Altman analysis requires a reporting period.",
                )

            return calculate_altman_z_score(
                metrics=payload.metrics,
                reporting_period=reporting_periods[-1],
            )

        if tool_name == "analyze_financial_trends":
            return analyze_financial_trends(payload.metrics)

        return ToolExecution(
            tool_name=tool_name,
            status="failed",
            error=f"No registered tool named {tool_name}.",
        )
    except Exception as exc:
        return ToolExecution(
            tool_name=tool_name,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _beneish_findings(
    execution: ToolExecution,
) -> tuple[List[RiskFactor], List[RiskFactor]]:
    risks: List[RiskFactor] = []
    protections: List[RiskFactor] = []

    if execution.status != "completed":
        return risks, protections

    possible_manipulator = execution.outputs.get(
        "possible_manipulator"
    )
    score = execution.outputs.get("score")

    if possible_manipulator:
        risks.append(
            RiskFactor(
                name="Beneish manipulation signal",
                description=(
                    f"Beneish M-Score {score} exceeds the configured "
                    "manipulation threshold."
                ),
                severity=RiskLevel.HIGH,
                score_impact=25,
                evidence_ids=execution.evidence_ids,
            )
        )
    else:
        protections.append(
            RiskFactor(
                name="No Beneish manipulation signal",
                description=(
                    f"Beneish M-Score {score} does not exceed the "
                    "configured manipulation threshold."
                ),
                severity=RiskLevel.LOW,
                score_impact=10,
                evidence_ids=execution.evidence_ids,
            )
        )

    return risks, protections


def _altman_findings(
    execution: ToolExecution,
) -> tuple[List[RiskFactor], List[RiskFactor]]:
    risks: List[RiskFactor] = []
    protections: List[RiskFactor] = []

    if execution.status != "completed":
        return risks, protections

    zone = execution.outputs.get("zone")
    score = execution.outputs.get("score")

    if zone == "distress":
        risks.append(
            RiskFactor(
                name="Altman distress-zone result",
                description=(
                    f"Altman Z-Score {score} falls in the "
                    "financial-distress zone."
                ),
                severity=RiskLevel.CRITICAL,
                score_impact=30,
                evidence_ids=execution.evidence_ids,
            )
        )
    elif zone == "grey":
        risks.append(
            RiskFactor(
                name="Altman grey-zone result",
                description=(
                    f"Altman Z-Score {score} falls in the grey zone."
                ),
                severity=RiskLevel.MODERATE,
                score_impact=15,
                evidence_ids=execution.evidence_ids,
            )
        )
    elif zone == "safe":
        protections.append(
            RiskFactor(
                name="Altman safe-zone result",
                description=(
                    f"Altman Z-Score {score} falls in the safe zone."
                ),
                severity=RiskLevel.LOW,
                score_impact=15,
                evidence_ids=execution.evidence_ids,
            )
        )

    return risks, protections


def _trend_findings(
    execution: ToolExecution,
) -> tuple[List[RiskFactor], List[RiskFactor]]:
    risks: List[RiskFactor] = []
    protections: List[RiskFactor] = []

    if execution.status != "completed":
        return risks, protections

    severity_map = {
        "low": RiskLevel.LOW,
        "medium": RiskLevel.MODERATE,
        "high": RiskLevel.HIGH,
        "critical": RiskLevel.CRITICAL,
    }

    impact_map = {
        RiskLevel.LOW: 5,
        RiskLevel.MODERATE: 10,
        RiskLevel.HIGH: 15,
        RiskLevel.CRITICAL: 20,
    }

    for signal in execution.outputs.get("risk_signals", []):
        severity = severity_map.get(
            signal.get("severity", "medium"),
            RiskLevel.MODERATE,
        )
        risks.append(
            RiskFactor(
                name=signal["code"],
                description=signal["message"],
                severity=severity,
                score_impact=impact_map[severity],
                evidence_ids=execution.evidence_ids,
            )
        )

    for signal in execution.outputs.get(
        "protective_signals",
        [],
    ):
        protections.append(
            RiskFactor(
                name=signal["code"],
                description=signal["message"],
                severity=RiskLevel.LOW,
                score_impact=5,
                evidence_ids=execution.evidence_ids,
            )
        )

    return risks, protections


def _extract_findings(
    executions: List[ToolExecution],
) -> tuple[List[RiskFactor], List[RiskFactor]]:
    risks: List[RiskFactor] = []
    protections: List[RiskFactor] = []

    for execution in executions:
        if execution.tool_name == "calculate_beneish_m_score":
            new_risks, new_protections = _beneish_findings(
                execution
            )
        elif execution.tool_name == "calculate_altman_z_score":
            new_risks, new_protections = _altman_findings(
                execution
            )
        elif execution.tool_name == "analyze_financial_trends":
            new_risks, new_protections = _trend_findings(
                execution
            )
        else:
            new_risks, new_protections = [], []

        risks.extend(new_risks)
        protections.extend(new_protections)

    return risks, protections


def run_risk_analysis_agent(
    payload: RiskAnalysisInput,
) -> RiskAnalysisResult:
    plan = create_risk_analysis_plan(payload)
    executions: List[ToolExecution] = []

    for selection in plan.selected_analyses:
        execution = _execute_tool(
            tool_name=selection.tool_name,
            payload=payload,
            reporting_periods=selection.reporting_periods,
        )
        executions.append(execution)

    risk_factors, protective_factors = _extract_findings(
        executions
    )

    missing_data = sorted(
        {
            missing
            for execution in executions
            for missing in execution.missing_inputs
        }
    )

    failures = [
        execution
        for execution in executions
        if execution.status
        in {
            "failed",
            "validation_failed",
            "insufficient_data",
        }
    ]

    conflicts = [
        execution.error
        for execution in failures
        if execution.error
    ]

    (
        overall_risk_score,
        risk_level,
        confidence,
        score_breakdown,
    ) = calculate_explainable_risk_score(
        executions=executions,
        missing_data_count=len(missing_data),
        conflict_count=len(conflicts),
    )         
    completed_count = sum(
        execution.status in {"completed", "partial"}
        for execution in executions
    )

    reasoning_summary = (
        f"Selected {len(plan.selected_analyses)} analyses and "
        f"completed {completed_count}. "
        f"Detected {len(risk_factors)} risk factors and "
        f"{len(protective_factors)} protective factors."
    )

    return RiskAnalysisResult(
        company_name=payload.company_name,
        selected_analyses=plan.selected_analyses,
        tool_executions=executions,
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        missing_data=missing_data,
        conflicts=conflicts,
        score_breakdown=score_breakdown,
        overall_risk_score=overall_risk_score,
        risk_level=risk_level,
        confidence=confidence,
        reasoning_summary=reasoning_summary,
        iteration_count=MAX_AGENT_ITERATIONS,
    )