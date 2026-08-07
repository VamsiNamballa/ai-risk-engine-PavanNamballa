from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"
    UNKNOWN = "Unknown"


class ConfidenceLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class EvidenceItem(BaseModel):
    evidence_id: str = Field(min_length=1)
    document_id: Optional[str] = None
    filename: Optional[str] = None
    page_number: Optional[int] = Field(default=None, ge=1)
    reporting_period: Optional[str] = None
    text: str = Field(min_length=1)


class FinancialMetric(BaseModel):
    name: str = Field(min_length=1)
    value: Optional[float] = None
    unit: Optional[str] = None
    reporting_period: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)


class RiskAnalysisInput(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = Field(default="default", min_length=1)
    company_name: Optional[str] = None
    metrics: List[FinancialMetric] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)


class AnalysisSelection(BaseModel):
    analysis_name: str
    reason: str
    required_metrics: List[str] = Field(default_factory=list)


class ToolExecution(BaseModel):
    tool_name: str
    status: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    missing_inputs: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class RiskFactor(BaseModel):
    name: str
    description: str
    severity: RiskLevel
    score_impact: float = Field(ge=0, le=100)
    evidence_ids: List[str] = Field(default_factory=list)


class ScoreComponent(BaseModel):
    category: str
    score: float = Field(ge=0, le=100)
    weight: float = Field(ge=0, le=1)
    weighted_score: float = Field(ge=0, le=100)
    explanation: str


class RiskAnalysisResult(BaseModel):
    company_name: Optional[str] = None
    selected_analyses: List[AnalysisSelection] = Field(default_factory=list)
    tool_executions: List[ToolExecution] = Field(default_factory=list)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    protective_factors: List[RiskFactor] = Field(default_factory=list)
    missing_data: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    score_breakdown: List[ScoreComponent] = Field(default_factory=list)

    # Score semantics: 0 means lowest risk; 100 means highest risk.
    overall_risk_score: Optional[float] = Field(default=None, ge=0, le=100)
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    reasoning_summary: str = ""
    iteration_count: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def require_score_for_known_risk(self):
        if (
            self.risk_level != RiskLevel.UNKNOWN
            and self.overall_risk_score is None
        ):
            raise ValueError(
                "overall_risk_score is required when risk_level is known"
            )
        return self