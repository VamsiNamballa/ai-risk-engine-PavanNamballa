"""
Risk Analysis Agent Data Schemas
Defines strict Pydantic models for inputs from Document Intelligence Agent
and outputs to Report Generation Agent.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class FinancialMetricsInput(BaseModel):
    """Input payload expected from Document Intelligence Agent or user upload."""
    model_config = ConfigDict(extra="ignore")

    company_name: str = Field(default="Unknown Company", description="Name of the company being analyzed")
    industry_sector: str = Field(default="General", description="Industry sector (e.g., Technology, Retail, Utilities)")
    reporting_year: Optional[int] = Field(default=None, description="Fiscal year of financial statements")

    # Core Financial Metrics (in floating-point currency units)
    revenue: Optional[float] = Field(default=None, description="Total Revenue / Sales")
    net_income: Optional[float] = Field(default=None, description="Net Income / Profit after tax")
    total_debt: Optional[float] = Field(default=None, description="Total Debt / Total Liabilities")
    total_equity: Optional[float] = Field(default=None, description="Total Shareholders' Equity")
    total_assets: Optional[float] = Field(default=None, description="Total Assets")
    current_assets: Optional[float] = Field(default=None, description="Total Current Assets")
    current_liabilities: Optional[float] = Field(default=None, description="Total Current Liabilities")
    inventory: Optional[float] = Field(default=None, description="Inventory Value")
    ebit: Optional[float] = Field(default=None, description="Operating Income / EBIT")
    interest_expense: Optional[float] = Field(default=None, description="Total Interest Expense")
    ebitda: Optional[float] = Field(default=None, description="EBITDA")
    operating_cash_flow: Optional[float] = Field(default=None, description="Operating Cash Flow")

    # Extended metrics for distress modeling (Altman Z-Score)
    retained_earnings: Optional[float] = Field(default=None, description="Retained Earnings")
    market_val_equity: Optional[float] = Field(default=None, description="Market Value of Equity / Market Cap")


class ComputedRatios(BaseModel):
    """Financial ratios computed by the Risk Analysis Agent."""
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_ratio: Optional[float] = None
    interest_coverage_ratio: Optional[float] = None
    net_margin: Optional[float] = None
    roa: Optional[float] = None
    roe: Optional[float] = None
    ebitda_margin: Optional[float] = None
    cash_flow_ratio: Optional[float] = None


class CategoryScores(BaseModel):
    """Weighted risk sub-scores across core financial pillars (0-100)."""
    solvency: Optional[float] = Field(default=None, description="Solvency & Default Risk score (35% weight)")
    liquidity: Optional[float] = Field(default=None, description="Liquidity Risk score (30% weight)")
    profitability: Optional[float] = Field(default=None, description="Profitability Risk score (20% weight)")
    efficiency: Optional[float] = Field(default=None, description="Asset Efficiency score (15% weight)")


class AltmanZScoreResult(BaseModel):
    """Altman Z-Score bankruptcy prediction model output."""
    score: Optional[float] = Field(default=None, description="Computed Z-Score value")
    zone: str = Field(default="Unknown", description="Bankruptcy Risk Zone: Safe | Grey | Distress | Unknown")
    interpretation: str = Field(default="Insufficient data to compute Altman Z-Score", description="Human-readable description")


class RiskAnalysisResult(BaseModel):
    """Final output payload produced by Risk Analysis Agent for Report Generation Agent."""
    company_name: str
    industry_sector: str
    reporting_year: Optional[int] = None
    
    risk_score: Optional[int] = Field(default=None, ge=0, le=100, description="Overall weighted health score (0-100)")
    risk_level: str = Field(default="Unknown", description="Low | Moderate | High | Critical | Unknown")
    data_confidence: float = Field(default=0.0, ge=0.0, le=100.0, description="Completeness of input metrics %")
    
    circuit_breakers_triggered: List[str] = Field(default_factory=list, description="Red-flag override codes")
    category_scores: CategoryScores = Field(default_factory=CategoryScores)
    altman_z_score: AltmanZScoreResult = Field(default_factory=AltmanZScoreResult)
    computed_ratios: ComputedRatios = Field(default_factory=ComputedRatios)
    
    risk_flags: List[str] = Field(default_factory=list, description="Detailed warning flags")
    actionable_recommendations: List[str] = Field(default_factory=list, description="Targeted recommendations")
