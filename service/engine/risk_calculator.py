"""
Financial Risk Calculator
Extracts financial metrics from document text, computes ratios, and
classifies overall risk with threshold-based flags.
"""

import json
import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

METRIC_FIELDS = [
    "revenue", "net_income", "total_debt", "total_equity",
    "current_assets", "current_liabilities", "inventory",
    "total_assets", "ebit", "interest_expense", "ebitda",
    "operating_cash_flow",
]

_EXTRACTION_PROMPT = """Read the financial document excerpts below and extract these
numeric fields if present: {fields}

Document Content:
{context}

Reply ONLY with valid JSON mapping each field name to a number (no currency symbols,
no commas). Use null for any field not found in the text. No markdown, no explanation.
Example: {{"revenue": 500000, "net_income": null, ...}}"""


def extract_financial_metrics(chunks: list[str]) -> dict:
    """Ask the LLM (or regex) to pull structured numbers: revenue, net income,
    total debt, total equity, current assets, current liabilities, etc."""
    if not chunks:
        return {field: None for field in METRIC_FIELDS}

    context = "\n\n".join(chunks)
    prompt = _EXTRACTION_PROMPT.format(fields=", ".join(METRIC_FIELDS), context=context)

    res = {field: None for field in METRIC_FIELDS}
    if GROQ_API_KEY:
        try:
            res = _extract_via_groq(prompt)
        except Exception:
            pass  # fall through to regex fallback

    # Merge regex extraction fallback for any missing/null fields
    regex_res = _extract_via_regex(context)
    for field in METRIC_FIELDS:
        if res.get(field) is None and regex_res.get(field) is not None:
            res[field] = regex_res[field]

    return res


def _extract_via_groq(prompt: str) -> dict:
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a precise financial data extraction assistant. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        },
    )
    if not response.ok:
        raise RuntimeError(f"Groq {response.status_code}: {response.text}")
    data = json.loads(response.json()["choices"][0]["message"]["content"])
    return {field: data.get(field) for field in METRIC_FIELDS}


def _extract_via_regex(text: str) -> dict:
    """Best-effort fallback if no LLM key is set. Looks for 'Label: $1,234' or
    'Label 1,234' style patterns near each field's common phrasing."""
    patterns = {
        "revenue": r"(?:total revenue|net revenue|revenue)[:\s]+\$?([\d,]+\.?\d*)",
        "net_income": r"net income[:\s]+\$?([\d,]+\.?\d*)",
        "total_debt": r"total (?:debt|liabilities)[:\s]+\$?([\d,]+\.?\d*)",
        "total_equity": r"(?:total )?(?:shareholders'? |stockholders'? )?equity[:\s]+\$?([\d,]+\.?\d*)",
        "current_assets": r"total current assets[:\s]+\$?([\d,]+\.?\d*)",
        "current_liabilities": r"total current liabilities[:\s]+\$?([\d,]+\.?\d*)",
        "inventory": r"inventor(?:y|ies)[:\s]+\$?([\d,]+\.?\d*)",
        "total_assets": r"total assets[:\s]+\$?([\d,]+\.?\d*)",
        "ebit": r"\bebit\b(?!da)[:\s]+\$?([\d,]+\.?\d*)",
        "interest_expense": r"interest expense[:\s]+\$?([\d,]+\.?\d*)",
        "ebitda": r"ebitda[:\s]+\$?([\d,]+\.?\d*)",
        "operating_cash_flow": r"(?:cash flow from operations|operating cash flow)[:\s]+\$?([\d,]+\.?\d*)",
    }
    result = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        result[field] = float(match.group(1).replace(",", "")) if match else None
    return result


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def calculate_ratios(metrics: dict) -> dict:
    """Debt-to-equity, current ratio, quick ratio, ROE, ROA, net margin."""
    g = metrics.get

    quick_assets = None
    if g("current_assets") is not None and g("inventory") is not None:
        quick_assets = g("current_assets") - g("inventory")

    return {
        "current_ratio": _safe_div(g("current_assets"), g("current_liabilities")),
        "quick_ratio": _safe_div(quick_assets, g("current_liabilities")),
        "debt_to_equity": _safe_div(g("total_debt"), g("total_equity")),
        "debt_ratio": _safe_div(g("total_debt"), g("total_assets")),
        "interest_coverage_ratio": _safe_div(g("ebit"), g("interest_expense")),
        "net_margin": _safe_div(g("net_income"), g("revenue")),
        "roa": _safe_div(g("net_income"), g("total_assets")),
        "roe": _safe_div(g("net_income"), g("total_equity")),
        "ebitda_margin": _safe_div(g("ebitda"), g("revenue")),
        "cash_flow_ratio": _safe_div(g("operating_cash_flow"), g("current_liabilities")),
    }


# ratio_name -> (bad_threshold, warning_threshold, direction)
# direction "low" = risky when value is LOW; "high" = risky when value is HIGH
_RISK_RULES = {
    "current_ratio":            (1.0, 1.5, "low"),
    "quick_ratio":               (0.5, 1.0, "low"),
    "debt_to_equity":           (2.0, 1.0, "high"),
    "debt_ratio":                (0.7, 0.5, "high"),
    "interest_coverage_ratio":  (1.5, 3.0, "low"),
    "net_margin":                (0.0, 0.05, "low"),
    "roa":                        (0.0, 0.05, "low"),
    "roe":                        (0.0, 0.10, "low"),
    "ebitda_margin":             (0.0, 0.10, "low"),
    "cash_flow_ratio":           (0.5, 1.0, "low"),
}

_FLAG_MESSAGES = {
    "current_ratio": "Current ratio indicates weak short-term liquidity",
    "quick_ratio": "Quick ratio shows limited ability to cover liabilities without selling inventory",
    "debt_to_equity": "High leverage relative to equity",
    "debt_ratio": "Large share of assets financed by debt",
    "interest_coverage_ratio": "Low earnings coverage of interest expense",
    "net_margin": "Weak net profit margin",
    "roa": "Low return on assets",
    "roe": "Low return on equity",
    "ebitda_margin": "EBITDA margin below healthy range",
    "cash_flow_ratio": "Weak operating cash coverage of current liabilities",
}


def classify_risk(ratios: dict) -> dict:
    """Apply thresholds -> {risk_score: int, risk_level: str, flags: [...]}"""
    points = []
    flags = []

    for name, value in ratios.items():
        if value is None or name not in _RISK_RULES:
            continue

        bad, warn, direction = _RISK_RULES[name]

        if direction == "low":
            if value <= bad:
                points.append(0)
                flags.append(_FLAG_MESSAGES[name])
            elif value <= warn:
                points.append(50)
                flags.append(_FLAG_MESSAGES[name])
            else:
                points.append(100)
        else:  # "high" is risky
            if value >= bad:
                points.append(0)
                flags.append(_FLAG_MESSAGES[name])
            elif value >= warn:
                points.append(50)
                flags.append(_FLAG_MESSAGES[name])
            else:
                points.append(100)

    if not points:
        return {"risk_score": None, "risk_level": "Unknown", "flags": ["Insufficient data to assess risk"]}

    risk_score = round(sum(points) / len(points))

    if risk_score >= 80:
        risk_level = "Low"
    elif risk_score >= 60:
        risk_level = "Moderate"
    elif risk_score >= 40:
        risk_level = "High"
    else:
        risk_level = "Critical"

    return {"risk_score": risk_score, "risk_level": risk_level, "flags": flags}

if __name__ == "__main__":
    sample_metrics = {
        "revenue": 500000,
        "net_income": 45000,
        "total_debt": 300000,
        "total_equity": 200000,
        "current_assets": 150000,
        "current_liabilities": 100000,
        "inventory": 40000,
        "total_assets": 500000,
        "ebit": 60000,
        "interest_expense": 20000,
        "ebitda": 80000,
        "operating_cash_flow": 55000,
    }

    print("=== HEALTHY COMPANY ===")
    ratios = calculate_ratios(sample_metrics)
    for k, v in ratios.items():
        print(f"  {k}: {v}")
    result = classify_risk(ratios)
    print(f"  -> risk_score: {result['risk_score']}, risk_level: {result['risk_level']}")
    print(f"  -> flags: {result['flags']}")

    print("\n=== RISKY COMPANY ===")
    risky_metrics = {
        "revenue": 500000,
        "net_income": -20000,       # losing money
        "total_debt": 800000,       # heavily leveraged
        "total_equity": 100000,
        "current_assets": 60000,
        "current_liabilities": 120000,  # can't cover short-term debts
        "inventory": 30000,
        "total_assets": 500000,
        "ebit": -5000,
        "interest_expense": 40000,  # can't cover interest
        "ebitda": 15000,
        "operating_cash_flow": -10000,
    }
    ratios2 = calculate_ratios(risky_metrics)
    for k, v in ratios2.items():
        print(f"  {k}: {v}")
    result2 = classify_risk(ratios2)
    print(f"  -> risk_score: {result2['risk_score']}, risk_level: {result2['risk_level']}")
    print(f"  -> flags: {result2['flags']}")