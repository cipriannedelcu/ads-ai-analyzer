import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

# -----------------------------
# Env
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Ads-Correct AI Analysis Service (Evidence-Aware + Check-Constrained)")

# -----------------------------
# Types
# -----------------------------
DriverType = Literal[
    "query_mix_shift",
    "auction_pressure",
    "budget_limited",
    "tracking_issue",
    "landing_page_issue",
    "creative_fatigue",
    "seasonality",
    "unknown",
]

class AIReasoning(BaseModel):
    summary: str
    drivers: List[DriverType]
    recommended_checks: List[str]

# -----------------------------
# Input contract (Ads-correct)
# -----------------------------
class EvidenceInput(BaseModel):
    # Query mix / intent shifts
    new_queries_spend_pct: Optional[float] = Field(None, ge=0, le=100)
    brand_share_clicks_change_pp: Optional[float] = Field(None)

    # Auction pressure (deltas vs baseline)
    search_impression_share_change_pp: Optional[float] = Field(None)
    abs_top_is_change_pp: Optional[float] = Field(None)

    # Budget limitation
    budget_limited: Optional[bool] = Field(None)

    # Tracking / measurement
    tracking_change_recent: Optional[bool] = Field(None)
    conversion_drop_sudden: Optional[bool] = Field(None)

    # Landing page / UX
    landing_page_change_recent: Optional[bool] = Field(None)
    page_speed_change_pct: Optional[float] = Field(None)

    # Creative fatigue
    ctr_change_pp: Optional[float] = Field(None)
    top_ads_same_for_days: Optional[int] = Field(None, ge=0)

    # Seasonality context
    seasonality_known: Optional[bool] = Field(None)

class MetricsInput(BaseModel):
    clicks: int = Field(..., ge=0)
    conversions: float = Field(..., ge=0)  # fractional allowed
    cost: float = Field(..., ge=0)         # spend/cost
    evidence: Optional[EvidenceInput] = None

# -----------------------------
# Deterministic metric computation (NO AI MATH)
# -----------------------------
def compute_metrics(clicks: int, conversions: float, cost: float) -> dict:
    conversion_rate = (conversions / clicks * 100.0) if clicks > 0 else 0.0
    cost_per_click = (cost / clicks) if clicks > 0 else 0.0
    cost_per_conversion = (cost / conversions) if conversions > 0 else None

    return {
        "conversion_rate": round(conversion_rate, 4),
        "cost_per_click": round(cost_per_click, 4),
        "cost_per_conversion": (round(cost_per_conversion, 4) if cost_per_conversion is not None else None),
    }

# -----------------------------
# Evidence -> allowed drivers (CODE GOVERNS)
# -----------------------------
def infer_allowed_drivers(e: Optional[EvidenceInput]) -> List[DriverType]:
    if e is None:
        return []

    allowed: List[DriverType] = []

    # Query mix shift evidence
    if (e.new_queries_spend_pct is not None and e.new_queries_spend_pct >= 15.0) or \
       (e.brand_share_clicks_change_pp is not None and abs(e.brand_share_clicks_change_pp) >= 10.0):
        allowed.append("query_mix_shift")

    # Auction pressure evidence
    if (e.search_impression_share_change_pp is not None and e.search_impression_share_change_pp <= -5.0) or \
       (e.abs_top_is_change_pp is not None and e.abs_top_is_change_pp <= -5.0):
        allowed.append("auction_pressure")

    # Budget limitation evidence
    if e.budget_limited is True:
        allowed.append("budget_limited")

    # Tracking evidence
    if e.tracking_change_recent is True or e.conversion_drop_sudden is True:
        allowed.append("tracking_issue")

    # Landing page evidence
    if e.landing_page_change_recent is True or (e.page_speed_change_pct is not None and e.page_speed_change_pct >= 20.0):
        allowed.append("landing_page_issue")

    # Creative fatigue evidence
    if (e.ctr_change_pp is not None and e.ctr_change_pp <= -1.0) or (e.top_ads_same_for_days is not None and e.top_ads_same_for_days >= 30):
        allowed.append("creative_fatigue")

    # Seasonality evidence
    if e.seasonality_known is True:
        allowed.append("seasonality")

    # Deduplicate preserve order
    deduped: List[DriverType] = []
    for d in allowed:
        if d not in deduped:
            deduped.append(d)
    return deduped

# -----------------------------
# Recommended check constraints (CODE GOVERNS)
# -----------------------------
CHECK_KEYWORDS_BY_DRIVER: Dict[str, List[str]] = {
    "query_mix_shift": ["query", "queries", "search term", "search terms", "intent", "brand", "non-brand", "match type"],
    "auction_pressure": ["auction", "impression share", "is", "top is", "abs. top", "overlap", "outranking", "competition", "competitor"],
    "budget_limited": ["budget", "limited", "pacing", "lost is (budget)", "lost impression share (budget)"],
    "tracking_issue": ["tracking", "tag", "ga4", "gtm", "conversion", "attribution", "consent", "hubspot"],
    "landing_page_issue": ["landing page", "lp", "page speed", "load time", "site", "checkout", "form", "cvr drop"],
    "creative_fatigue": ["creative", "ad copy", "rsa", "assets", "ctr", "frequency", "fatigue"],
    "seasonality": ["seasonality", "seasonal", "holiday", "promo", "promotion", "demand"],
}

# Optional generic checks that are always safe (do NOT imply a cause)
GENERIC_SAFE_CHECKS = [
    "Verify the reporting date range and attribution settings are consistent with your baseline.",
]

def constrain_recommended_checks(checks: List[str], allowed_drivers: List[DriverType]) -> List[str]:
    """
    Enforce that checks map ONLY to allowed drivers, plus generic safe checks.
    If allowed_drivers is empty -> return checks as-is (they should be generic).
    """
    if not checks:
        return []

    allowed = [d for d in allowed_drivers if d != "unknown"]
    if not allowed:
        # No drivers allowed -> keep only generic safe checks (and at most 5)
        filtered = []
        for c in checks:
            if c.strip() in GENERIC_SAFE_CHECKS:
                filtered.append(c.strip())
        # If AI didn't include our generic safe check, add one
        if not filtered:
            filtered = GENERIC_SAFE_CHECKS[:]
        return filtered[:5]

    # Drivers allowed -> keep checks that match any allowed driver's keywords, plus generic safe checks
    keywords = []
    for d in allowed:
        keywords.extend(CHECK_KEYWORDS_BY_DRIVER.get(d, []))
    keywords = [k.lower() for k in keywords]

    filtered: List[str] = []
    for c in checks:
        c_clean = c.strip()
        c_lower = c_clean.lower()

        if c_clean in GENERIC_SAFE_CHECKS:
            filtered.append(c_clean)
            continue

        if any(k in c_lower for k in keywords):
            filtered.append(c_clean)

    # If filtering removed everything, fall back to generic safe check
    if not filtered:
        filtered = GENERIC_SAFE_CHECKS[:]

    return filtered[:5]

# -----------------------------
# AI call (bounded + constrained checks)
# -----------------------------
def call_ai_for_reasoning(payload_for_ai: dict, allowed_drivers: List[DriverType]) -> AIReasoning:
    """
    AI may ONLY choose drivers from allowed_drivers.
    recommended_checks must ONLY reference allowed drivers (no budget checks unless budget_limited is allowed, etc.)
    """

    allowed_list = [d for d in allowed_drivers if d != "unknown"]

    system_rules = (
        "You are a Google Ads analyst. Return ONLY valid JSON. No markdown. "
        "Do NOT calculate metrics; assume metrics provided are correct. "
        "CRITICAL: Do NOT invent causes.\n"
        "- If allowed_drivers is empty, drivers MUST be ['unknown'].\n"
        "- If allowed_drivers is not empty, drivers must be a subset of allowed_drivers.\n"
        "- recommended_checks MUST ONLY relate to allowed_drivers (do not mention unrelated areas). "
        "You may include 1 generic safe check about reporting/attribution consistency.\n"
        "Output schema: {summary: string, drivers: string[], recommended_checks: string[]}"
    )

    user_rules = (
        f"allowed_drivers = {allowed_list}\n\n"
        "Rules:\n"
        "- drivers must be from: "
        "[query_mix_shift, auction_pressure, budget_limited, tracking_issue, landing_page_issue, creative_fatigue, seasonality, unknown]\n"
        "- If allowed_drivers is empty, set drivers to ['unknown'].\n"
        "- If allowed_drivers is NOT empty, drivers must be a subset of allowed_drivers (no extra drivers).\n"
        "- recommended_checks MUST be verification steps and MUST ONLY relate to allowed_drivers.\n"
        "- Do NOT include checks about budget unless budget_limited is allowed.\n"
        "- Do NOT include checks about tracking unless tracking_issue is allowed.\n"
        "- Do NOT include landing page checks unless landing_page_issue is allowed.\n"
        "- Do NOT include creative checks unless creative_fatigue is allowed.\n"
        "- You may include 0–1 generic safe check about reporting/attribution consistency.\n"
        "- Return 3–5 recommended_checks.\n"
    )

    api_payload = {
        "model": "gpt-4.1-mini",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_rules},
            {
                "role": "user",
                "content": (
                    "Given this Ads snapshot and included evidence, write a concise summary, "
                    "select drivers according to the rules, and list constrained verification checks.\n\n"
                    f"{json.dumps(payload_for_ai, ensure_ascii=False)}\n\n"
                    f"{user_rules}"
                )
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(OPENAI_URL, headers=headers, json=api_payload, timeout=30)
    r.raise_for_status()

    raw = r.json()["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError("AI returned invalid JSON") from e

    reasoning = AIReasoning(**parsed)

    # Enforce driver constraints post-hoc (CODE GOVERNS)
    if not allowed_list:
        reasoning.drivers = ["unknown"]
    else:
        filtered_drivers = [d for d in reasoning.drivers if d in allowed_list]
        reasoning.drivers = filtered_drivers if filtered_drivers else ["unknown"]

    # Enforce check constraints post-hoc (CODE GOVERNS)
    reasoning.recommended_checks = constrain_recommended_checks(reasoning.recommended_checks, allowed_drivers)

    return reasoning

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/analyze")
def analyze(input: MetricsInput):
    try:
        metrics = compute_metrics(input.clicks, input.conversions, input.cost)

        # Deterministic alerts (code governs)
        cpa = metrics["cost_per_conversion"]
        alerts = {
            "zero_conversions": input.conversions == 0,
            "high_cpa": (cpa is not None and cpa > 50),
            "low_cvr": (metrics["conversion_rate"] < 2.0 and input.clicks >= 50),
        }

        allowed = infer_allowed_drivers(input.evidence)

        payload_for_ai: Dict[str, Any] = {
            "inputs": {
                "clicks": input.clicks,
                "conversions": input.conversions,
                "cost": input.cost,
            },
            "computed_metrics": metrics,
            "alerts": alerts,
            "evidence": (input.evidence.model_dump() if input.evidence is not None else {}),
        }

        reasoning = call_ai_for_reasoning(payload_for_ai, allowed)

        return {
            "status": "ok",
            "inputs": payload_for_ai["inputs"],
            "metrics": metrics,
            "alerts": alerts,
            "allowed_drivers": allowed,
            "reasoning": reasoning.model_dump(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
