"""
AI-Powered Retention Insights
==============================
Uses the Anthropic Claude API to generate natural-language retention
recommendations based on a customer's churn risk profile.
"""

import json
import os

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


SYSTEM_PROMPT = """You are a senior Customer Retention Strategist at a UK car rental company.
Given a customer's profile and churn risk score, provide:
1. A plain-English risk summary (2-3 sentences)
2. Three specific, actionable retention recommendations ranked by expected impact
3. Estimated retention probability improvement if all recommendations are implemented

Keep language professional but concise. Use UK English spelling.
Format your response as JSON with keys: risk_summary, recommendations (list of objects
with action, rationale, expected_impact), and estimated_retention_lift_pct."""


def generate_ai_insights(customer_profile: dict, churn_probability: float) -> dict:
    """
    Generate AI-powered retention recommendations for a customer.

    Args:
        customer_profile: dict of customer features
        churn_probability: float 0-1 predicted churn probability

    Returns:
        dict with risk_summary, recommendations, estimated_retention_lift_pct
    """

    risk_level = "HIGH" if churn_probability > 0.7 else "MEDIUM" if churn_probability > 0.4 else "LOW"

    prompt = f"""Analyse this customer and provide retention recommendations.

Customer Profile:
- Age: {customer_profile.get('age')}
- Tenure: {customer_profile.get('tenure_months')} months
- Monthly Spend: £{customer_profile.get('monthly_spend_gbp', 0):.2f}
- Rentals (12 months): {customer_profile.get('num_rentals_12m')}
- Digital Engagement Score: {customer_profile.get('digital_engagement_score', 0):.3f}
- Pricing Satisfaction: {customer_profile.get('pricing_satisfaction')}/10
- Service Satisfaction: {customer_profile.get('service_satisfaction')}/10
- NPS Score: {customer_profile.get('nps_score')}
- Loyalty Member: {'Yes' if customer_profile.get('loyalty_member') else 'No'}
- Complaints (6 months): {customer_profile.get('complaints_last_6m')}
- Region: {customer_profile.get('region', 'Unknown')}

Predicted Churn Probability: {churn_probability:.1%}
Risk Level: {risk_level}

Respond with only the JSON object, no markdown formatting."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if HAS_ANTHROPIC and api_key:
        return _call_claude(prompt, api_key)
    else:
        return _rule_based_fallback(customer_profile, churn_probability, risk_level)


def _call_claude(prompt: str, api_key: str) -> dict:
    """Call Claude API for AI-generated insights."""
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _rule_based_fallback(profile: dict, prob: float, risk: str) -> dict:
    """Deterministic fallback when Claude API is unavailable."""
    recommendations = []

    if profile.get("age", 30) < 26:
        recommendations.append({
            "action": "Offer student/young driver discount programme with gamified loyalty tiers",
            "rationale": "18-25 age group shows 25% higher churn; price sensitivity is the primary driver",
            "expected_impact": "High"
        })

    if profile.get("digital_engagement_score", 0.5) < 0.25:
        recommendations.append({
            "action": "Send personalised app onboarding sequence with first-rental-free incentive",
            "rationale": "Low digital engagement correlates strongly with churn; app users retain 40% better",
            "expected_impact": "High"
        })

    if profile.get("pricing_satisfaction", 5) < 5:
        recommendations.append({
            "action": "Proactive outreach with tailored pricing plan based on rental frequency",
            "rationale": "Below-average pricing satisfaction is the #2 churn predictor after age",
            "expected_impact": "Medium"
        })

    if profile.get("complaints_last_6m", 0) >= 2:
        recommendations.append({
            "action": "Assign dedicated account manager and issue service recovery credit",
            "rationale": "Repeat complaints signal unresolved issues; personal attention reverses 60% of at-risk cases",
            "expected_impact": "High"
        })

    if not profile.get("loyalty_member"):
        recommendations.append({
            "action": "Invite to loyalty programme with immediate 500 bonus points",
            "rationale": "Loyalty members churn at half the rate of non-members",
            "expected_impact": "Medium"
        })

    if profile.get("nps_score", 0) < -20:
        recommendations.append({
            "action": "Trigger NPS detractor recovery workflow with manager callback within 24h",
            "rationale": "Detractors converted to promoters have 3x higher lifetime value",
            "expected_impact": "High"
        })

    # Take top 3
    recommendations = recommendations[:3]
    if len(recommendations) < 3:
        recommendations.append({
            "action": "Schedule quarterly check-in with personalised usage insights email",
            "rationale": "Regular touchpoints reduce churn by 12% on average",
            "expected_impact": "Low"
        })

    lift = min(25, int(prob * 35))

    return {
        "risk_summary": (
            f"This customer has a {risk} churn risk ({prob:.0%} probability). "
            f"Key risk factors include {'young age, ' if profile.get('age', 30) < 26 else ''}"
            f"{'low digital engagement, ' if profile.get('digital_engagement_score', 0.5) < 0.25 else ''}"
            f"{'pricing dissatisfaction, ' if profile.get('pricing_satisfaction', 5) < 5 else ''}"
            f"{'repeat complaints, ' if profile.get('complaints_last_6m', 0) >= 2 else ''}"
            f"which together suggest targeted intervention could meaningfully reduce churn risk."
        ),
        "recommendations": recommendations[:3],
        "estimated_retention_lift_pct": lift,
    }


if __name__ == "__main__":
    # Demo with sample customer
    sample = {
        "age": 22, "tenure_months": 4, "monthly_spend_gbp": 85.50,
        "num_rentals_12m": 3, "digital_engagement_score": 0.12,
        "pricing_satisfaction": 4, "service_satisfaction": 6,
        "nps_score": -35, "loyalty_member": 0, "complaints_last_6m": 2,
        "region": "London"
    }
    result = generate_ai_insights(sample, churn_probability=0.78)
    print(json.dumps(result, indent=2))
