"""
reward.py — Multi-Signal Reward Engine
========================================
Computes global and per-agent reward signals given a lead and an action.
No external dependencies — pure business logic.
"""

from __future__ import annotations
from models import Lead, RewardBreakdown
import config


# ── Action impact tables ───────────────────────────────────────────────────
# Each entry: (conversion_prob_modifier, clv_multiplier, risk_modifier, speed_modifier)
ACTION_IMPACT: dict[str, dict] = {
    "pursue_lead": {
        "conversion_prob": 0.80, "clv_mult": 1.0,
        "risk_mod": 0.0,  "speed": 1.0,  "alignment": 0.8,
    },
    "nurture_lead": {
        "conversion_prob": 0.40, "clv_mult": 1.1,
        "risk_mod": -0.05, "speed": 0.4, "alignment": 0.5,
    },
    "reject_lead": {
        "conversion_prob": 0.0,  "clv_mult": 0.0,
        "risk_mod": -0.2,  "speed": 0.8, "alignment": 0.3,
    },
    "escalate_to_enterprise": {
        "conversion_prob": 0.65, "clv_mult": 1.8,
        "risk_mod": 0.05,  "speed": 0.6, "alignment": 0.9,
    },
    "offer_discount": {
        "conversion_prob": 0.70, "clv_mult": 0.75,
        "risk_mod": 0.0,  "speed": 0.9,  "alignment": 0.4,
    },
    "request_more_info": {
        "conversion_prob": 0.30, "clv_mult": 1.05,
        "risk_mod": -0.1,  "speed": 0.3, "alignment": 0.6,
    },
    "notify_compliance": {
        "conversion_prob": 0.0,  "clv_mult": 0.0,
        "risk_mod": -0.4,  "speed": 0.5, "alignment": 1.0,
    },
    "schedule_demo": {
        "conversion_prob": 0.55, "clv_mult": 1.2,
        "risk_mod": -0.05, "speed": 0.7, "alignment": 0.7,
    },
}

NORMALISE = 10_000.0   # scale factor to keep rewards in human-readable range


class RewardEngine:
    """
    Computes a decomposed, multi-signal reward.

    global_reward = revenue_component - cost_component - risk_penalty
                  + speed_bonus + alignment_bonus
    """

    def compute(self, lead: Lead, action: str, budget_remaining: float) -> RewardBreakdown:
        impact = ACTION_IMPACT.get(action, ACTION_IMPACT["request_more_info"])

        # ── 1. Revenue component ───────────────────────────────────────────
        effective_conv   = impact["conversion_prob"] * lead.lead_score
        clv              = lead.deal_value * impact["clv_mult"]
        revenue          = round((effective_conv * clv) / NORMALISE, 4)

        # ── 2. Cost component (Dynamic Economy Shaping) ────────────────────
        budget_ratio     = max(0.001, budget_remaining / config.INITIAL_BUDGET) # avoid zero div
        # Exponential cost scaling: cost is heavily penalized as budget approaches zero
        if budget_ratio < 0.2:
            budget_pressure = 1.0 + (1.0 / budget_ratio)**1.5
        else:
            budget_pressure = 1.0 + (1.0 - budget_ratio)
            
        cost_of_action   = self._cost(lead, action)
        cost_component   = round((cost_of_action * budget_pressure) / NORMALISE, 4)

        # ── 3. Risk penalty (Economy-Aware Risk Shaping) ───────────────────
        effective_risk   = max(0.0, lead.risk_score + impact["risk_mod"])
        compliance_hits  = len(lead.compliance_flags) * 0.05
        
        # Dynamic Risk Shaping: Companies near bankruptcy cannot afford ANY risk
        risk_economy_multiplier = 1.0 + (1.0 / budget_ratio) if budget_ratio < 0.3 else 1.0
        
        risk_penalty     = round(effective_risk * (1.0 + compliance_hits) * 2.0 * risk_economy_multiplier, 4)

        # ── 4. Speed bonus ─────────────────────────────────────────────────
        urgency_match    = lead.urgency * impact["speed"]
        time_cost        = lead.time_decay * (1.0 - impact["speed"])
        speed_bonus      = round((urgency_match - time_cost) * 0.5, 4)

        # ── 5. Alignment bonus ─────────────────────────────────────────────
        alignment_bonus  = round(impact["alignment"] * 0.3, 4)

        # ── Global reward ──────────────────────────────────────────────────
        global_reward = round(
            revenue - cost_component - risk_penalty + speed_bonus + alignment_bonus, 4
        )

        # ── Per-agent rewards ──────────────────────────────────────────────
        sales_reward      = round(revenue * 2.0 - cost_component * 0.3, 4)
        finance_reward    = round(-cost_component * 2.0 + revenue * 0.5, 4)
        compliance_reward = round(-risk_penalty * 2.0 + alignment_bonus, 4)
        strategy_reward   = round(global_reward * 1.2, 4)

        return RewardBreakdown(
            global_reward     = global_reward,
            sales_reward      = sales_reward,
            finance_reward    = finance_reward,
            compliance_reward = compliance_reward,
            strategy_reward   = strategy_reward,
            breakdown = {
                "revenue":         revenue,
                "cost_penalty":    -cost_component,
                "risk_penalty":    -risk_penalty,
                "speed_bonus":     speed_bonus,
                "alignment_bonus": alignment_bonus,
            },
        )

    def _cost(self, lead: Lead, action: str) -> float:
        costs = {
            "pursue_lead":            lead.acquisition_cost,
            "nurture_lead":           lead.acquisition_cost * 0.30,
            "reject_lead":            0.0,
            "escalate_to_enterprise": lead.acquisition_cost * 0.50,
            "offer_discount":         lead.deal_value * 0.10,
            "request_more_info":      lead.acquisition_cost * 0.05,
            "notify_compliance":      500.0,
            "schedule_demo":          lead.acquisition_cost * 0.15,
        }
        return costs.get(action, 0.0)
