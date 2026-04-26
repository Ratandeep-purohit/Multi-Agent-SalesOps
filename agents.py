"""
agents.py — Multi-Agent Definitions
=====================================
Defines Sales, Finance, Compliance, and Strategy Manager agents.
Each agent:
  - Has a distinct role and objective
  - Calls the LLM (with fallback)
  - Uses policy weights from memory
  - Returns a structured AgentRecommendation
"""

from __future__ import annotations
from models import Lead, AgentRecommendation
from hf_client import HFClient
from memory import ExperienceMemory
import config


def _lead_to_state(lead: Lead) -> dict:
    """Convert a Lead to a compact dict for LLM prompts."""
    return {
        "lead_id":               lead.lead_id,
        "company_size":          lead.company_size,
        "industry":              lead.industry,
        "deal_value":            lead.deal_value,
        "lead_score":            lead.lead_score,
        "urgency":               lead.urgency,
        "acquisition_cost":      lead.acquisition_cost,
        "risk_score":            lead.risk_score,
        "compliance_flags":      lead.compliance_flags,
        "time_decay":            lead.time_decay,
        "previous_interactions": lead.previous_interactions,
        "market_condition":      lead.market_condition,
        "budget_remaining":      lead.budget_remaining,
        "estimated_margin_pct":  round(
            (lead.deal_value - lead.acquisition_cost) / max(lead.deal_value, 1) * 100, 1
        ),
    }


class BaseAgent:
    """Shared logic for all agents."""

    name:      str = "Base Agent"
    role:      str = "Make good decisions."

    def __init__(self, client: HFClient, memory: ExperienceMemory):
        self.client = client
        self.memory = memory

    def recommend(self, lead: Lead) -> AgentRecommendation:
        state   = _lead_to_state(lead)
        bucket  = self.memory.bucket_for_lead(
            lead.deal_value, lead.risk_score, lead.urgency, lead.acquisition_cost
        )
        policy  = self.memory.get_policy(self.name)
        pw      = policy.get(bucket)
        best_pw = policy.best_action(bucket)

        # Retrieve similar past experiences to pass as context
        mem_ctx = self.memory.retrieve_similar(bucket, n=config.MEMORY_WINDOW)

        # Ask LLM (or fallback)
        raw = self.client.recommend(
            agent_name    = self.name,
            agent_role    = self.role,
            state_summary = state,
            actions       = config.ACTIONS,
            memory_context= mem_ctx,
        )

        action = raw.get("recommended_action", best_pw)
        if action not in config.ACTIONS:
            action = best_pw

        return AgentRecommendation(
            agent               = self.name,
            recommended_action  = action,
            confidence          = float(raw.get("confidence", 0.60)),
            reason              = raw.get("reason", "No reason provided."),
            context_bucket      = bucket,
            policy_weight_used  = round(pw.get(action, 1.0), 4),
        )


# ── Concrete Agents ────────────────────────────────────────────────────────

class SalesAgent(BaseAgent):
    name = "Sales Agent"
    role = (
        "Maximize revenue and conversions. Prioritize high-value, high-score leads. "
        "Be aggressive in pursuing deals that have strong upside potential. "
        "Accept moderate risk if the deal value justifies it."
    )


class FinanceAgent(BaseAgent):
    name = "Finance Agent"
    role = (
        "Minimize Customer Acquisition Cost (CAC) and protect profit margins. "
        "Reject leads with thin margins or excessive acquisition cost. "
        "Flag budget pressure early. Prefer conservative, high-margin opportunities."
    )


class ComplianceAgent(BaseAgent):
    name = "Compliance Agent"
    role = (
        "Reduce legal, ethical, and brand risk. Block or flag leads with high risk scores "
        "or active compliance flags (GDPR, blacklisted regions, churn risk). "
        "When in doubt, request more information or notify compliance."
    )


class StrategyManagerAgent(BaseAgent):
    name = "Strategy Manager"
    role = (
        "Resolve agent conflicts and optimize global business outcomes. "
        "Balance revenue generation, cost control, and risk management. "
        "Consider long-term customer lifetime value and strategic alignment. "
        "Your decision must be explainable and defensible to all stakeholders."
    )


# ── Agent Factory ─────────────────────────────────────────────────────────

def build_agents(client: HFClient, memory: ExperienceMemory) -> dict[str, BaseAgent]:
    """Return a dict of all agents, keyed by role name."""
    return {
        "sales":      SalesAgent(client, memory),
        "finance":    FinanceAgent(client, memory),
        "compliance": ComplianceAgent(client, memory),
        "strategy":   StrategyManagerAgent(client, memory),
    }
