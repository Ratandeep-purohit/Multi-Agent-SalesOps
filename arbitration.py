"""
arbitration.py — Strategy Manager Arbitration Engine
======================================================
Resolves conflicts between agent recommendations.

Strategy:
  1. Ask Strategy Manager LLM (via HFClient) for final action — uses the
     structured agent_recommendations + lead_state prompt.
  2. If LLM unavailable → fall back to weighted voting with risk/budget dampeners.

Returns a fully explainable ArbitrationResult.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from models import AgentRecommendation, ArbitrationResult
import config

if TYPE_CHECKING:
    from hf_client import HFClient


# ── Action scoring ────────────────────────────────────────────────────────────
HIGH_RISK_ACTIONS = {"pursue_lead", "escalate_to_enterprise", "offer_discount"}
LOW_RISK_ACTIONS  = {"reject_lead", "notify_compliance", "request_more_info"}


def _action_score(action: str) -> float:
    """Map each action to an aggressiveness score (0=conservative, 1=aggressive)."""
    scores = {
        "pursue_lead":            1.0,
        "escalate_to_enterprise": 0.9,
        "offer_discount":         0.8,
        "schedule_demo":          0.7,
        "nurture_lead":           0.5,
        "request_more_info":      0.3,
        "notify_compliance":      0.1,
        "reject_lead":            0.0,
    }
    return scores.get(action, 0.5)


class ArbitrationEngine:
    """
    Hybrid arbitration:
      - Tries LLM-based Strategy Manager decision (via HFClient) first.
      - Falls back to weighted voting if API is unavailable.
    """

    def __init__(self, hf_client: Optional["HFClient"] = None):
        self.hf_client = hf_client

    def decide(
        self,
        recommendations: dict[str, AgentRecommendation],
        risk_score:       float = 0.5,
        budget_ratio:     float = 1.0,
        lead_state:       Optional[dict] = None,
        memory_context:   Optional[list] = None,
    ) -> ArbitrationResult:

        votes:    dict[str, str]   = {rec.agent: rec.recommended_action for rec in recommendations.values()}
        conf_map: dict[str, float] = {rec.agent: rec.confidence          for rec in recommendations.values()}
        conflict  = len(set(votes.values())) > 1

        # ── Try LLM-backed Strategy Manager ───────────────────────────────
        if self.hf_client and lead_state:
            rec_payload = {
                rec.agent: {
                    "recommended_action": rec.recommended_action,
                    "confidence":         rec.confidence,
                    "reason":             rec.reason,
                }
                for rec in recommendations.values()
            }
            try:
                llm_result = self.hf_client.strategy_decide(
                    lead_state            = lead_state,
                    agent_recommendations = rec_payload,
                    memory_context        = memory_context or [],
                    actions               = config.ACTIONS,
                )
                final_action = llm_result.get("final_action", "")
                if final_action in config.ACTIONS:
                    return ArbitrationResult(
                        final_action      = final_action,
                        reason            = llm_result.get("reason", "LLM decision."),
                        votes             = votes,
                        confidence_map    = conf_map,
                        conflict_detected = llm_result.get("conflict_detected", conflict),
                        weighted_score    = {},   # not applicable for LLM path
                    )
            except Exception:
                pass   # fall through to weighted voting

        # ── Fallback: Weighted voting ──────────────────────────────────────
        scores: dict[str, float] = {}
        for agent_key, rec in recommendations.items():
            base_weight    = config.ARBITRATION_WEIGHTS.get(agent_key, 0.25)
            aggressiveness = _action_score(rec.recommended_action)
            risk_dampener  = 1.0 - (risk_score * 0.4 * aggressiveness)
            budget_damper  = 1.0 - ((1.0 - budget_ratio) * 0.3 * aggressiveness)
            dampener       = max(0.2, risk_dampener * budget_damper)
            vote_score     = rec.confidence * base_weight * dampener
            scores[rec.recommended_action] = scores.get(rec.recommended_action, 0.0) + vote_score

        final_action = max(scores, key=scores.get)
        reason = self._explain(final_action, scores, votes, risk_score, budget_ratio, conflict)

        return ArbitrationResult(
            final_action      = final_action,
            reason            = reason,
            votes             = votes,
            confidence_map    = conf_map,
            conflict_detected = conflict,
            weighted_score    = {k: round(v, 4) for k, v in scores.items()},
        )

    def _explain(
        self,
        final_action:      str,
        scores:            dict[str, float],
        votes:             dict[str, str],
        risk_score:        float,
        budget_ratio:      float,
        conflict_detected: bool,
    ) -> str:
        top_score   = scores.get(final_action, 0)
        all_scores  = ", ".join(
            f"{a}: {s:.3f}" for a, s in sorted(scores.items(), key=lambda x: -x[1])
        )
        agent_votes = ", ".join(f"{a} -> {v}" for a, v in votes.items())

        if not conflict_detected:
            return (
                f"All agents agreed on '{final_action}'. "
                f"Unanimous (score: {top_score:.3f}). "
                f"Risk={risk_score:.2f}, Budget ratio={budget_ratio:.2f}."
            )

        context_note = ""
        if risk_score >= 0.65:
            context_note += f" High risk ({risk_score:.2f}) dampened aggressive actions."
        if budget_ratio < 0.40:
            context_note += f" Low budget ({budget_ratio:.2f}) penalized expensive moves."

        return (
            f"Conflict detected. Votes: [{agent_votes}]. "
            f"Weighted scores: [{all_scores}]. "
            f"'{final_action}' wins ({top_score:.3f}).{context_note}"
        )

