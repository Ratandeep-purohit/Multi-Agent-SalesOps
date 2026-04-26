"""
models.py — Pydantic Data Models
==================================
All shared data structures used across the system.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Lead (Environment Observation) ───────────────────────────────────────
class Lead(BaseModel):
    lead_id:               str
    company_size:          str
    industry:              str
    deal_value:            float
    lead_score:            float   # 0.0–1.0
    urgency:               float   # 0.0–1.0
    acquisition_cost:      float
    risk_score:            float   # 0.0–1.0
    compliance_flags:      list[str]
    time_decay:            float   # penalty per step of delay (0.0–0.1)
    previous_interactions: int
    market_condition:      str
    budget_remaining:      float


# ── Agent Recommendation ─────────────────────────────────────────────────
class AgentRecommendation(BaseModel):
    agent:                str
    recommended_action:   str
    confidence:           float = Field(ge=0.0, le=1.0)
    reason:               str
    context_bucket:       Optional[str] = None
    policy_weight_used:   Optional[float] = None


# ── Arbitration Result ────────────────────────────────────────────────────
class ArbitrationResult(BaseModel):
    final_action:      str
    reason:            str
    votes:             dict[str, str]           # agent_name → recommended action
    confidence_map:    dict[str, float]         # agent_name → confidence
    conflict_detected: bool
    weighted_score:    dict[str, float]         # action → weighted vote score


# ── Reward Breakdown ──────────────────────────────────────────────────────
class RewardBreakdown(BaseModel):
    global_reward:      float
    sales_reward:       float
    finance_reward:     float
    compliance_reward:  float
    strategy_reward:    float
    breakdown: dict[str, float] = Field(default_factory=dict)
    # breakdown keys: revenue, cost_penalty, risk_penalty, speed_bonus, alignment_bonus


# ── Experience (Memory Entry) ─────────────────────────────────────────────
class Experience(BaseModel):
    episode:            int
    lead_id:            str
    context_bucket:     str
    state_summary:      dict[str, Any]
    recommendations:    dict[str, str]      # agent_name → action
    confidences:        dict[str, float]    # agent_name → confidence
    final_action:       str
    reward:             float
    reward_breakdown:   dict[str, float]
    outcome:            str                 # "positive" | "negative" | "neutral"
    explanation:        str
    policy_weights_snapshot: dict[str, dict[str, dict[str, float]]]  # agent → bucket → action → weight


# ── Step Result (env.step return) ─────────────────────────────────────────
class StepResult(BaseModel):
    next_state:    Optional[Lead]
    reward:        RewardBreakdown
    done:          bool
    info:          dict[str, Any] = Field(default_factory=dict)


# ── Episode Result ────────────────────────────────────────────────────────
class EpisodeResult(BaseModel):
    episode:             int
    total_reward:        float
    avg_reward:          float
    conversions:         int
    rejections:          int
    risk_incidents:      int
    budget_spent:        float
    budget_remaining:    float
    action_distribution: dict[str, int]
    experiences:         list[Experience]
