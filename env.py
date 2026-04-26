"""
env.py — SalesOps Reinforcement Learning Environment
======================================================
Simulates a realistic CRM/SalesOps environment.
Exposes an OpenAI Gym-style interface: reset(), step(), state(), render_logs().
"""

from __future__ import annotations
import random
import uuid
from typing import Optional
from models import Lead, StepResult, RewardBreakdown
from reward import RewardEngine
import config


class SalesOpsEnvironment:
    """
    A stochastic SalesOps simulation environment.

    Each episode contains LEADS_PER_EPISODE leads.
    Agents take actions on each lead; the environment evaluates outcomes.
    """

    def __init__(self):
        self.reward_engine   = RewardEngine()
        self.episode_num     = 0
        self.lead_queue:     list[Lead] = []
        self.current_lead:   Optional[Lead] = None
        self.budget          = config.INITIAL_BUDGET
        self.step_count      = 0
        self.episode_log:    list[dict] = []
        self._all_logs:      list[dict] = []
        self.done            = False

    # ── Public Interface ──────────────────────────────────────────────────

    def reset(self, custom_leads: Optional[list[Lead]] = None) -> Lead:
        """Start a new episode. Returns first lead observation."""
        self.episode_num   += 1
        self.budget         = config.INITIAL_BUDGET
        self.step_count     = 0
        self.done           = False
        self.episode_log    = []
        
        if custom_leads:
            self.lead_queue = custom_leads.copy()
        else:
            self.lead_queue = self._generate_lead_batch(config.LEADS_PER_EPISODE)
            
        self.current_lead   = self.lead_queue.pop(0)
        return self.current_lead

    def step(self, action: str) -> StepResult:
        """
        Execute action on current lead.
        Returns: (next_state, reward_breakdown, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if action not in config.ACTIONS:
            raise ValueError(f"Invalid action: {action}. Choose from {config.ACTIONS}")

        lead = self.current_lead
        self.step_count += 1

        # ── Compute reward ─────────────────────────────────────────────────
        reward = self.reward_engine.compute(lead, action, self.budget)

        # ── Update budget ──────────────────────────────────────────────────
        cost_incurred = self._action_cost(lead, action)
        self.budget   = max(0.0, self.budget - cost_incurred)

        # ── Log decision ───────────────────────────────────────────────────
        log_entry = {
            "episode":        self.episode_num,
            "step":           self.step_count,
            "lead_id":        lead.lead_id,
            "industry":       lead.industry,
            "deal_value":     lead.deal_value,
            "risk_score":     lead.risk_score,
            "action":         action,
            "global_reward":  round(reward.global_reward, 4),
            "budget_remaining": round(self.budget, 2),
            "cost_incurred":  round(cost_incurred, 2),
        }
        self.episode_log.append(log_entry)
        self._all_logs.append(log_entry)

        # ── Advance to next lead ───────────────────────────────────────────
        if self.lead_queue:
            self.current_lead = self.lead_queue.pop(0)
            next_state = self.current_lead
        else:
            self.done         = True
            self.current_lead = None
            next_state        = None

        info = {
            "cost_incurred":    cost_incurred,
            "budget_remaining": self.budget,
            "lead_id":          lead.lead_id,
            "step":             self.step_count,
            "episode":          self.episode_num,
            "conversion":       self._is_conversion(action, lead),
            "risk_event":       action == "notify_compliance" or lead.risk_score > 0.75,
        }

        return StepResult(
            next_state=next_state,
            reward=reward,
            done=self.done,
            info=info,
        )

    def state(self) -> Optional[Lead]:
        """Return current lead (observation)."""
        return self.current_lead

    def render_logs(self, n: int = 10) -> list[dict]:
        """Return the last N decision log entries."""
        return self._all_logs[-n:]

    # ── Lead Generation ────────────────────────────────────────────────────

    def _generate_lead_batch(self, n: int) -> list[Lead]:
        return [self._generate_lead() for _ in range(n)]

    def _generate_lead(self) -> Lead:
        company_size  = random.choice(config.COMPANY_SIZES)
        deal_value    = self._deal_value_for_size(company_size)
        acq_cost      = deal_value * random.uniform(0.05, 0.45)
        risk_score    = round(random.betavariate(2, 5), 2)   # skewed toward low-risk
        compliance_flags = []
        if risk_score > 0.6:
            compliance_flags.append("gdpr_concern")
        if risk_score > 0.75:
            compliance_flags.append("high_churn_probability")
        if random.random() < 0.1:
            compliance_flags.append("blacklisted_region")

        return Lead(
            lead_id               = f"L-{uuid.uuid4().hex[:6].upper()}",
            company_size          = company_size,
            industry              = random.choice(config.INDUSTRIES),
            deal_value            = round(deal_value, 2),
            lead_score            = round(random.uniform(0.2, 1.0), 2),
            urgency               = round(random.uniform(0.0, 1.0), 2),
            acquisition_cost      = round(acq_cost, 2),
            risk_score            = risk_score,
            compliance_flags      = compliance_flags,
            time_decay            = round(random.uniform(0.01, 0.08), 3),
            previous_interactions = random.randint(0, 10),
            market_condition      = random.choice(config.MARKET_CONDITIONS),
            budget_remaining      = self.budget,
        )

    def _deal_value_for_size(self, size: str) -> float:
        ranges = {
            "startup":    (5_000,  30_000),
            "smb":        (20_000, 100_000),
            "mid-market": (80_000, 400_000),
            "enterprise": (300_000, 1_500_000),
        }
        lo, hi = ranges.get(size, (10_000, 100_000))
        return random.uniform(lo, hi)

    def _action_cost(self, lead: Lead, action: str) -> float:
        """Estimate operational cost for taking an action on this lead."""
        base_costs = {
            "pursue_lead":            lead.acquisition_cost * 1.0,
            "nurture_lead":           lead.acquisition_cost * 0.3,
            "reject_lead":            0.0,
            "escalate_to_enterprise": lead.acquisition_cost * 0.5,
            "offer_discount":         lead.deal_value * 0.10,
            "request_more_info":      lead.acquisition_cost * 0.05,
            "notify_compliance":      500.0,
            "schedule_demo":          lead.acquisition_cost * 0.15,
        }
        return base_costs.get(action, 0.0)

    def _is_conversion(self, action: str, lead: Lead) -> bool:
        conversion_actions = {"pursue_lead", "escalate_to_enterprise", "offer_discount", "schedule_demo"}
        if action not in conversion_actions:
            return False
        prob = lead.lead_score * (1.0 - lead.risk_score * 0.5)
        return random.random() < prob
